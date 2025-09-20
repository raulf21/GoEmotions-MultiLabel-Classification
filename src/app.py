# app.py — self-contained Gradio demo for GoEmotions (BiLSTM + Attention)

import json
import pickle
import re
from typing import Optional, List

import numpy as np
import gradio as gr
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Flair imports 
from flair.models import TextClassifer
from flair.data import Sentence

# --- custom layer (must be registered with @keras.saving.register_keras_serializable in models.py) ---
from models import AdditiveAttentionPooling

# ---------------- paths ----------------
MODEL_PATH = "best_bilstm_model_final.keras"
TOKENIZER_PATH = "tokenizer.pkl"
CONFIG_PATH = "preprocess_config.json"
CLASS_THRESHOLDS_PATH = "per_class_thresholds.npy"  # optional
EMOJI_MAP_PATH = "emoji_map.json"                   # optional export you may have saved

# ------------- load artifacts ----------
print("[startup] Loading BiLSTm model…")
model = keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={
        "Custom>AdditiveAttentionPooling": AdditiveAttentionPooling,  # registered name
        "AdditiveAttentionPooling": AdditiveAttentionPooling,         # alias, just in case
    },
)
print("[startup] Model loaded.")

print("[startup] Loading tokenizer/config…")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)
MAX_LEN = int(cfg["MAX_LEN"])
print(f"[startup] MAX_LEN={MAX_LEN}")

try:
    per_class_thresh = np.load(CLASS_THRESHOLDS_PATH)
    print("[startup] Per-class thresholds loaded.")
except Exception:
    per_class_thresh = None
    print("[startup] No per-class thresholds; will use slider only.")

try:
    with open(EMOJI_MAP_PATH, "r") as f:
        EMOJI_MAP = json.load(f)  # e.g. {"😂": " face_with_tears_of_joy_emoji ", ...}
    print("[startup] Emoji map loaded.")
except Exception:
    EMOJI_MAP = None
    print("[startup] No emoji map; skipping emoji normalization.")


# Load flair
print("[startup] Loading Flair model…")
flair_model = TextClassifer.load("flair_mulit_label/final-model.pt")
print("[startup] Flair model loaded.")


# ---------------- labels & display emojis ----------------
LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude",
    "grief","joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"
]
EMOJI = {
    "admiration":"👏","amusement":"😂","anger":"😡","annoyance":"😒","approval":"✅","caring":"🤗",
    "confusion":"🤔","curiosity":"🧐","desire":"😍","disappointment":"😞","disapproval":"🚫","disgust":"🤮",
    "embarrassment":"😳","excitement":"🤩","fear":"😨","gratitude":"🙏","grief":"😢","joy":"😄",
    "love":"❤️","nervousness":"😬","optimism":"😊","pride":"😌","realization":"💡","relief":"😮‍💨",
    "remorse":"😔","sadness":"😢","surprise":"😮","neutral":"😐"
}

# ---------------- minimal, training-mirrored cleaner (no heavy deps at runtime) ----------------
_CONTRACTIONS = {
    r"\bI'm\b": "i 'm",
    r"\bit's\b": "it 's",
    r"\bcan't\b": "ca n't",
    r"\bdon't\b": "do n't",
    r"\bthey're\b": "they 're",
}
_NON_ASCII = re.compile(r"[^\x00-\x7F]+")
_REPEAT = re.compile(r"(.)\1{2,}")
_KEEP = re.compile(r"[^a-z'\s_#@]")

def clean_and_tokenize(text: str, emoji_map: Optional[dict]) -> List[str]:
    # 1) placeholders & html
    text = text.replace("[NAME]", " name_token ")
    text = re.sub(r"<[^>]+>", " ", text)

    # 2) contractions
    for patt, repl in _CONTRACTIONS.items():
        text = re.sub(patt, repl, text, flags=re.IGNORECASE)

    # 3) hashtags & handles
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"@\w+", " user_token ", text)

    # 4) emoji normalization (if available)
    if emoji_map:
        for emo, repl in emoji_map.items():
            text = text.replace(emo, repl)

    # 5) remove non-ascii, collapse repeats
    text = _NON_ASCII.sub(" ", text)
    text = _REPEAT.sub(r"\1\1", text)  # <-- NOTE: _REPEAT (single underscore), NOT __REPEAT

    # 6) lowercase & keep letters/_/'/space
    text = text.lower()
    text = _KEEP.sub(" ", text)

    # 7) tokenize (space split; OK for inference since tokenizer was fit on cleaned strings)
    return [t for t in text.split() if t]

def to_ids_from_tokens(tokens_list: List[List[str]]):
    texts = [" ".join(toks) for toks in tokens_list]
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")

# ---------------- Prediction functions ----------------
def predict_bilstm(text: str, use_class_thresh: bool, global_thresh: float):
    toks = clean_and_tokenize(text, EMOJI_MAP)
    X = to_ids_from_tokens([toks])
    probs = model.predict(X, verbose=0)[0]

    if use_class_thresh and per_class_thresh is not None:
        th = per_class_thresh
    else:
        th = np.full_like(probs, float(global_thresh), dtype=np.float32)

    idx = np.where(probs >= th)[0]
    kept = sorted([(LABELS[i], float(probs[i])) for i in idx], key=lambda x: x[1], reverse=True)

    if not kept:
        top3 = np.argsort(-probs)[:3]
        top3 = [(LABELS[i], float(probs[i])) for i in top3]
        return " · ".join(f"{EMOJI.get(lbl,'')} **{lbl}** ({p:.2f})" for lbl, p in top3)
    return " · ".join(f"{EMOJI.get(lbl,'')} **{lbl}** ({p:.2f})" for lbl, p in kept)

def predict_flair(text: str, threshold: float):
    sentence = Sentence(text)
    flair_model.predict(sentence)
    labels = [(l.value, l.score) for l in sentence.labels]
    kept = [(lbl, sc) for lbl, sc in labels if sc >= threshold]
    if not kept:
        top3 = sorted(labels, key=lambda x: x[1], reverse=True)[:3]
        return " · ".join(f"{EMOJI.get(lbl,'')} **{lbl}** ({p:.2f})" for lbl, p in top3)
    return " · ".join(f"{EMOJI.get(lbl,'')} **{lbl}** ({p:.2f})" for lbl, p in kept)

def predict_both(text: str, use_class_thresh: bool, global_thresh: float):
    if not text.strip():
        return "Type something above 👆", "Type something above 👆"

    bilstm_out = predict_bilstm(text, use_class_thresh, global_thresh)
    flair_out = predict_flair(text, threshold=global_thresh)

    return bilstm_out, flair_out

# ---------------- UI ----------------
with gr.Blocks(title="GoEmotions — BiLSTM + Flair") as demo:
    gr.Markdown("## 😄 GoEmotions — BiLSTM + Flair\nType a sentence, and see predictions from both models.")

    txt = gr.Textbox(label="Input text", placeholder="e.g., I'm so excited for tonight!! 😂", lines=3)
    with gr.Row():
        use_class = gr.Checkbox(label="Use per-class calibrated thresholds (BiLSTM only)", value=True)
        slider = gr.Slider(0.0, 1.0, value=0.4, step=0.01, label="Threshold")

    with gr.Row():
        bilstm_box = gr.Markdown(label="BiLSTM Prediction")
        flair_box = gr.Markdown(label="Flair Prediction")

    gr.Button("Predict").click(predict_both, inputs=[txt, use_class, slider], outputs=[bilstm_box, flair_box])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, debug=True)