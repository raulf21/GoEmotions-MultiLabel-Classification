# app.py — self-contained Gradio demo for GoEmotions (BiLSTM + Attention)

import json
import pickle
import re
from typing import Optional, List

import numpy as np
import gradio as gr
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- custom layer (must be registered with @keras.saving.register_keras_serializable in models.py) ---
from models import AdditiveAttentionPooling

# ---------------- paths ----------------
MODEL_PATH = "best_bilstm_model_final.keras"
TOKENIZER_PATH = "tokenizer.pkl"
CONFIG_PATH = "preprocess_config.json"
CLASS_THRESHOLDS_PATH = "per_class_thresholds.npy"  # optional
EMOJI_MAP_PATH = "emoji_map.json"                   # optional export you may have saved

# ------------- load artifacts ----------
print("[startup] Loading model…")
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

# ---------------- prediction ----------------
def predict(text: str, use_class_thresh: bool, global_thresh: float):
    if not text or not text.strip():
        return "Type something above 👆", ""

    toks = clean_and_tokenize(text, EMOJI_MAP)
    X = to_ids_from_tokens([toks])
    probs = model.predict(X, verbose=0)[0]  # shape (28,)

    # choose thresholds
    if use_class_thresh and per_class_thresh is not None and len(per_class_thresh) == len(LABELS):
        th = per_class_thresh
    else:
        th = np.full_like(probs, float(global_thresh), dtype=np.float32)

    # multi-label: return ALL labels with prob >= threshold
    idx = np.where(probs >= th)[0]
    kept = sorted([(LABELS[i], float(probs[i])) for i in idx], key=lambda x: x[1], reverse=True)

    if not kept:
        # show top-3 for feedback
        topk = np.argsort(-probs)[:3]
        top3 = [(LABELS[i], float(probs[i])) for i in topk]
        strip_ = " ".join(EMOJI.get(lbl, "") for lbl, _ in top3)
        md = f"{strip_}\n\n_No labels ≥ threshold; top-3 scores:_\n\n" + " · ".join(
            f"{EMOJI.get(lbl,'')} **{lbl}** ({p:.2f})" for lbl, p in top3
        )
        table = "\n".join(f"{lbl}\t{p:.3f}" for lbl, p in top3)
        return md, table

    strip_ = " ".join(EMOJI.get(lbl, "") for lbl, _ in kept)
    md = f"{strip_}\n\n" + " · ".join(f"{EMOJI.get(lbl,'')} **{lbl}** ({p:.2f})" for lbl, p in kept)
    table = "\n".join(f"{lbl}\t{p:.3f}" for lbl, p in kept)
    return md, table

# ---------------- UI ----------------
with gr.Blocks(title="GoEmotions — BiLSTM + Attention") as demo:
    gr.Markdown("## 😄 GoEmotions — BiLSTM + Attention\nType a sentence, set the threshold, hit **Predict**.")
    txt = gr.Textbox(label="Input text", placeholder="e.g., I'm so excited for tonight!! 😂", lines=3)
    with gr.Row():
        use_class = gr.Checkbox(label="Use per-class calibrated thresholds (validation)", value=True)
        slider = gr.Slider(0.0, 1.0, value=0.4, step=0.01, label="Global threshold (used if unchecked)")
    out_md = gr.Markdown()
    out_txt = gr.Textbox(label="Label\tProb", interactive=False)
    gr.Button("Predict").click(predict, inputs=[txt, use_class, slider], outputs=[out_md, out_txt])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, debug=True)
