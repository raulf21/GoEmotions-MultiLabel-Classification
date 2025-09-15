import pickle, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models import AdditiveAttentionPooling  # import registers the class too

# Load tokenizer + configs
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("preprocess_config.json", "r") as f:
    config = json.load(f)

MAX_LEN = config["MAX_LEN"]

# Load thresholds
per_class_thresh = np.load("per_class_thresholds.npy")

# Load model
model = tf.keras.models.load_model(
    "best_bilstm_model_final.keras",
    custom_objects={"AdditiveAttentionPooling": AdditiveAttentionPooling},
    compile=False
)
# Example preprocessing function (mirror your data_preprocessing.py)
def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

# Predict
def predict_labels(texts):
    X = np.vstack([preprocess_text(t) for t in texts])
    probs = model.predict(X, verbose=0)
    preds = (probs >= per_class_thresh).astype(int)
    return preds, probs

# --- Example ---
texts = ["I am so happy and excited!", "This is really sad news."]
preds, probs = predict_labels(texts)

print("Probabilities:", probs)
print("Predictions:", preds)
