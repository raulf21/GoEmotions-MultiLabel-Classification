# app.py — Multi-Model GoEmotions Emotion Classifier
"""
Gradio app for emotion classification using:
- BiLSTM + Attention
- Flair
- DistilBERT

All models use consistent threshold (default 0.4) for fair comparison.
Supports 3 granularities: Fine (28), Ekman (6), Sentiment (3)
"""

import json
import pickle
import os
from typing import Optional, List, Tuple

import numpy as np
import gradio as gr
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Flair imports 
from flair.models import TextClassifier
from flair.data import Sentence

# Transformers imports
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Custom layer
from models import AdditiveAttentionPooling

# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================

# Model paths for each granularity
BILSTM_PATHS = {
    "fine": "best_bilstm_fine.keras",
    "ekman": "best_bilstm_ekman.keras",
    "sentiment": "best_bilstm_sentiment.keras"
}

FLAIR_PATHS = {
    "fine": "flair_models/fine_model/final-model.pt",
    "ekman": "flair_models/ekman_model/final-model.pt",
    "sentiment": "flair_models/sentiment_model/final-model.pt"
}

DISTILBERT_PATHS = {
    "fine": "distilbert_fine_final",
    "ekman": "distilbert_ekman_final",
    "sentiment": "distilbert_sentiment_final"
}

# Preprocessing artifacts for BiLSTM
TOKENIZER_PATH = "tokenizer_{granularity}.pkl"
CONFIG_PATH = "preprocess_config_{granularity}.json"

# Default threshold (aligned across all models)
DEFAULT_THRESHOLD = 0.4

# ============================================================================
# LABEL DEFINITIONS
# ============================================================================

LABELS_FINE = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral"
]

LABELS_EKMAN = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

LABELS_SENTIMENT = ["positive", "negative", "neutral"]

LABELS_MAP = {
    "fine": LABELS_FINE,
    "ekman": LABELS_EKMAN,
    "sentiment": LABELS_SENTIMENT
}

# Emoji mappings for display
EMOJI_FINE = {
    "admiration":"👏", "amusement":"😂", "anger":"😡", "annoyance":"😒", 
    "approval":"✅", "caring":"🤗", "confusion":"🤔", "curiosity":"🧐", 
    "desire":"😍", "disappointment":"😞", "disapproval":"🚫", "disgust":"🤮",
    "embarrassment":"😳", "excitement":"🤩", "fear":"😨", "gratitude":"🙏", 
    "grief":"😢", "joy":"😄", "love":"❤️", "nervousness":"😬", 
    "optimism":"😊", "pride":"😌", "realization":"💡", "relief":"😮‍💨",
    "remorse":"😔", "sadness":"😢", "surprise":"😮", "neutral":"😐"
}

EMOJI_EKMAN = {
    "anger":"😡", "disgust":"🤮", "fear":"😨", 
    "joy":"😄", "sadness":"😢", "surprise":"😮"
}

EMOJI_SENTIMENT = {
    "positive":"😊", "negative":"😞", "neutral":"😐"
}

EMOJI_MAP = {
    "fine": EMOJI_FINE,
    "ekman": EMOJI_EKMAN,
    "sentiment": EMOJI_SENTIMENT
}

# ============================================================================
# MODEL LOADING
# ============================================================================

class ModelContainer:
    """Container for all models and their artifacts"""
    def __init__(self):
        self.bilstm_models = {}
        self.flair_models = {}
        self.distilbert_models = {}
        self.distilbert_tokenizers = {}
        self.bilstm_tokenizers = {}
        self.bilstm_max_lens = {}
        self.errors = []
    
    def load_all(self):
        """Load all models for all granularities"""
        for granularity in ["fine", "ekman", "sentiment"]:
            self._load_bilstm(granularity)
            self._load_flair(granularity)
            self._load_distilbert(granularity)
    
    def _load_bilstm(self, granularity):
        """Load BiLSTM model and tokenizer"""
        try:
            model_path = BILSTM_PATHS[granularity]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"BiLSTM model not found: {model_path}")
            
            # Load model
            model = keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={
                    "Custom>AdditiveAttentionPooling": AdditiveAttentionPooling,
                    "AdditiveAttentionPooling": AdditiveAttentionPooling,
                },
            )
            self.bilstm_models[granularity] = model
            
            # Load tokenizer and config
            tokenizer_path = TOKENIZER_PATH.format(granularity=granularity)
            config_path = CONFIG_PATH.format(granularity=granularity)
            
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config not found: {config_path}")
            
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)
            with open(config_path, "r") as f:
                cfg = json.load(f)
            
            self.bilstm_tokenizers[granularity] = tokenizer
            self.bilstm_max_lens[granularity] = int(cfg["max_len"])
            
            print(f"✓ BiLSTM ({granularity}) loaded successfully")
            
        except Exception as e:
            error_msg = f"❌ BiLSTM ({granularity}): {str(e)}"
            self.errors.append(error_msg)
            print(error_msg)
    
    def _load_flair(self, granularity):
        """Load Flair model"""
        try:
            model_path = FLAIR_PATHS[granularity]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Flair model not found: {model_path}")
            
            model = TextClassifier.load(model_path)
            self.flair_models[granularity] = model
            
            print(f"✓ Flair ({granularity}) loaded successfully")
            
        except Exception as e:
            error_msg = f"❌ Flair ({granularity}): {str(e)}"
            self.errors.append(error_msg)
            print(error_msg)
    
    def _load_distilbert(self, granularity):
        """Load DistilBERT model and tokenizer"""
        try:
            model_path = DISTILBERT_PATHS[granularity]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"DistilBERT model not found: {model_path}")
            
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            model = DistilBertForSequenceClassification.from_pretrained(model_path)
            model.eval()
            
            self.distilbert_tokenizers[granularity] = tokenizer
            self.distilbert_models[granularity] = model
            
            print(f"✓ DistilBERT ({granularity}) loaded successfully")
            
        except Exception as e:
            error_msg = f"❌ DistilBERT ({granularity}): {str(e)}"
            self.errors.append(error_msg)
            print(error_msg)

# Initialize models
print("="*70)
print("LOADING MODELS...")
print("="*70)
models = ModelContainer()
models.load_all()

if models.errors:
    print("\n" + "="*70)
    print("⚠️  ERRORS DURING MODEL LOADING:")
    print("="*70)
    for error in models.errors:
        print(error)
    print("\nSome models may not be available in the app.")
    print("="*70)
else:
    print("\n" + "="*70)
    print("✅ ALL MODELS LOADED SUCCESSFULLY!")
    print("="*70)

# ============================================================================
# TEXT PREPROCESSING (from data_preprocessing.py)
# ============================================================================

import re

def clean_text_simple(text: str) -> str:
    """
    Simplified text cleaning for inference (mimics training preprocessing)
    """
    # Basic cleaning
    text = text.replace("[NAME]", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Common contractions
    text = re.sub(r"\bI'm\b", "I 'm", text, flags=re.IGNORECASE)
    text = re.sub(r"\bit's\b", "it 's", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcan't\b", "ca n't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdon't\b", "do n't", text, flags=re.IGNORECASE)
    
    # URLs and handles
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    
    # Remove non-ASCII
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    
    # Collapse repeated characters (e.g., "sooooo" -> "soo")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    
    # Lowercase
    text = text.lower()
    
    # Keep only letters, numbers, spaces, apostrophes
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    return text

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_bilstm(text: str, granularity: str, threshold: float) -> str:
    """Predict using BiLSTM model"""
    if granularity not in models.bilstm_models:
        return f"❌ BiLSTM ({granularity}) model not available"
    
    try:
        # Preprocess
        cleaned = clean_text_simple(text)
        
        # Tokenize
        tokenizer = models.bilstm_tokenizers[granularity]
        max_len = models.bilstm_max_lens[granularity]
        
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        
        # Predict
        model = models.bilstm_models[granularity]
        probs = model.predict(padded, verbose=0)[0]
        
        # Apply threshold
        labels = LABELS_MAP[granularity]
        emojis = EMOJI_MAP[granularity]
        
        predictions = [(labels[i], float(probs[i])) for i in range(len(labels)) if probs[i] >= threshold]
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        if not predictions:
            # Show top 3 if nothing passes threshold
            top3_idx = np.argsort(-probs)[:3]
            predictions = [(labels[i], float(probs[i])) for i in top3_idx]
            result = "⚠️ No emotions above threshold. Top 3:\n\n"
        else:
            result = ""
        
        # Format output
        output = result + "\n".join(
            f"{emojis.get(lbl, '')} **{lbl}**: {score:.3f}" 
            for lbl, score in predictions
        )
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


def predict_flair(text: str, granularity: str, threshold: float) -> str:
    """Predict using Flair model"""
    if granularity not in models.flair_models:
        return f"❌ Flair ({granularity}) model not available"
    
    try:
        sentence = Sentence(text)
        model = models.flair_models[granularity]
        model.predict(sentence)
        
        # Apply threshold (aligned with other models)
        emojis = EMOJI_MAP[granularity]
        
        predictions = [(label.value, label.score) for label in sentence.labels if label.score >= threshold]
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        if not predictions:
            # ✨ FIXED: Show whatever Flair predicted (might be < 3)
            all_preds = [(label.value, label.score) for label in sentence.labels]
            all_preds.sort(key=lambda x: x[1], reverse=True)
            predictions = all_preds  # Don't limit to [:3] since Flair might return fewer
            
            if predictions:
                result = f"⚠️ No emotions above threshold. Showing all {len(predictions)} prediction(s):\n\n"
            else:
                # Edge case: Flair predicted nothing at all
                return "⚠️ Flair did not predict any emotions for this text."
        else:
            result = ""
        
        # Format output
        output = result + "\n".join(
            f"{emojis.get(lbl, '')} **{lbl}**: {score:.3f}" 
            for lbl, score in predictions
        )
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


def predict_distilbert(text: str, granularity: str, threshold: float) -> str:
    """Predict using DistilBERT model"""
    if granularity not in models.distilbert_models:
        return f"❌ DistilBERT ({granularity}) model not available"
    
    try:
        tokenizer = models.distilbert_tokenizers[granularity]
        model = models.distilbert_models[granularity]
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).numpy()[0]
        
        # Apply threshold
        labels = LABELS_MAP[granularity]
        emojis = EMOJI_MAP[granularity]
        
        predictions = [(labels[i], float(probs[i])) for i in range(len(labels)) if probs[i] >= threshold]
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        if not predictions:
            # Show top 3 if nothing passes threshold
            top3_idx = np.argsort(-probs)[:3]
            predictions = [(labels[i], float(probs[i])) for i in top3_idx]
            result = "⚠️ No emotions above threshold. Top 3:\n\n"
        else:
            result = ""
        
        # Format output
        output = result + "\n".join(
            f"{emojis.get(lbl, '')} **{lbl}**: {score:.3f}" 
            for lbl, score in predictions
        )
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


def predict_all(text: str, granularity: str, threshold: float) -> Tuple[str, str, str]:
    """Run all three models and return results"""
    if not text.strip():
        empty_msg = "👆 Type something above to see predictions"
        return empty_msg, empty_msg, empty_msg
    
    bilstm_result = predict_bilstm(text, granularity, threshold)
    flair_result = predict_flair(text, granularity, threshold)
    distilbert_result = predict_distilbert(text, granularity, threshold)
    
    return bilstm_result, flair_result, distilbert_result


# ============================================================================
# GRADIO UI
# ============================================================================

with gr.Blocks(title="GoEmotions Multi-Model Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎭 GoEmotions: Multi-Model Emotion Classifier
        
        Compare predictions from **three different models** on the same text:
        - **BiLSTM + Attention**: LSTM with attention mechanism and GloVe embeddings
        - **Flair**: Transformer-based document embeddings
        - **DistilBERT**: Fine-tuned DistilBERT transformer
        
        All models use the **same threshold (default 0.4)** for fair comparison.
        """
    )
    
    # Display any loading errors
    if models.errors:
        with gr.Accordion("⚠️ Model Loading Errors (click to expand)", open=False):
            gr.Markdown("\n".join(f"- {error}" for error in models.errors))
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="e.g., I'm so excited for the concert tonight! 🎉",
                lines=4
            )
            
            with gr.Row():
                granularity_dropdown = gr.Dropdown(
                    choices=["fine", "ekman", "sentiment"],
                    value="fine",
                    label="Granularity Level",
                    info="fine=28 emotions, ekman=6 emotions, sentiment=3 categories"
                )
                
                threshold_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_THRESHOLD,
                    step=0.01,
                    label=f"Confidence Threshold (research default: {DEFAULT_THRESHOLD})",
                    info="Adjust to control prediction sensitivity"
                )
            
            predict_btn = gr.Button("🔮 Predict Emotions", variant="primary", size="lg")
    
    gr.Markdown("### Model Predictions")
    
    with gr.Row():
        bilstm_output = gr.Markdown(label="BiLSTM + Attention")
        flair_output = gr.Markdown(label="Flair")
        distilbert_output = gr.Markdown(label="DistilBERT")
    
    # Wire up the prediction
    predict_btn.click(
        fn=predict_all,
        inputs=[text_input, granularity_dropdown, threshold_slider],
        outputs=[bilstm_output, flair_output, distilbert_output]
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["I can't believe I got the job! This is the best day ever! 🎉", "fine", 0.4],
            ["This is absolutely disgusting and unacceptable.", "fine", 0.4],
            ["I'm not sure how to feel about this... it's complicated.", "ekman", 0.4],
            ["Great news everyone!", "sentiment", 0.4],
            ["I'm really disappointed with how things turned out.", "fine", 0.3],
        ],
        inputs=[text_input, granularity_dropdown, threshold_slider],
    )
    
    gr.Markdown(
        """
        ---
        **Note**: The threshold slider is a user preference control for exploration. 
        All models were evaluated at threshold=0.4 for research comparison.
        
        **Granularity Levels**:
        - **Fine** (28 emotions): Most detailed emotion categories
        - **Ekman** (6 emotions): anger, disgust, fear, joy, sadness, surprise
        - **Sentiment** (3 categories): positive, negative, neutral
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )