# GoEmotions-MultiLabel-Classification 😄

This project explores **fine-grained multi-label emotion classification** using the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions).  
Our goal was to replicate and extend Google’s original work by experimenting with multiple architectures and deployment approaches.

---

## 🚀 Features
- **Multi-Architecture Comparison**
  - **BiLSTM + Attention**: Custom additive attention with focal loss
  - **Flair Transformer**: sentence-transformers/all-MiniLM-L6-v2 embeddings
  - **DistilBERT Fine-tuning**: (In progress) Full transformer fine-tuning

- **Multi-Granularity Training**
  - **Fine-grained**: 28 emotion categories
  - **Ekman**: 6 basic emotion groups  
  - **Sentiment**: 3-class positive/negative/neutral 

- **Interactive Demo (Gradio App)**  
  - Input any text → returns predicted emotions as emojis + probabilities.  
  - Adjustable threshold slider (global or per-class).  
  - Example:  
    > _"OMG, yep!!! That is the final answer! Thank you so much!"_  
    → 🎉 excitement · 🙏 gratitude · ✅ approval  
---

## 📂 Repository Structure
```
Go-Emotions/
│── notebooks/                # Exploratory data analysis + modeling notebooks
│── src/
│   ├── train_biLSTM.py       # Training script (BiLSTM + Attention + focal loss)
│   ├── flair_classifier.py      # Flair transformer training
│   ├── models.py             # Model architectures & custom layers
│   ├── data_preprocessing.py # Preprocessing pipeline (tokenizer, cleaning)
│   ├── inference.py          # Simple CLI inference for quick testing
│   ├── app.py                # Gradio web app for interactive predictions
│   ├── tokenizer.pkl         # Saved tokenizer (generated during training)
│   ├── preprocess_config.json# Config (e.g., MAX_LEN, vocab size)
│── requirements.txt
│── README.md
```

---

## 🛠️ Installation & Setup
```bash
# Clone repository
git clone https://github.com/raulf21/GoEmotions-MultiLabel-Classification.git
cd GoEmotions-MultiLabel-Classification

# Create conda environment
conda create -n nlp python=3.9
conda activate nlp

# Install dependencies
pip install -r requirements.txt

# Download GloVe embeddings (for BiLSTM)
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
```

---

## ▶️ Usage

### Train the BiLSTM Model
```bash
cd src
python train_biLSTM.py
# Train Flair transformer models  
python flair_classifier.py

# Fine-tune DistilBERT (coming soon)
python train_distilbert.py
```

### Run Inference (CLI)
```bash
cd src
python inference.py
```

### Launch Gradio App
```bash
cd src
python app.py
```
This will open a local browser window where you can try the model interactively.

---

## 🎥 Demo
Here’s a short GIF demo of the app in action:  

## 🎥 Demo
Here’s a short GIF demo of the app in action:  

![GoEmotions Demo 1](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExcWhxMnVyanppM210NTZtamRneTE3Z2x6bzM2aG9nZ2llYTBqNGx1dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/b2Nfm9ZF9VUdJ6qaYf/giphy.gif)

![GoEmotions Demo 2](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExdDYzd2w1aWE0cHFqamJlcTBid2gyZmtnN2h1b2t4amc2ejJpbmM0aiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/wcjhX0zRpVKghwlG7Y/giphy.gif)

![GoEmotions Demo 3](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExYXNlbmlmYTY4ZWI1aWZrZ2h4ODg4NGM5bHp3Y3RrNDQ1emJpcDM5cyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/wMpxPJeesUibSl9LtR/giphy.gif)


### Key Findings

#### **1. Architecture Trade-offs Revealed**
- **BiLSTM wins on fine-grained emotions** (28 classes): Better handling of rare emotions
- **Flair wins on coarser granularities** (6, 3 classes): Benefits from contextual understanding
- **Efficiency vs Performance**: BiLSTM is 5-10x faster with competitive results

#### **2. Class Imbalance Handling**
BiLSTM's focal loss approach proves superior for handling severe class imbalance:
- **Fine-grained Flair problems**: Complete failure on 5 rare emotions (0.00 F1)
- **BiLSTM consistency**: Better recall across all emotion categories
- **Precision-Recall trade-off**: Flair too conservative, BiLSTM more balanced

#### **3. Computational Efficiency**
| Model | Training Speed | Inference Speed | Memory Usage |
|-------|---------------|-----------------|--------------|
| BiLSTM + Attention | Fast (3-4 min) | Very Fast | Low |
| Flair Transformer | Slow (13-20 min) | Moderate | High |
| DistilBERT | TBD | TBD | TBD |

### Detailed Performance Breakdown

#### Fine-Grained Results (28 Emotions)

**BiLSTM + Attention:**
- Macro F1: 0.4651 | Micro F1: 0.5740
- Macro Precision: 0.4998 | Macro Recall: 0.4583
- **Strengths**: Balanced precision-recall, handles rare emotions

**Flair Transformer:**
- Macro F1: 0.3649 | Micro F1: 0.5616  
- Macro Precision: 0.5661 | Macro Recall: 0.3122
- **Issue**: High precision but very low recall (overly conservative)

### Comparison to Published Baselines

| Model | Architecture | Fine-grained F1 | Status |
|-------|-------------|-----------------|---------|
| Google BERT (Original) | BERT-base | ~0.46 | Published baseline |
| Google BiLSTM (Original) | Standard BiLSTM | ~0.41 | Published baseline |
| **Our BiLSTM + Attention** | Enhanced BiLSTM | **0.4651** | **Exceeds both** |
| Our Flair Transformer | sentence-transformers | 0.3649 | Below baseline |
---

---

## 🧪 Experimental Methodology

### Data Integrity
- **No data leakage**: Test set completely held out until final evaluation
- **Consistent splits**: Same train/val/test across all models
- **Rigorous CV**: 3-fold cross-validation for hyperparameter selection

### Evaluation Protocol  
- **Primary metric**: Macro F1-score (handles class imbalance)
- **Fixed threshold**: 0.4 for all multilabel predictions
- **Comprehensive metrics**: Precision, recall, per-class performance

---

## 📈 Next Steps & Future Work

### **Immediate Next Steps:**
- [ ] **DistilBERT fine-tuning implementation**
  - Full transformer fine-tuning with class-balanced loss
  - Comparison with BiLSTM and Flair approaches
  - Performance vs computational cost analysis

- [ ] **Multi-model Gradio app enhancement**
  - Side-by-side comparison of all three architectures
  - Real-time performance and prediction confidence display
  - Interactive threshold adjustment per model

## 🙌 Acknowledgments

- **Dataset**: [GoEmotions by Google Research](https://github.com/google-research/google-research/tree/master/goemotions)
- **Embeddings**: [GloVe Twitter](https://nlp.stanford.edu/projects/glove/) | [sentence-transformers](https://www.sbert.net/)
- **Libraries**: TensorFlow/Keras, Scikit-learn, Flair, Transformers, Gradio

---

## 📄 Citation
```bibtex
@misc{goemotions-multiarch-2024,
  title={Multi-Architecture Emotion Classification: BiLSTM vs Transformers on GoEmotions},
  author={[Your Name]},
  year={2024},
  url={https://github.com/raulf21/GoEmotions-MultiLabel-Classification}
}
```