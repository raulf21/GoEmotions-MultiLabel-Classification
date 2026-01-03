# GoEmotions-MultiLabel-Classification 😄

This project explores **fine-grained multi-label emotion classification** using the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions).  
Our goal was to replicate and extend Google's original work by experimenting with multiple architectures and deployment approaches.

---

## 🚀 Features
- **Multi-Architecture Comparison**
  - **BiLSTM + Attention**: Custom additive attention with focal loss
  - **Flair Transformer**: sentence-transformers/all-MiniLM-L6-v2 embeddings
  - **DistilBERT Fine-tuning**: ✅ Complete - Fine-tuned transformer with strong performance

- **Multi-Granularity Training**
  - **Fine-grained**: 28 emotion categories
  - **Ekman**: 6 basic emotion groups  
  - **Sentiment**: 3-class positive/negative/neutral 

- **Interactive Demo (Gradio App)**  
  - Input any text → returns predicted emotions as emojis + probabilities
  - Side-by-side comparison of all three models
  - Adjustable threshold slider
  - Example:  
    > _"OMG, yep!!! That is the final answer! Thank you so much!"_  
    → 🎉 excitement · 🙏 gratitude · ✅ approval  

---

## 📂 Repository Structure
```
Go-Emotions/
│── notebooks/                # Exploratory data analysis + modeling notebooks
│── src/
│   ├── train_biLSTM.py            # BiLSTM training script
│   ├── flair_classifier.py        # Flair transformer training
│   ├── train_distilbert.py        # DistilBERT fine-tuning
│   ├── models.py                  # Model architectures & custom layers
│   ├── data_preprocessing.py      # Preprocessing pipeline (type-safe)
│   ├── app.py                     # Gradio web app for interactive predictions
│   ├── tokenizer_*.pkl            # Saved tokenizers per granularity
│   ├── preprocess_config_*.json   # Config files per granularity
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

## 🧹 Preprocessing Approach

**Key Design Decision:** Minimal preprocessing optimized for GloVe Twitter embeddings.

### What We Do:
- ✅ **Preserve case** ("OMG" vs "omg" carry different emotional weight)
- ✅ **No lemmatization** (GloVe trained on raw tweets, not lemmatized text)
- ✅ **Keep contractions** ("I'm", "can't" preserve emotional tone)
- ✅ **Gentle normalization** (allow elongations like "soooo" up to 4 repetitions)
- ✅ **Smart emoji handling** (check GloVe vocabulary before converting to text)

### Why This Matters:
GloVe Twitter embeddings were trained on **2 billion raw tweets** with original casing, slang, and informal patterns. Heavy preprocessing (lemmatization, lowercasing) creates a train/test mismatch between our processed data and what GloVe "learned."

**Example:**
```
Input:  "OMG I'm SOOOOO happy!!! 😂"
Output: ['OMG', "I'm", 'SOOO', 'happy', '!!!', '😂']
```
Not: `['omg', 'be', 'so', 'happy', '!', 'face_with_tears_of_joy']` ❌

This Twitter-native approach maintains alignment with GloVe's training distribution, improving emotion detection performance.

---

## ▶️ Usage

### Train Models
```bash
cd src

# Train BiLSTM (all granularities)
python train_biLSTM.py

# Train Flair transformer
python flair_classifier.py

# Fine-tune DistilBERT
python train_distilbert.py
```

### Launch Gradio App
```bash
cd src
python app.py
```
Opens interactive demo at `http://127.0.0.1:7860` with side-by-side model comparison.

---

## 📊 Results

### Complete Model Comparison Across All Granularities

| Model | Granularity | Macro F1 | Micro F1 | Precision | Recall | Training Time | Winner |
|-------|-------------|----------|----------|-----------|--------|---------------|--------|
| BiLSTM | Fine (28) | 0.4295 | 0.5548 | 0.4843 | 0.4172 | 7.6 min | - |
| **DistilBERT** | Fine (28) | **0.4604** | **0.6001** | **0.6482** | **0.5587** | 77 min | 🥇 |
| Flair | Fine (28) | 0.3718 | 0.5538 | 0.5674 | 0.3216 | 20.7 min | - |
| BiLSTM | Ekman (6) | 0.5667 | 0.6495 | 0.5930 | 0.5795 | 6.9 min | - |
| **DistilBERT** | Ekman (6) | **0.6163** | **0.7086** | **0.6889** | **0.7295** | 78 min | 🥇 |
| Flair | Ekman (6) | 0.5864 | 0.6720 | 0.6074 | 0.6023 | 13.6 min | - |
| BiLSTM | Sentiment (3) | 0.6945 | 0.7037 | 0.6077 | **0.8123** | 8.1 min | - |
| **DistilBERT** | Sentiment (3) | **0.7271** | **0.7367** | **0.7180** | 0.7565 | 78 min | 🥇 |
| Flair | Sentiment (3) | 0.7212 | 0.7307 | 0.7051 | 0.7381 | 17.4 min | - |

**Key Takeaways:**
- 🥇 **DistilBERT dominates**: Best macro F1 across all granularities
- 🎯 **BiLSTM excels at recall**: Best recall on sentiment (0.8123)
- 💪 **BiLSTM most efficient**: 93% of DistilBERT's fine-grained F1 in 10% of training time

---

### Detailed Performance Breakdown

#### Fine-Grained Results (28 Emotions)

**DistilBERT (Fine-tuned):**
- Macro F1: **0.4604** | Micro F1: **0.6001**
- Macro Precision: **0.6482** | Macro Recall: **0.5587**
- **Strengths**: Best overall, highest precision, balanced performance

**BiLSTM + Attention:**
- Macro F1: 0.4295 | Micro F1: 0.5548
- Macro Precision: 0.4843 | Macro Recall: 0.4172
- **Strengths**: Fast training (7.6 min), good baseline

**Flair Transformer:**
- Macro F1: 0.3718 | Micro F1: 0.5538  
- Macro Precision: 0.5674 | Macro Recall: 0.3216
- Training time: 20.7 min (1239s)
- **Issue**: High precision but very low recall (overly conservative)

#### Ekman Results (6 Emotions)

**DistilBERT:**
- Macro F1: **0.6163** | Micro F1: **0.7086**
- Macro Precision: **0.6889** | Macro Recall: **0.7295**
- **Best performing model** for Ekman emotions

**BiLSTM + Attention:**
- Macro F1: 0.5667 | Micro F1: 0.6495
- Macro Precision: 0.5930 | Macro Recall: 0.5795
- **Strengths**: Fast training (6.9 min)

**Flair Transformer:**
- Macro F1: 0.5864 | Micro F1: 0.6720
- Macro Precision: 0.6074 | Macro Recall: 0.6023
- Training time: 13.6 min (813s)
- **Strengths**: Balanced precision-recall

#### Sentiment Results (3 Classes)

**DistilBERT:**
- Macro F1: **0.7271** | Micro F1: **0.7367**
- Macro Precision: **0.7180** | Macro Recall: 0.7565
- **Best performing model** for sentiment

**BiLSTM + Attention:**
- Macro F1: 0.6945 | Micro F1: 0.7037
- Macro Precision: 0.6077 | Macro Recall: **0.8123**
- **Strengths**: Exceptional recall, fast training (8.1 min)

**Flair Transformer:**
- Macro F1: 0.7212 | Micro F1: 0.7307
- Macro Precision: 0.7051 | Macro Recall: 0.7381
- Training time: 17.4 min (1046s)
- **Strengths**: Balanced performance across all metrics

---

### Comparison to Published Baselines

| Model | Architecture | Fine-grained F1 | Status |
|-------|-------------|-----------------|---------|
| Google BERT (Original) | BERT-base | ~0.46 | Published baseline |
| Google BiLSTM (Original) | Standard BiLSTM | ~0.41 | Published baseline |
| **Our DistilBERT** | DistilBERT-base-cased | **0.4604** | **Matches BERT** ✅ |
| **Our BiLSTM + Attention** | Enhanced BiLSTM | **0.4295** | **Exceeds standard BiLSTM** ✅ |
| Our Flair Transformer | sentence-transformers | 0.3718 | Below baseline |

---

## ⚙️ Best Hyperparameters (BiLSTM)

Our 3-fold cross-validation search identified these optimal configurations:

### Fine-Grained (28 emotions)
- **Focal Loss γ**: 2.0
- **Alpha Scale**: 1.0
- **CV Macro F1**: 0.4370
- **Test Macro F1**: 0.4295

### Ekman (6 emotions)
- **Focal Loss γ**: 2.0
- **Alpha Scale**: 0.75
- **CV Macro F1**: 0.5626
- **Test Macro F1**: 0.5667

### Sentiment (3 classes)
- **Focal Loss γ**: 1.5
- **Alpha Scale**: 1.0
- **CV Macro F1**: 0.6966
- **Test Macro F1**: 0.6945


---


### Evaluation Protocol  
- **Primary metric**: Macro F1-score (treats all emotions equally)
- **Fixed threshold**: 0.4 for all models (fair comparison)
- **Comprehensive metrics**: Precision, recall, per-class performance

---

## 📈 Next Steps

### Recently Completed ✅
- [x] Multi-granularity training (fine/ekman/sentiment) across all 3 models
- [x] Multi-model Gradio app with side-by-side comparison
- [x] DistilBERT fine-tuning across all granularities
- [x] Flair training across all granularities
- [x] Fair comparison protocol (same splits, threshold=0.4)


---

## 🙌 Acknowledgments

- **Dataset**: [GoEmotions by Google Research](https://github.com/google-research/google-research/tree/master/goemotions)
- **Embeddings**: [GloVe Twitter](https://nlp.stanford.edu/projects/glove/) | [sentence-transformers](https://www.sbert.net/)
- **Libraries**: TensorFlow/Keras, Scikit-learn, Flair, Transformers, Gradio

---