# GoEmotions-MultiLabel-Classification 😄

This project explores **fine-grained multi-label emotion classification** using the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions).  
Our goal was to replicate and extend Google’s original work by experimenting with multiple architectures and deployment approaches.

---

## 🚀 Features
- **BiLSTM + Attention Model (final)**  
  - Custom `AdditiveAttentionPooling` layer.  
  - Trained on preprocessed GoEmotions dataset with class-balanced focal loss.  
  - Per-class threshold calibration for improved multi-label prediction.  
  - Achieved competitive **Macro-F1 on the test set**.  

- **Interactive Demo (Gradio App)**  
  - Input any text → returns predicted emotions as emojis + probabilities.  
  - Adjustable threshold slider (global or per-class).  
  - Example:  
    > _"OMG, yep!!! That is the final answer! Thank you so much!"_  
    → 🎉 excitement · 🙏 gratitude · ✅ approval  

- **Other explored methods**
  - Transformer-based: DistilBERT, SentenceTransformers (MiniLM).  
  - Rule-based baselines with NLTK + Flair.  
  - Pre-trained embeddings (GloVe Twitter).  

---

## 📂 Repository Structure
```
Go-Emotions/
│── notebooks/                # Exploratory data analysis + modeling notebooks
│── src/
│   ├── train_biLSTM.py       # Training script (BiLSTM + Attention + focal loss)
│   ├── models.py             # Model architectures & custom layers
│   ├── data_preprocessing.py # Preprocessing pipeline (tokenizer, cleaning)
│   ├── inference.py          # Simple CLI inference for quick testing
│   ├── app.py                # Gradio web app for interactive predictions
│   ├── tokenizer.pkl         # Saved tokenizer (generated during training)
│   ├── preprocess_config.json# Config (e.g., MAX_LEN, vocab size)
│   └── per_class_thresholds.npy # Saved thresholds (calibrated on validation)
│── requirements.txt
│── README.md
```

---

## 🛠️ Installation & Setup
```bash
# Clone repo
git clone https://github.com/raulf21/GoEmotions-MultiLabel-Classification.git
cd GoEmotions-MultiLabel-Classification

# Create conda env
conda create -n nlp python=3.9
conda activate nlp

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the BiLSTM Model
```bash
cd src
python train_biLSTM.py
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


---

## 📊 Results

- **Final BiLSTM + Attention model** (with focal loss + per-class thresholds):  
  - **Macro-F1 (Test): 0.459**  
  - **Micro-F1 (Test): 0.540**  
  - **Macro Precision: 0.517 · Macro Recall: 0.475**

- **Comparison to GoEmotions paper**  
  - Google BERT baseline (28 emotions): **Macro-F1 ≈ 0.46**  
  - Google biLSTM baseline: **Macro-F1 ≈ 0.41**  
  - 👉 Our BiLSTM+Attention matches BERT-level performance while surpassing their biLSTM baseline.

- **Per-class performance highlights**  
  - Strong classes: *gratitude (F1 = 0.90)*, *love (0.80)*, *amusement (0.77)*, *neutral (0.68)*.  
  - Challenging classes: *relief (0.09)*, *realization (0.14)*, *grief (0.31)* — rare labels with very low support.  

- **Key takeaway**  
  Attention pooling + focal loss + per-class thresholding **substantially improve macro-F1**, especially balancing frequent vs rare emotions.  


---

## 🙌 Acknowledgments
- Dataset: [GoEmotions by Google Research](https://github.com/google-research/google-research/tree/master/goemotions)  
- Pre-trained embeddings: [GloVe Twitter](https://nlp.stanford.edu/projects/glove/)  
- Libraries: TensorFlow/Keras, Scikit-learn, NLTK, Gradio  

---

## 💡 Next Steps
- Add GIF walkthroughs of model training and results.  
- Experiment with ensemble of BiLSTM + Transformer.  
- Add visualization of attention weights for explainability.
