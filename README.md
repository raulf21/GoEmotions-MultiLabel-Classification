# GoEmotions-MultiLabel-Classification
The project explores fine-grained emotion classification by building upon the methodology of Google's original GoEmotions paper. The primary objective was to investigate wether different deep learning models and embedding strategies could achieve or surpass the baseline results on this dataset. 

The project evaluates and compares several approaches, including:

* Transformer-based Models: Fine-tuning of sentence-transformers/all-MiniLM-L6-v2 and DistilBERT for multi-label classification.

* LSTM Architectures: A Bi-directional LSTM model with an attention mechanism, leveraging pre-trained GloVe Twitter embeddings.

* Rule-based & Traditional Methods: Exploration of NLTK and a Flair-based classification model.

The repository includes all code, data preprocessing steps, and model evaluations, providing a complete and reproducible workflow
