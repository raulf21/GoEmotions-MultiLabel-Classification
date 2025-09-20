# flair_classifier.py
"""
Flair-based multi-label classifier for GoEmotions.
Supports training at different granularities:
- fine (28 emotions)
- ekman (6 groups)  
- sentiment (3 groups)

FIXED: Proper train/val/test split handling to match BiLSTM evaluation protocol.
"""

import os
import time
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from flair.data import Sentence
from flair.datasets import ClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# ---- Project imports ----
from data_preprocessing import preprocess_for_flair

# -----------------------------
# Config
# -----------------------------
EPOCHS = 10
BATCH_SIZE = 8
LR = 5e-4
LABEL_TYPE = "emotion"
MODEL_BASE = "flair_models"

# -----------------------------
# Training
# -----------------------------
def train_flair(dataset_folder, granularity):
    """Train a Flair classifier at the given granularity."""
    corpus = ClassificationCorpus(
        dataset_folder,
        train_file="train.txt",
        dev_file="dev.txt",  # Keep separate for validation
        test_file="test.txt",
        label_type=LABEL_TYPE
    )
    label_dict = corpus.make_label_dictionary(label_type=LABEL_TYPE)

    embeddings = TransformerDocumentEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

    classifier = TextClassifier(
        embeddings=embeddings,
        label_dictionary=label_dict,
        label_type=LABEL_TYPE,
        multi_label=True,
    )

    out_dir = os.path.join(MODEL_BASE, f"{granularity}_model")
    trainer = ModelTrainer(classifier, corpus)
    
    print(f"Training Flair model for {granularity}...")
    start_time = time.time()
    
    trainer.train(
        base_path=out_dir,
        learning_rate=LR,
        mini_batch_size=BATCH_SIZE,
        max_epochs=EPOCHS,
        train_with_dev=False,  # ✅ Keep validation separate
        main_evaluation_metric=("micro avg", "f1-score"),
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")

    return out_dir, label_dict.get_items(), training_time

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_flair(df_split, model_path, label_list, split_name="TEST"):
    """Evaluate Flair model on a dataframe split."""
    model = TextClassifier.load(os.path.join(model_path, "final-model.pt"))

    true_labels, predicted_labels = [], []

    print(f"Predicting on {len(df_split)} {split_name.lower()} samples...")
    for _, row in df_split.iterrows():
        sentence = Sentence(row["text"])
        model.predict(sentence)
        predicted_labels.append([label.value for label in sentence.labels] if sentence.labels else [])
        true_labels.append(row["label_strs"])

    mlb = MultiLabelBinarizer(classes=label_list)
    mlb.fit([label_list])  # Fit on all possible labels

    y_true_bin = mlb.transform(true_labels)
    y_pred_bin = mlb.transform(predicted_labels)

    # Calculate metrics
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    macro_prec = precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    macro_rec = recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    micro_prec = precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    micro_rec = recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)

    print(f"\n===== {split_name} RESULTS ({len(label_list)} labels) =====")
    print(f"Macro Precision: {macro_prec:.4f}, Macro Recall: {macro_rec:.4f}, Macro F1: {macro_f1:.4f}")
    print(f"Micro Precision: {micro_prec:.4f}, Micro Recall: {micro_rec:.4f}, Micro F1: {micro_f1:.4f}")
    print("\nPer-class performance:")
    print(classification_report(y_true_bin, y_pred_bin, target_names=label_list, zero_division=0))

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec
    }

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    results = []
    
    for granularity in ["fine", "ekman", "sentiment"]:
        print(f"\n{'='*60}")
        print(f"Running Flair pipeline for {granularity.upper()}")
        print(f"{'='*60}")

        # Preprocess dataset
        (df_train, df_val, df_test, dataset_folder), label_list = preprocess_for_flair(granularity)

        # Train
        model_path, label_list, training_time = train_flair(dataset_folder, granularity)

        # Evaluate only on test set (matching BiLSTM protocol)
        test_metrics = evaluate_flair(df_test, model_path, label_list, split_name="TEST")
        
        # Store results
        result = {
            'granularity': granularity,
            'model': 'Flair',
            'training_time': training_time,
            'test_metrics': test_metrics
        }
        results.append(result)

        print(f"✅ {granularity.upper()} model completed!")

    # Summary comparison
    print("\n" + "="*60)
    print("FLAIR RESULTS SUMMARY")
    print("="*60)
    for result in results:
        print(f"\n{result['granularity'].upper()}:")
        print(f"  Training time: {result['training_time']:.1f}s") 
        print(f"  Test Macro F1: {result['test_metrics']['macro_f1']:.4f}")
        print(f"  Test Micro F1: {result['test_metrics']['micro_f1']:.4f}")
    
    print(f"\n{'='*60}")
    print("READY FOR COMPARISON WITH BiLSTM RESULTS")
    print(f"{'='*60}")