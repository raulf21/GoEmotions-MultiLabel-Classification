# flair_classifier_fixed.py
"""
Flair-based multi-label classifier for GoEmotions - ALIGNED VERSION
Supports training at different granularities:
- fine (28 emotions)
- ekman (6 groups)  
- sentiment (3 groups)

ALIGNED WITH BiLSTM/DistilBERT:
- Uses original train/val/test splits
- Fixed threshold of 0.4 for all classes
- Evaluates on both validation and test sets
"""

import os
import time
import numpy as np
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
THRESHOLD = 0.4  # ✅ Fixed threshold aligned with BiLSTM/DistilBERT

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
        train_with_dev=False,  # ✅ Keep validation separate (aligned with BiLSTM)
        main_evaluation_metric=("micro avg", "f1-score"),
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")

    return out_dir, label_dict.get_items(), training_time

# -----------------------------
# Evaluation with explicit threshold control
# -----------------------------
def evaluate_flair(df_split, model_path, label_list, split_name="TEST", threshold=THRESHOLD):
    """
    Evaluate Flair model on a dataframe split with explicit threshold control.
    
    Args:
        df_split: DataFrame with text and label_strs columns
        model_path: Path to saved model
        label_list: List of label names
        split_name: Name of split for printing (e.g., "TEST", "VALIDATION")
        threshold: Probability threshold for positive prediction (default 0.4)
    """
    model = TextClassifier.load(os.path.join(model_path, "final-model.pt"))

    true_labels, predicted_labels = [], []

    print(f"Predicting on {len(df_split)} {split_name.lower()} samples with threshold={threshold}...")
    
    for _, row in df_split.iterrows():
        sentence = Sentence(row["text"])
        model.predict(sentence)
        
        # ✅ CRITICAL FIX: Apply explicit threshold instead of using Flair's default
        # Flair's predict() method assigns labels based on scores, but we need to
        # filter by our threshold of 0.4 to align with BiLSTM/DistilBERT
        predicted = [label.value for label in sentence.labels if label.score >= threshold]
        predicted_labels.append(predicted)
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

    print(f"\n===== {split_name} RESULTS (threshold={threshold}) =====")
    print(f"Macro Precision: {macro_prec:.4f}, Macro Recall: {macro_rec:.4f}, Macro F1: {macro_f1:.4f}")
    print(f"Micro Precision: {micro_prec:.4f}, Micro Recall: {micro_rec:.4f}, Micro F1: {micro_f1:.4f}")
    print("\nPer-class performance:")
    print(classification_report(y_true_bin, y_pred_bin, target_names=label_list, zero_division=0))

    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'threshold': threshold
    }

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("FLAIR TRAINING - ALIGNED WITH BiLSTM/DISTILBERT")
    print("="*70)
    print("Configuration:")
    print(f"  ✓ Uses ORIGINAL train/val/test splits")
    print(f"  ✓ Fixed threshold = {THRESHOLD} for ALL classes")
    print(f"  ✓ Evaluates on BOTH validation and test sets")
    print(f"  ✓ train_with_dev = False (validation kept separate)")
    print("="*70)
    
    results = []
    
    for granularity in ["fine", "ekman", "sentiment"]:
        print(f"\n{'='*70}")
        print(f"Running Flair pipeline for {granularity.upper()}")
        print(f"{'='*70}")

        # Preprocess dataset
        output, label_list = preprocess_for_flair(granularity)

        # Access data through named attributes
        df_train = output.df_train
        df_val = output.df_val
        df_test = output.df_test
        dataset_folder = output.dataset_folder

        print(f"\nDataset splits:")
        print(f"  Training: {len(df_train)} samples")
        print(f"  Validation: {len(df_val)} samples")
        print(f"  Test: {len(df_test)} samples")

        # Train
        model_path, label_list, training_time = train_flair(dataset_folder, granularity)

        # ✅ ALIGNED: Evaluate on BOTH validation and test sets (like BiLSTM)
        print("\n" + "="*70)
        print("EVALUATION PHASE")
        print("="*70)
        
        val_metrics = evaluate_flair(df_val, model_path, label_list, 
                                     split_name="VALIDATION", threshold=THRESHOLD)
        test_metrics = evaluate_flair(df_test, model_path, label_list, 
                                      split_name="TEST", threshold=THRESHOLD)
        
        # Store results
        result = {
            'granularity': granularity,
            'model': 'Flair',
            'training_time': training_time,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'threshold': THRESHOLD
        }
        results.append(result)

        print(f"\n✅ {granularity.upper()} model completed!")

    # Summary comparison
    print("\n" + "="*70)
    print("FLAIR RESULTS SUMMARY")
    print("="*70)
    for result in results:
        print(f"\n{result['granularity'].upper()}:")
        print(f"  Training time: {result['training_time']:.1f}s")
        print(f"  Validation Macro F1: {result['val_metrics']['macro_f1']:.4f}")
        print(f"  Test Macro F1: {result['test_metrics']['macro_f1']:.4f}")
        print(f"  Threshold: {result['threshold']}")
    
    print("\n" + "="*70)
    print("✅ READY FOR FAIR COMPARISON WITH BiLSTM AND DISTILBERT")
    print("="*70)
    print("All models now:")
    print("  • Use same train/val/test splits")
    print("  • Use fixed threshold = 0.4")
    print("  • Evaluate on same validation/test sets")
    print("="*70)