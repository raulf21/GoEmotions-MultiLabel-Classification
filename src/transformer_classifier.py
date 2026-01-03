import numpy as np
import torch
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, precision_score, recall_score
from data_preprocessing import preprocess_data


def compute_metrics(eval_pred):
    """
    Compute metrics for multi-label classification
    Uses 0.4 threshold (matches your best setup)
    """
    logits, labels = eval_pred
    # Sigmoid activation
    probs = 1 / (1 + np.exp(-logits))
    # Threshold at 0.4 (your optimal threshold)
    preds = (probs > 0.4).astype(int)
    
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0),
    }


def train_distilbert(
    granularity="fine",
    max_length=512,
    epochs=8,
    batch_size=8,
    gradient_accumulation_steps=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    early_stopping_patience=None
):
    """
    Train DistilBERT model for emotion classification
    Default hyperparameters match your best performing setup
    
    Args:
        granularity: "fine" (28 emotions), "ekman" (6 emotions), or "sentiment" (3 categories)
        max_length: Maximum sequence length for tokenization
        epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        weight_decay: Weight decay for AdamW optimizer
        fp16: Use mixed precision training
        early_stopping_patience: Early stopping patience (None = no early stopping)
    """
    
    # Step 1: Load and preprocess data (using base-cased tokenizer)
    print("Loading and preprocessing data...")
    data, classes = preprocess_data(
        "distilbert", 
        granularity=granularity, 
        model_name="distilbert-base-cased",
        max_length=max_length
    )
    df_train, df_val, df_test, train_dataset, val_dataset, test_dataset, tokenizer = data
    
    num_labels = len(classes)
    print(f"\nNumber of labels: {num_labels}")
    print(f"Classes: {classes}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    # Step 2: Initialize model
    print("\nInitializing DistilBERT model (base-cased)...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-cased",
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
    # Step 3: Set up training arguments (matching your best config)
    output_dir = f"./distilbert_{granularity}_results"
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_dir=f'./logs_{granularity}',
        logging_steps=100,
        save_total_limit=2,
        report_to="none",  # Disable wandb warnings
    )
    
    # Step 4: Initialize trainer
    callbacks = []
    if early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    
    # Step 5: Train
    print("\nStarting training...")
    print(f"Training for {epochs} epochs with lr={learning_rate}")
    trainer.train()
    
    # Step 6: Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Step 7: Save model
    model_save_path = f"./distilbert_{granularity}_final"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    return trainer, test_results, classes


def predict_emotions(text, granularity="fine", model_path=None, threshold=0.4):
    """
    Predict emotions for a given text
    
    Args:
        text: Input text string or list of texts
        granularity: "fine", "ekman", or "sentiment"
        model_path: Path to saved model (if None, uses default path)
        threshold: Probability threshold for positive prediction (default 0.4)
    
    Returns:
        If single text: (predicted_labels, predicted_scores)
        If list of texts: [(predicted_labels, predicted_scores), ...]
    """
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    
    if model_path is None:
        model_path = f"./distilbert_{granularity}_final"
    
    # Load model and tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Get class names
    _, classes = preprocess_data("distilbert", granularity=granularity, max_length=128)
    
    # Handle single text or list of texts
    is_single = isinstance(text, str)
    texts = [text] if is_single else text
    
    # Tokenize input
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits).numpy()
    
    results = []
    for pred in predictions:
        # Get predicted labels (using threshold)
        predicted_indices = np.where(pred > threshold)[0]
        predicted_labels = [classes[i] for i in predicted_indices]
        predicted_scores = {classes[i]: float(pred[i]) for i in predicted_indices}
        results.append((predicted_labels, predicted_scores))
    
    return results[0] if is_single else results


def main():
    """Main training function"""
    
    # Train on all three granularities with your best hyperparameters
    configs = {
        "fine": {
            "epochs": 8,
            "batch_size": 8,
            "gradient_accumulation_steps": 3,
            "learning_rate": 2e-5,
        },
        "ekman": {
            "epochs": 8,
            "batch_size": 8,
            "gradient_accumulation_steps": 3,
            "learning_rate": 2e-5,
        },
        "sentiment": {
            "epochs": 8,
            "batch_size": 8,
            "gradient_accumulation_steps": 3,
            "learning_rate": 2e-5,
        }
    }
    
    results_summary = {}
    
    for granularity, config in configs.items():
        print(f"\n{'='*80}")
        print(f"Training DistilBERT for {granularity.upper()} granularity")
        print(f"{'='*80}\n")
        
        trainer, results, classes = train_distilbert(
            granularity=granularity,
            max_length=512,
            **config
        )
        
        results_summary[granularity] = {
            "micro_f1": results.get('eval_micro_f1', 0),
            "macro_f1": results.get('eval_macro_f1', 0),
            "num_classes": len(classes)
        }
        
        print(f"\nCompleted training for {granularity}")
        print(f"  Micro F1: {results_summary[granularity]['micro_f1']:.4f}")
        print(f"  Macro F1: {results_summary[granularity]['macro_f1']:.4f}")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    for granularity, metrics in results_summary.items():
        print(f"\n{granularity.upper()}:")
        print(f"  Classes: {metrics['num_classes']}")
        print(f"  Micro F1: {metrics['micro_f1']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    
    # Example predictions
    print("\n" + "="*80)
    print("Example Predictions")
    print("="*80)
    
    example_texts = [
        "I'm so happy and excited about this amazing news!",
        "This is absolutely disgusting and makes me angry.",
        "I feel sad and disappointed about what happened."
    ]
    
    for text in example_texts:
        print(f"\nText: '{text}'")
        for granularity in configs.keys():
            try:
                labels, scores = predict_emotions(text, granularity, threshold=0.4)
                print(f"  {granularity.upper()}: {labels}")
                if scores:
                    top_score = max(scores.values())
                    print(f"    (confidence: {top_score:.3f})")
            except Exception as e:
                print(f"  {granularity.upper()}: Could not predict - {e}")


if __name__ == "__main__":
    # Quick test - train only fine-grained
    print("Training Fine-Grained Emotion Classification")
    print("="*80)
    
    main()