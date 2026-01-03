# train_biLSTM_fixed.py
"""
BiLSTM + Attention training with FIXED SPLITS (aligned with Flair/DistilBERT)
- Uses original train/val/test splits from GoEmotions
- Uses fixed threshold of 0.4 for all classes (no per-class optimization)
- Fair comparison with other models
"""

import time
import pickle  # Add this
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report
)
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# ---- project imports ----
from data_preprocessing import preprocess_for_bilstm
from models import build_bilstm_with_attention

# -----------------------------
# Config / Hyperparameters
# -----------------------------
N_FOLDS        = 3
EPOCHS_CV      = 5
EPOCHS_FINAL   = 10
BATCH_SIZE     = 64
HIDDEN_DIM     = 128
DROPOUT_RATE   = 0.5
THRESHOLD      = 0.4  # Fixed threshold for all classes
GAMMAS         = [1.5, 2.0]
ALPHA_SCALES   = [0.75, 1.0]
SEED           = 42


# -----------------------------
# Utilities
# -----------------------------
def macro_f1_at_threshold(y_true, y_prob, thresh=0.4):
    """Calculate macro F1 at a fixed threshold"""
    y_pred = (y_prob >= thresh).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def class_balanced_alpha(y_train, beta=0.999):
    """Calculate class-balanced alpha weights for focal loss"""
    n_c = y_train.sum(axis=0) + 1e-9
    eff_num = 1.0 - np.power(beta, n_c)
    w = (1.0 - beta) / eff_num
    w = w / w.mean()
    return w.astype(np.float32)


def make_focal_loss(alpha_vec, gamma):
    """Create focal loss with given alpha and gamma"""
    return keras.losses.BinaryFocalCrossentropy(
        gamma=gamma,
        alpha=alpha_vec,
        from_logits=False
    )


def evaluate_model(y_true, y_prob, labels, threshold=0.4, split_name="TEST"):
    """Evaluate model with fixed threshold"""
    y_pred = (y_prob >= threshold).astype(int)
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    print(f"\n===== {split_name} PERFORMANCE (threshold={threshold}) =====")
    print(f"Macro Precision: {macro_prec:.4f}, Macro Recall: {macro_rec:.4f}, Macro F1: {macro_f1:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")
    print("\nPer-class performance:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec
    }


# -----------------------------
# Training loop
# -----------------------------
def run_training(granularity="fine"):
    print(f"\n### Training BiLSTM for {granularity.upper()} labels ###")

    # Load & preprocess (using original splits)
    output, LABELS = preprocess_for_bilstm(granularity=granularity)

    X_train = output.X_train
    X_val = output.X_val
    X_test = output.X_test
    y_train = output.y_train
    y_val = output.y_val
    y_test = output.y_test
    tokenizer = output.tokenizer
    embedding_matrix = output.embedding_matrix
    MAX_LEN = output.max_len
    vocab_size = output.vocab_size

    print(f"\nDataset splits:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Calculate class weights on training set only
    base_alpha = class_balanced_alpha(y_train, beta=0.999)

    # =====================================================================
    # PHASE 1: Cross-validation for hyperparameter selection (on train only)
    # =====================================================================
    print("\n" + "="*70)
    print("PHASE 1: Hyperparameter Search (CV on training set only)")
    print("="*70)
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    best_mean_f1 = -1.0
    best_params = None

    for gamma in GAMMAS:
        for scale in ALPHA_SCALES:
            alpha_vec = (base_alpha * scale).astype(np.float32)
            fold_f1s = []

            for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
                K.clear_session()
                model = build_bilstm_with_attention(
                    vocab_size=vocab_size,
                    max_len=MAX_LEN,
                    embedding_matrix=embedding_matrix,
                    output_dim=y_train.shape[1],
                    hidden_dim=HIDDEN_DIM,
                    dropout_rate=DROPOUT_RATE,
                    trainable_embeddings=True,
                )
                model.compile(optimizer="adam", loss=make_focal_loss(alpha_vec, gamma))

                es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
                model.fit(
                    X_train[tr_idx], y_train[tr_idx],
                    validation_data=(X_train[va_idx], y_train[va_idx]),  # ← CV within training set
                    epochs=EPOCHS_CV, batch_size=BATCH_SIZE,
                    callbacks=[es], verbose=0
                )

                probs = model.predict(X_train[va_idx], verbose=0)
                f1 = macro_f1_at_threshold(y_train[va_idx], probs, thresh=THRESHOLD)
                fold_f1s.append(f1)
                print(f"[CV γ={gamma}, α×={scale}] Fold {fold}: F1={f1:.4f}")

            mean_f1 = float(np.mean(fold_f1s))
            print(f"→ γ={gamma}, α_scale={scale} | mean CV F1 = {mean_f1:.4f}")

            if mean_f1 > best_mean_f1:
                best_mean_f1, best_params = mean_f1, (gamma, scale)

    print(f"\n🏆 BEST HYPERPARAMETERS (from CV): γ={best_params[0]}, α_scale={best_params[1]} (CV F1={best_mean_f1:.4f})")

    # =====================================================================
    # PHASE 2: Train final model on full training set
    # =====================================================================
    print("\n" + "="*70)
    print("PHASE 2: Training Final Model on Full Training Set")
    print("="*70)
    
    gamma_best, scale_best = best_params
    alpha_best = (base_alpha * scale_best).astype(np.float32)

    K.clear_session()
    final_model = build_bilstm_with_attention(
        vocab_size=vocab_size,
        max_len=MAX_LEN,
        embedding_matrix=embedding_matrix,
        output_dim=y_train.shape[1],
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        trainable_embeddings=True,
    )
    final_model.compile(optimizer="adam", loss=make_focal_loss(alpha_best, gamma_best))

    ckpt_path = f"best_bilstm_{granularity}.keras"
    ckpt = keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)
    es_final = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    print(f"\nTraining with best hyperparameters (γ={gamma_best}, α_scale={scale_best})...")
    t0 = time.time()
    final_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),  # ← Original validation set (never seen during CV)
        epochs=EPOCHS_FINAL, batch_size=BATCH_SIZE,
        callbacks=[es_final, ckpt],
        verbose=1
    )
    training_time = time.time() - t0
    print(f"\nTraining completed in {training_time:.1f}s")
    print(f"Model saved to: {ckpt_path}")

    # =====================================================================
    # PHASE 3: Final evaluation on validation and test sets
    # =====================================================================
    print("\n" + "="*70)
    print("PHASE 3: Final Evaluation")
    print("="*70)
    
    # Evaluate on validation set
    probs_val = final_model.predict(X_val, verbose=0)
    val_results = evaluate_model(y_val, probs_val, LABELS, threshold=THRESHOLD, split_name="VALIDATION")
    
    # Evaluate on test set
    probs_test = final_model.predict(X_test, verbose=0)
    test_results = evaluate_model(y_test, probs_test, LABELS, threshold=THRESHOLD, split_name="TEST")

    # Save tokenizer and config for app deployment
    print("\n" + "="*70)
    print("PHASE 4: Saving Artifacts for Deployment")
    print("="*70)
    
    # Save tokenizer
    tokenizer_path = f"tokenizer_{granularity}.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"✓ Saved tokenizer: {tokenizer_path}")
    
    # Save preprocessing config
    config = {
        "max_len": MAX_LEN,
        "vocab_size": vocab_size,
        "granularity": granularity,
        "threshold": THRESHOLD,
        "labels": LABELS
    }
    config_path = f"preprocess_config_{granularity}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config: {config_path}")
    
    print("="*70)

    return {
        'granularity': granularity,
        'best_params': best_params,
        'cv_f1': best_mean_f1,
        'training_time': training_time,
        'val_results': val_results,
        'test_results': test_results,
        'threshold': THRESHOLD
    }


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("BiLSTM TRAINING - ALIGNED WITH FLAIR/DISTILBERT")
    print("="*70)
    print("Changes from original:")
    print("  ✓ Uses ORIGINAL train/val/test splits (no merge/resplit)")
    print("  ✓ CV for hyperparameters on TRAIN set only")
    print("  ✓ Fixed threshold = 0.4 for ALL classes (no per-class optimization)")
    print("  ✓ Evaluates on SAME validation set as Flair/DistilBERT")
    print("="*70)
    
    results = []
    for granularity in ["fine", "ekman", "sentiment"]:
        result = run_training(granularity=granularity)
        results.append(result)
    
    # Summary comparison
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    for result in results:
        print(f"\n{result['granularity'].upper()}:")
        print(f"  Best params: γ={result['best_params'][0]}, α_scale={result['best_params'][1]}")
        print(f"  CV F1 (on train): {result['cv_f1']:.4f}")
        print(f"  Validation F1: {result['val_results']['macro_f1']:.4f}")
        print(f"  Test F1: {result['test_results']['macro_f1']:.4f}")
        print(f"  Training time: {result['training_time']:.1f}s")
    
    print("\n" + "="*70)
    print("✅ READY FOR FAIR COMPARISON WITH FLAIR AND DISTILBERT")
    print("="*70)
    print("All models now:")
    print("  • Use same train/val/test splits")
    print("  • Use fixed threshold = 0.4")
    print("  • Evaluate on same validation/test sets")
    print("="*70)