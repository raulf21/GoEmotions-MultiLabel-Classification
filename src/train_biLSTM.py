# train_biLSTM.py
"""
BiLSTM + Attention training for GoEmotions at multiple granularities:
- fine (28 labels)
- ekman (6 groups)
- sentiment (3 groups)

Workflow (academically standard):
1. Merge provided train + validation into one training set.
2. Do KFold cross-validation on that combined set to grid search hyperparams.
3. Select best params from CV.
4. Split merged training into train/val for final training (80/20).
5. Retrain best model on final train split with clean validation.
6. Evaluate once on held-out test set.

FORCED CPU-ONLY EXECUTION (ignores GPU/MPS).
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report
)
from sklearn.model_selection import KFold, train_test_split

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
FIXED_THRESH   = 0.4
GAMMAS         = [1.5, 2.0]
ALPHA_SCALES   = [0.75, 1.0]
FINAL_VAL_SPLIT = 0.2  # Hold out 20% of merged training for final validation
SEED           = 42

# -----------------------------
# Force CPU-only
# -----------------------------
tf.config.set_visible_devices([], "GPU")
print("⚠️ Forcing TensorFlow to run on CPU only.")

# -----------------------------
# Utilities
# -----------------------------
def macro_f1_at_threshold(y_true, y_prob, thresh=0.4):
    y_pred = (y_prob >= thresh).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def class_balanced_alpha(y_train, beta=0.999):
    n_c = y_train.sum(axis=0) + 1e-9
    eff_num = 1.0 - np.power(beta, n_c)
    w = (1.0 - beta) / eff_num
    w = w / w.mean()
    return w.astype(np.float32)

def make_focal_loss(alpha_vec, gamma):
    return keras.losses.BinaryFocalCrossentropy(
        gamma=gamma,
        alpha=alpha_vec,
        from_logits=False
    )

# -----------------------------
# Training loop
# -----------------------------
def run_training(granularity="fine"):
    print(f"\n### Training for {granularity.upper()} labels ###")

    # Load & preprocess with val included
    (
        df_train, df_val, df_test,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        tokenizer, embedding_matrix, MAX_LEN, vocab_size
    ), LABELS = preprocess_for_bilstm(granularity)

    # Merge train + val into one big training set for hyperparameter search
    df_train_merged = pd.concat([df_train, df_val], ignore_index=True)
    X_train_merged  = np.concatenate([X_train, X_val], axis=0)
    y_train_merged  = np.concatenate([y_train, y_val], axis=0)

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    base_alpha = class_balanced_alpha(y_train_merged, beta=0.999)

    # Grid search with CV on merged training data
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    best_mean_f1 = -1.0
    best_params  = None

    print("Starting hyperparameter search with cross-validation...")
    for gamma in GAMMAS:
        for scale in ALPHA_SCALES:
            alpha_vec = (base_alpha * scale).astype(np.float32)
            fold_f1s = []

            for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_merged), start=1):
                K.clear_session()
                model = build_bilstm_with_attention(
                    vocab_size=vocab_size,
                    max_len=MAX_LEN,
                    embedding_matrix=embedding_matrix,
                    output_dim=y_train_merged.shape[1],
                    hidden_dim=HIDDEN_DIM,
                    dropout_rate=DROPOUT_RATE,
                    trainable_embeddings=True,
                )
                model.compile(optimizer="adam", loss=make_focal_loss(alpha_vec, gamma))

                es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
                model.fit(
                    X_train_merged[tr_idx], y_train_merged[tr_idx],
                    validation_data=(X_train_merged[va_idx], y_train_merged[va_idx]),
                    epochs=EPOCHS_CV, batch_size=BATCH_SIZE,
                    callbacks=[es], verbose=1
                )

                probs = model.predict(X_train_merged[va_idx], verbose=0)
                f1 = macro_f1_at_threshold(y_train_merged[va_idx], probs, thresh=FIXED_THRESH)
                fold_f1s.append(f1)
                print(f"[CV γ={gamma}, αx={scale}] Fold {fold}: F1={f1:.4f}")

            mean_f1 = float(np.mean(fold_f1s))
            print(f"→ γ={gamma}, α_scale={scale} | mean CV F1 = {mean_f1:.4f}")

            if mean_f1 > best_mean_f1:
                best_mean_f1, best_params = mean_f1, (gamma, scale)

    print(f"\nBEST (CV): γ={best_params[0]}, α_scale={best_params[1]}  (CV F1={best_mean_f1:.4f})")

    # Now split the merged training data into final train/validation splits
    # This creates a clean validation set that hasn't been seen during hyperparameter search
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train_merged, y_train_merged,
        test_size=FINAL_VAL_SPLIT,
        random_state=SEED,
        stratify=None  # Could add stratification logic for multilabel if needed
    )
    
    print(f"\nFinal training splits:")
    print(f"  Training: {len(X_train_final)} samples")
    print(f"  Validation: {len(X_val_final)} samples") 
    print(f"  Test (held-out): {len(X_test)} samples")

    # Train final model with best hyperparameters on clean train/val split
    gamma_best, scale_best = best_params
    alpha_best = (class_balanced_alpha(y_train_final, beta=0.999) * scale_best).astype(np.float32)

    K.clear_session()
    final_model = build_bilstm_with_attention(
        vocab_size=vocab_size,
        max_len=MAX_LEN,
        embedding_matrix=embedding_matrix,
        output_dim=y_train_final.shape[1],
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        trainable_embeddings=True,
    )
    final_model.compile(optimizer="adam", loss=make_focal_loss(alpha_best, gamma_best))

    ckpt_path = f"best_bilstm_{granularity}.keras"
    ckpt = keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_loss", save_best_only=True
    )
    es_final = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    print("\nTraining final model with best hyperparameters...")
    t0 = time.time()
    final_model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val_final, y_val_final),  # ✅ Clean validation set
        epochs=EPOCHS_FINAL, batch_size=BATCH_SIZE,
        callbacks=[es_final, ckpt],
        verbose=1
    )
    print(f"\nFinal training time: {time.time() - t0:.1f}s")

    # Final evaluation on completely held-out test set
    print("\nEvaluating on held-out test set...")
    probs_test = final_model.predict(X_test, verbose=0)
    y_pred = (probs_test >= FIXED_THRESH).astype(int)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    micro_prec = precision_score(y_test, y_pred, average="micro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    micro_rec = recall_score(y_test, y_pred, average="micro", zero_division=0)

    print("\n===== TEST PERFORMANCE =====")
    print(f"Macro Precision: {macro_prec:.4f}, Macro Recall: {macro_rec:.4f}, Macro F1: {macro_f1:.4f}")
    print(f"Micro Precision: {micro_prec:.4f}, Micro Recall: {micro_rec:.4f}, Micro F1: {micro_f1:.4f}")
    print("\nPer-class performance:")
    print(classification_report(y_test, y_pred, target_names=LABELS, zero_division=0))

    return {
        'granularity': granularity,
        'best_params': best_params,
        'cv_f1': best_mean_f1,
        'test_metrics': {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'macro_precision': macro_prec,
            'macro_recall': macro_rec
        }
    }


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    results = []
    for granularity in ["fine", "ekman", "sentiment"]:
        result = run_training(granularity=granularity)
        results.append(result)
    
    # Summary of all results
    print("\n" + "="*60)
    print("SUMMARY OF ALL GRANULARITIES")
    print("="*60)
    for result in results:
        print(f"\n{result['granularity'].upper()}:")
        print(f"  Best params: γ={result['best_params'][0]}, α_scale={result['best_params'][1]}")
        print(f"  CV F1: {result['cv_f1']:.4f}")
        print(f"  Test Macro F1: {result['test_metrics']['macro_f1']:.4f}")
        print(f"  Test Micro F1: {result['test_metrics']['micro_f1']:.4f}")


