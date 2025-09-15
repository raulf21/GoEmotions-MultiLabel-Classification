# train_model.py
import time
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, classification_report, precision_score, recall_score
from sklearn.model_selection import KFold
import json, pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---- project imports ----
from data_preprocessing import main as preprocess
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
SEED           = 42

LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude",
    "grief","joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"
]

# -----------------------------
# Utilities
# -----------------------------
def macro_f1_at_threshold(y_true, y_prob, thresh=0.4):
    y_pred = (y_prob >= thresh).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def macro_f1_with_per_class_thresh(y_true, y_prob, th_vec):
    y_pred = (y_prob >= th_vec[np.newaxis, :]).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def find_per_class_thresholds(y_true, y_prob):
    C = y_true.shape[1]
    th = np.zeros(C, dtype=np.float32)
    for c in range(C):
        p, r, t = precision_recall_curve(y_true[:, c], y_prob[:, c])
        if len(t) == 0:
            th[c] = 0.5
            continue
        f1 = 2 * p * r / (p + r + 1e-9)
        best_idx = np.nanargmax(f1[:-1]) if len(f1) > 1 else 0
        th[c] = t[best_idx]
    return th

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
# Main
# -----------------------------
def main():
    # Load & preprocess
    (df_train, df_val, df_test,
     X_train, X_val, X_test,
     y_train, y_val, y_test,
     tokenizer, embedding_matrix, MAX_LEN, vocab_size) = preprocess()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    base_alpha = class_balanced_alpha(y_train, beta=0.999)

    # K-fold CV
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    best_mean_f1 = -1.0
    best_params  = None

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

                es = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
                model.fit(
                    X_train[tr_idx], y_train[tr_idx],
                    validation_data=(X_train[va_idx], y_train[va_idx]),
                    epochs=EPOCHS_CV, batch_size=BATCH_SIZE,
                    callbacks=[es], verbose=1
                )

                probs = model.predict(X_train[va_idx], verbose=0)
                f1 = macro_f1_at_threshold(y_train[va_idx], probs, thresh=FIXED_THRESH)
                fold_f1s.append(f1)
                print(f"[CV γ={gamma}, αx={scale}] Fold {fold}: F1={f1:.4f}")

            mean_f1 = float(np.mean(fold_f1s))
            print(f"→ γ={gamma}, α_scale={scale} | mean CV F1 = {mean_f1:.4f}")

            if mean_f1 > best_mean_f1:
                best_mean_f1, best_params = mean_f1, (gamma, scale)

    print(f"\nBEST (CV): γ={best_params[0]}, α_scale={best_params[1]}  (CV F1={best_mean_f1:.4f})")

    # With best params, train on TRAIN and calibrate thresholds on VAL
    gamma_best, scale_best = best_params
    alpha_best = (base_alpha * scale_best).astype(np.float32)

    K.clear_session()
    thresh_model = build_bilstm_with_attention(
        vocab_size=vocab_size,
        max_len=MAX_LEN,
        embedding_matrix=embedding_matrix,
        output_dim=y_train.shape[1],
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        trainable_embeddings=True,
    )
    thresh_model.compile(optimizer="adam", loss=make_focal_loss(alpha_best, gamma_best))

    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    thresh_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_CV, batch_size=BATCH_SIZE,
        callbacks=[es], verbose=1
    )

    probs_val = thresh_model.predict(X_val, verbose=0)
    per_class_thresh = find_per_class_thresholds(y_val, probs_val)
    np.save("per_class_thresholds.npy", per_class_thresh)
    print("[Saved] per_class_thresholds.npy")

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("preprocess_config.json", "w") as f:
        json.dump({"MAX_LEN": int(MAX_LEN), "vocab_size": int(vocab_size)}, f)
    print("[Saved] tokenizer.pkl and preprocess_config.json")

    # Final training on TRAIN+VAL
    X_final = np.concatenate([X_train, X_val], axis=0)
    y_final = np.concatenate([y_train, y_val], axis=0)

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

    ckpt = keras.callbacks.ModelCheckpoint(
        "best_bilstm_model_final.keras",
        monitor="val_loss",
        save_best_only=True
    )
    es_final = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    t0 = time.time()
    final_model.fit(
        X_final, y_final,
        validation_data=(X_test, y_test),
        epochs=EPOCHS_FINAL, batch_size=BATCH_SIZE,
        callbacks=[es_final, ckpt],
        verbose=1
    )
    print(f"\nFinal training time: {time.time() - t0:.1f}s")

    # Evaluate
    probs_test = final_model.predict(X_test, verbose=0)
    y_pred = (probs_test >= per_class_thresh[np.newaxis, :]).astype(int)

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

if __name__ == "__main__":
    main()
