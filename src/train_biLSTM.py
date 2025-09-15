# train_model.py
import time
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import KFold
import json, pickle
from sklearn.metrics import f1_score


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
FIXED_THRESH   = 0.4            # kept for quick baseline reporting during CV
GAMMAS         = [1.5, 2.0]
ALPHA_SCALES   = [0.75, 1.0]    # still useful to modulate class-balanced alpha
SEED           = 42

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
    """
    For each class c, choose the threshold that maximizes F1 on the given validation set.
    """
    C = y_true.shape[1]
    th = np.zeros(C, dtype=np.float32)
    for c in range(C):
        p, r, t = precision_recall_curve(y_true[:, c], y_prob[:, c])
        if len(t) == 0:              # degenerate case
            th[c] = 0.5
            continue
        f1 = 2 * p * r / (p + r + 1e-9)
        # precision_recall_curve returns thresholds of length len(p)-1
        best_idx = np.nanargmax(f1[:-1]) if len(f1) > 1 else 0
        th[c] = t[best_idx]
    return th

def class_balanced_alpha(y_train, beta=0.999):
    """
    Effective number of samples weighting (Cui et al., 2019).
    Returns a vector alpha of shape [num_classes], normalized around ~1.
    """
    # y_train: [N, C] multi-hot
    n_c = y_train.sum(axis=0) + 1e-9
    eff_num = 1.0 - np.power(beta, n_c)
    w = (1.0 - beta) / eff_num
    w = w / w.mean()   # normalize so average weight ~ 1
    return w.astype(np.float32)

def make_focal_loss(alpha_vec, gamma):
    """
    Keras BinaryFocalCrossentropy supports class-wise alpha of shape [num_classes].
    """
    return keras.losses.BinaryFocalCrossentropy(
        gamma=gamma,
        alpha=alpha_vec,
        from_logits=False
    )

# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Load & preprocess
    (df_train, df_val, df_test,
     X_train, X_val, X_test,
     y_train, y_val, y_test,
     tokenizer, embedding_matrix, MAX_LEN, vocab_size) = preprocess()

    # 2) Reproducibility
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # 3) Class-balanced alpha (instead of raw inverse-frequency)
    base_alpha = class_balanced_alpha(y_train, beta=0.999)

    # 4) K-fold CV on TRAIN to select (gamma, alpha_scale)
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
                # during CV, use a fixed threshold just for model selection simplicity
                f1 = macro_f1_at_threshold(y_train[va_idx], probs, thresh=FIXED_THRESH)
                fold_f1s.append(f1)
                print(f"[CV γ={gamma}, αx={scale}] Fold {fold}: F1={f1:.4f}")

            mean_f1 = float(np.mean(fold_f1s))
            print(f"→ γ={gamma}, α_scale={scale} | mean CV F1 = {mean_f1:.4f}")

            if mean_f1 > best_mean_f1:
                best_mean_f1, best_params = mean_f1, (gamma, scale)

    print(f"\nBEST (CV): γ={best_params[0]}, α_scale={best_params[1]}  (CV F1={best_mean_f1:.4f})")

    # 5) With best (gamma, alpha_scale), train on TRAIN and pick per-class thresholds on VAL
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

    # Compute per-class thresholds on VAL
    probs_val = thresh_model.predict(X_val, verbose=0)
    per_class_thresh = find_per_class_thresholds(y_val, probs_val)  # your existing function
    print("\nPer-class thresholds (first 10):", np.round(per_class_thresh[:10], 3))

    # SAVE thresholds
    np.save("per_class_thresholds.npy", per_class_thresh)
    print("[Saved] per_class_thresholds.npy")

    # (Optional but recommended) SAVE tokenizer & preprocess config for inference
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open("preprocess_config.json", "w") as f:
        json.dump({"MAX_LEN": int(MAX_LEN), "vocab_size": int(vocab_size)}, f)
    print("[Saved] tokenizer.pkl and preprocess_config.json")

    # ─────────────────────────────────────────────────────
    # STEP 2: Final training on TRAIN+VAL, evaluate on TEST using saved thresholds
    # ─────────────────────────────────────────────────────
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
        validation_data=(X_test, y_test),   # keep TEST as hold-out
        epochs=EPOCHS_FINAL, batch_size=BATCH_SIZE,
        callbacks=[es_final, ckpt],
        verbose=1
    )
    print(f"\nFinal training time: {time.time() - t0:.1f}s")

    # Evaluate with per-class thresholds
    probs_test = final_model.predict(X_test, verbose=0)
    f1_test = macro_f1_with_per_class_thresh(y_test, probs_test, per_class_thresh)
    print(f"TEST Macro-F1 (per-class thresholds): {f1_test:.4f}")

if __name__ == "__main__":
    main()