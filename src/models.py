import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom")
class AdditiveAttentionPooling(keras.layers.Layer):
    def __init__(self, units=128, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.score = keras.layers.Dense(units, activation='tanh')
        self.v = keras.layers.Dense(1)

    def call(self, H, mask=None):
        # H: (B, T, D)
        s = self.v(self.score(H))  # (B, T, 1)
        if mask is not None:
            mask = tf.cast(mask[:, :, tf.newaxis], tf.float32)  # (B, T, 1)
            s = tf.where(mask > 0, s, tf.fill(tf.shape(s), -1e9))
        a = tf.nn.softmax(s, axis=1)            # (B, T, 1)
        ctx = tf.reduce_sum(H * a, axis=1)      # (B, D)
        return ctx, tf.squeeze(a, -1)           # (B, T)

    def compute_mask(self, inputs, mask=None):
        # After pooling we return a fixed-size vector; no downstream mask.
        return None

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.score.units})
        return cfg
    def build(self, input_shape):
        # No weights to create; just mark as built.
        super().build(input_shape)


def build_bilstm_with_attention(
    vocab_size: int,
    max_len: int,
    embedding_matrix=None,
    output_dim: int = 28,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    dropout_rate: float = 0.5,
    trainable_embeddings: bool = True,
):
    # IMPORTANT: tuple shape (max_len,) not (max_len)
    inp = Input(shape=(int(max_len),), dtype="int32", name="tokens")

    if embedding_matrix is not None:
        embed_layer = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=int(max_len),
            trainable=trainable_embeddings,
            mask_zero=True,           # pad=0 will be masked
            name="embedding",
        )
    else:
        embed_layer = Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            input_length=int(max_len),
            trainable=True,
            mask_zero=True,
            name="embedding",
        )

    x = embed_layer(inp)
    h = Bidirectional(LSTM(hidden_dim, return_sequences=True), name="bilstm")(x)
    ctx, attn = AdditiveAttentionPooling(units=hidden_dim)(h)  # no query arg

    out = Dropout(dropout_rate)(ctx)
    out = Dense(output_dim, activation='sigmoid', name="probs")(out)

    return Model(inputs=inp, outputs=out, name="BiLSTM_AdditiveAttn")
