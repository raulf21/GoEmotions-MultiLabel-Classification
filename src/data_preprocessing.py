import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import emoji
from collections import Counter
import numpy as np

# Tokenizer/padding (local, no fetching)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure tokenizers/lemmatizer data
nltk.download('punkt')
nltk.download('wordnet')

NUM_CLASSES = 28

def get_emoji_map(df):
    """
    Generates a map of the top 50 most frequent emojis to descriptive tokens.
    """
    all_nonascii = df['text'].str.findall(r'[^\x00-\x7F]').explode().dropna()
    all_emojis = [ch for ch in all_nonascii if ch in emoji.EMOJI_DATA]
    top_50 = [emo for emo, _ in Counter(all_emojis).most_common(50)]

    emoji_map = {}
    for emo in top_50:
        desc = emoji.demojize(emo).strip(':').replace('/', '_')
        emoji_map[emo] = f" {desc}_emoji "
    return emoji_map

def clean_and_tokenize(text, emoji_map, lemmatizer):
    """
    Cleans, tokenizes, and lemmatizes one text string.
    """
    # 1. Placeholder & HTML
    text = text.replace("[NAME]", " name_token ")
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. Standardize contractions
    for patt, repl in {
        r"\bI'm\b": "i 'm",
        r"\bit's\b": "it 's",
        r"\bcan't\b": "ca n't",
        r"\bdon't\b": "do n't",
        r"\bthey're\b": "they 're"
    }.items():
        text = re.sub(patt, repl, text, flags=re.IGNORECASE)

    # 3. Hashtags & handles
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'@\w+', ' user_token ', text)

    # 4. Emojis → tokens
    for emo, repl in emoji_map.items():
        text = text.replace(emo, repl)

    # 5. Remove non-ASCII, collapse repeats
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 6. Lowercase & keep letters/underscores/spaces/quotes
    text = text.lower()
    text = re.sub(r"[^a-z'\s_#@]", ' ', text)

    # 7. Tokenize & lemmatize
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t, pos='v') if t.isalpha() else t for t in tokens]
    return tokens

# Rebuild label vectors after filtering
def build_label_vector(lbls, NUM_CLASSES=NUM_CLASSES):
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    vec[lbls] = 1.0
    return vec

# ----------------------------
# Tokenizer, padding, and GloVe embeddings (LOCAL ONLY)
# ----------------------------
def build_sequences_and_embeddings(df_train, df_val, df_test,
                                   glove_path="./glove.twitter.27B.100d.txt",
                                   embedding_dim=100):
    """
    From tokenized DataFrames, build:
      - tokenizer fit on train
      - padded integer sequences (X_train/val/test)
      - label matrices (y_train/val/test) from label_vector
      - embedding_matrix loaded from LOCAL GloVe file
      - MAX_LEN and vocab_size
      - glove_words: set of words present in the GloVe file (for raw OOV analysis)
    """
    # y_* from multi-hot label_vector
    y_train = np.stack(df_train["label_vector"].values)
    y_val   = np.stack(df_val["label_vector"].values)
    y_test  = np.stack(df_test["label_vector"].values)

    # Join tokens into strings for tokenizer
    texts_train = [' '.join(t) for t in df_train['tokens']]
    texts_val   = [' '.join(t) for t in df_val  ['tokens']]
    texts_test  = [' '.join(t) for t in df_test ['tokens']]

    # Fit tokenizer on train only
    tokenizer = Tokenizer(oov_token="<OOV>")  # 0: PAD, 1: OOV
    tokenizer.fit_on_texts(texts_train)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"[Tokenizer] vocab size = {vocab_size:,}")

    # Sequences and padding
    train_seq = tokenizer.texts_to_sequences(texts_train)
    val_seq   = tokenizer.texts_to_sequences(texts_val)
    test_seq  = tokenizer.texts_to_sequences(texts_test)

    # 95th percentile length, with a small floor
    MAX_LEN = int(df_train['tokens'].map(len).quantile(0.95))
    print(f"[Padding] MAX_LEN (95th pct) = {MAX_LEN}")

    X_train = pad_sequences(train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_val   = pad_sequences(val_seq,   maxlen=MAX_LEN, padding='post', truncating='post')
    X_test  = pad_sequences(test_seq,  maxlen=MAX_LEN, padding='post', truncating='post')

    print(f"[Shapes] X_train {X_train.shape} | X_val {X_val.shape} | X_test {X_test.shape}")
    print(f"         y_train {y_train.shape} | y_val {y_val.shape} | y_test {y_test.shape}")

    # Quick peek
    if len(X_train):
        import textwrap
        idx = np.random.randint(len(X_train))
        print("\n[Sample seq] original:", textwrap.shorten(texts_train[idx], 120))
        print("              indices :", X_train[idx, :20], "...")
        print("              tokens  :", tokenizer.sequences_to_texts([X_train[idx, :]]))
        print("              labels  :", y_train[idx, :])

        two_label_idxs = np.where(y_train.sum(axis=1) == 2)[0]
        if len(two_label_idxs):
            idx2 = np.random.choice(two_label_idxs)
            print("\n[Sample seq: 2 labels] original:", textwrap.shorten(texts_train[idx2], 120))
            print("                        indices :", X_train[idx2, :20], "...")
            print("                        tokens  :", tokenizer.sequences_to_texts([X_train[idx2, :]]))
            print("                        labels  :", y_train[idx2, :])
        else:
            print("No examples with exactly 2 labels found in the training set.")

    # Build GloVe embedding matrix from LOCAL file only
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype="float32")
    hits, misses = 0, 0
    try:
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word, vec = parts[0], parts[1:]
                if word in tokenizer.word_index:
                    embedding_matrix[tokenizer.word_index[word]] = np.asarray(vec, dtype='float32')
                    hits +=1
                else:
                    misses +=1
    except FileNotFoundError:
        print(f"[GloVe] File not found at: {glove_path}. "
              f"Place glove.twitter.27B.100d.txt in the same directory as this script or update glove_path.")

    coverage = hits / vocab_size
    print(f"[GloVe] words covered = {hits:,} / {vocab_size:,}  ({coverage:.1%})")
    print("[GloVe] embedding_matrix shape:", embedding_matrix.shape)

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            tokenizer, embedding_matrix, MAX_LEN, vocab_size)
# -------------- main --------------

def main():
    """
    Load GoEmotions (simplified), preprocess, and return processed DataFrames
    plus tokenized sequences and GloVe embeddings (from local file).
    """
    # Load the datasets (pandas + parquet from HF hub)
    splits = {
        'train':      'simplified/train-00000-of-00001.parquet',
        'validation': 'simplified/validation-00000-of-00001.parquet',
        'test':       'simplified/test-00000-of-00001.parquet'
    }
    base = "hf://datasets/google-research-datasets/go_emotions/"

    df_train = pd.read_parquet(base + splits['train'])
    df_val   = pd.read_parquet(base + splits['validation'])
    df_test  = pd.read_parquet(base + splits['test'])

    print("Datasets loaded.")
    print("Train shape:", df_train.shape)
    print("Validation shape:", df_val.shape)
    print("Test shape:", df_test.shape)

    # Emoji map from train
    emoji_map = get_emoji_map(df_train)
    print("Emoji map (first 10):", list(emoji_map.items())[:10])

    lemmatizer = WordNetLemmatizer()

    # Tokenize all splits
    print("Applying cleaning and tokenization to all splits...")
    df_train['tokens'] = df_train['text'].apply(lambda x: clean_and_tokenize(x, emoji_map, lemmatizer))
    df_val['tokens']   = df_val['text'].apply(lambda x: clean_and_tokenize(x, emoji_map, lemmatizer))
    df_test['tokens']  = df_test['text'].apply(lambda x: clean_and_tokenize(x, emoji_map, lemmatizer))

    # Label vectors
    print("Building label vectors...")
    df_train['label_vector'] = df_train['labels'].apply(build_label_vector)
    df_val['label_vector']   = df_val['labels'].apply(build_label_vector)
    df_test['label_vector']  = df_test['labels'].apply(build_label_vector)

    # Verify a couple of rows
    print("First train row post-process:\n", df_train[['text','labels','tokens','label_vector']].iloc[0])
    print("First val row post-process:\n", df_val[['text','labels','tokens','label_vector']].iloc[0])
    print("First test row post-process:\n", df_test[['text','labels','tokens','label_vector']].iloc[0])

    # Build sequences + embeddings from local file
    print("\nBuilding sequences and GloVe embeddings...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     tokenizer, embedding_matrix, MAX_LEN, vocab_size) = build_sequences_and_embeddings(
        df_train, df_val, df_test,
        glove_path="./glove.twitter.27B.100d.txt",
        embedding_dim=100
    )

    print("\nData preprocessing complete.")
    return (df_train, df_val, df_test,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            tokenizer, embedding_matrix, MAX_LEN, vocab_size)

if __name__ == '__main__':
    (train_data, val_data, test_data,
     X_train, X_val, X_test,
     y_train, y_val, y_test,
     tokenizer, embedding_matrix, MAX_LEN, vocab_size) = main()
