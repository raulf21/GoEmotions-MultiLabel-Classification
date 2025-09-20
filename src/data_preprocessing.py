# data_preprocessing.py
import os
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import emoji
from collections import Counter
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# --------------------------
# Label definitions
# --------------------------
LABELS_FINE = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear","gratitude",
    "grief","joy","love","nervousness","optimism","pride","realization","relief","remorse","sadness","surprise","neutral"
]

ekman_mapping = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy","amusement","approval","excitement","gratitude",
            "love","optimism","relief","pride","admiration","desire","caring"],
    "sadness": ["sadness","disappointment","embarrassment","grief","remorse"],
    "surprise": ["surprise","realization","confusion","curiosity"]
}
label_to_ekman = {emo: group for group, emos in ekman_mapping.items() for emo in emos}

positive = set(ekman_mapping["joy"])
negative = set(ekman_mapping["anger"] + ekman_mapping["disgust"] + ekman_mapping["fear"] + ekman_mapping["sadness"])
neutral = {"neutral"}

# --------------------------
# Mapping helpers
# --------------------------
def map_labels_to_ekman(fine_labels, label_names=LABELS_FINE):
    """Map fine-grained labels -> Ekman categories, drop neutral."""
    return list({label_to_ekman[label_names[i]] for i in fine_labels if label_names[i] in label_to_ekman})

def map_to_sentiment(fine_labels, label_names=LABELS_FINE):
    """Map fine-grained labels -> sentiment (pos/neg/neutral)."""
    groups = set()
    for i in fine_labels:
        l = label_names[i]
        if l in positive:
            groups.add("positive")
        elif l in negative:
            groups.add("negative")
        elif l in neutral:
            groups.add("neutral")
    return list(groups)

def add_label_strs(df, granularity):
    """Add label_strs column to dataframe based on granularity"""
    if granularity == "fine":
        df['label_strs'] = df['labels'].apply(lambda idxs: [LABELS_FINE[i] for i in idxs])
        return LABELS_FINE
    elif granularity == "ekman":
        df['label_strs'] = df['labels'].apply(lambda idxs: map_labels_to_ekman(idxs, LABELS_FINE))
        return ["anger","disgust","fear","joy","sadness","surprise"]
    elif granularity == "sentiment":
        df['label_strs'] = df['labels'].apply(lambda idxs: map_to_sentiment(idxs, LABELS_FINE))
        return ["positive","negative","neutral"]
    else:
        raise ValueError("granularity must be fine|ekman|sentiment")

# -------------------------------------
# Data Loading
# -------------------------------------
def load_goemotions():
    splits = {
        'train':      'simplified/train-00000-of-00001.parquet',
        'validation': 'simplified/validation-00000-of-00001.parquet',
        'test':       'simplified/test-00000-of-00001.parquet'
    }
    base = "hf://datasets/google-research-datasets/go_emotions/"
    return (
        pd.read_parquet(base + splits['train']),
        pd.read_parquet(base + splits['validation']),
        pd.read_parquet(base + splits['test']),
    )

# --------------------------
# BiLSTM Specific Preprocessing
# --------------------------
def get_emoji_map(df):
    all_nonascii = df['text'].str.findall(r'[^\x00-\x7F]').explode().dropna()
    all_emojis = [ch for ch in all_nonascii if ch in emoji.EMOJI_DATA]
    top_50 = [emo for emo, _ in Counter(all_emojis).most_common(50)]

    emoji_map = {}
    for emo in top_50:
        desc = emoji.demojize(emo).strip(':').replace('/', '_')
        emoji_map[emo] = f" {desc}_emoji "
    return emoji_map

def clean_and_tokenize(text, emoji_map, lemmatizer):
    text = text.replace("[NAME]", " name_token ")
    text = re.sub(r'<[^>]+>', ' ', text)
    
    text = re.sub(r'/u/\w+', ' user_mention ', text)      # Reddit user mentions like /u/username
    text = re.sub(r'/r/\w+', ' subreddit_mention ', text) # Subreddit mentions like /r/funny
    text = re.sub(r'www\.\S+', ' url_token ', text)       # URLs
    
    # Your existing contractions (keep this):
    for patt, repl in {
        r"\bI'm\b": "i 'm",
        r"\bit's\b": "it 's",
        r"\bcan't\b": "ca n't",
        r"\bdon't\b": "do n't",
        r"\bthey're\b": "they 're"
    }.items():
        text = re.sub(patt, repl, text, flags=re.IGNORECASE)

    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'@\w+', ' user_token ', text)

    for emo, repl in emoji_map.items():
        text = text.replace(emo, repl)

    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    text = text.lower()
    text = re.sub(r"[^a-z'\s_#@]", ' ', text)

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t, pos='v') if t.isalpha() else t for t in tokens]
    return tokens

def build_label_vector(lbls, classes):
    vec = np.zeros(len(classes), dtype=np.float32)
    for l in lbls:
        if isinstance(l, int):   # fine labels (indices)
            vec[l] = 1.0
        else:                    # ekman/sentiment (strings)
            if l in classes:
                vec[classes.index(l)] = 1.0
    return vec

def build_sequences_and_embeddings(df_train, df_val, df_test,
                                   glove_path="./glove.twitter.27B.100d.txt",
                                   embedding_dim=100):
    """Build sequences and embedding matrix for BiLSTM"""
    y_train = np.stack(df_train["label_vector"].values)
    y_val   = np.stack(df_val["label_vector"].values)
    y_test  = np.stack(df_test["label_vector"].values)

    texts_train = [' '.join(t) for t in df_train['tokens']]
    texts_val   = [' '.join(t) for t in df_val['tokens']]
    texts_test  = [' '.join(t) for t in df_test['tokens']]

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(texts_train)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"[Tokenizer] vocab size = {vocab_size:,}")

    train_seq = tokenizer.texts_to_sequences(texts_train)
    val_seq   = tokenizer.texts_to_sequences(texts_val)
    test_seq  = tokenizer.texts_to_sequences(texts_test)

    MAX_LEN = int(df_train['tokens'].map(len).quantile(0.95))
    print(f"[Padding] MAX_LEN (95th pct) = {MAX_LEN}")

    X_train = pad_sequences(train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_val   = pad_sequences(val_seq,   maxlen=MAX_LEN, padding='post', truncating='post')
    X_test  = pad_sequences(test_seq,  maxlen=MAX_LEN, padding='post', truncating='post')

    # Load GloVe embeddings
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype="float32")
    try:
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word, vec = parts[0], parts[1:]
                if word in tokenizer.word_index:
                    embedding_matrix[tokenizer.word_index[word]] = np.asarray(vec, dtype='float32')
    except FileNotFoundError:
        print(f"[GloVe] File not found at {glove_path}")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            tokenizer, embedding_matrix, MAX_LEN, vocab_size)

def preprocess_for_bilstm(granularity="fine", glove_path="./glove.twitter.27B.100d.txt"):
    df_train, df_val, df_test = load_goemotions()

    classes = None
    for df in [df_train, df_val, df_test]:
        classes = add_label_strs(df, granularity)

    emoji_map = get_emoji_map(df_train)
    lemmatizer = WordNetLemmatizer()

    for df in [df_train, df_val, df_test]:
        df['tokens'] = df['text'].apply(lambda x: clean_and_tokenize(x, emoji_map, lemmatizer))
        df['label_vector'] = df['label_strs'].apply(lambda lbls: build_label_vector(lbls, classes))

    sequences_and_embeddings = build_sequences_and_embeddings(df_train, df_val, df_test, glove_path)

    print(f"[DEBUG] {granularity.upper()} unique labels: {set().union(*df_train['label_strs'])}")
    return (df_train, df_val, df_test, *sequences_and_embeddings), classes

# --------------------------
# Flair Specific Preprocessing
# --------------------------
def clean_text_flair(text):
    text = text.replace("[NAME]", "NAME")
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def df_to_flair_txt(df, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            labels = row.get("label_strs", [])
            if not labels:
                continue
            clean_text = clean_text_flair(str(row["text"]))
            label_prefixes = [f"__label__{l}" for l in labels]
            line = " ".join(label_prefixes) + " " + clean_text
            f.write(line + "\n")

def prepare_flair_dataset(df_train, df_val, df_test, granularity):
    folder = f"flair_dataset_{granularity}"
    os.makedirs(folder, exist_ok=True)
    df_to_flair_txt(df_train, os.path.join(folder, "train.txt"))
    df_to_flair_txt(df_val, os.path.join(folder, "dev.txt"))
    df_to_flair_txt(df_test, os.path.join(folder, "test.txt"))
    print(f"Flair dataset prepared in: {folder}")
    print(f"[DEBUG] {granularity.upper()} unique labels: {set().union(*df_train['label_strs'])}")
    return folder

def preprocess_for_flair(granularity="fine"):
    df_train, df_val, df_test = load_goemotions()
    classes = None
    for df in [df_train, df_val, df_test]:
        classes = add_label_strs(df, granularity)
    dataset_folder = prepare_flair_dataset(df_train, df_val, df_test, granularity)
    return (df_train, df_val, df_test, dataset_folder), classes

# --------------------------
# Unified interface
# --------------------------
def preprocess_data(model_type, granularity="fine", **kwargs):
    if model_type.lower() == "bilstm":
        return preprocess_for_bilstm(granularity, kwargs.get('glove_path', './glove.twitter.27B.100d.txt'))
    elif model_type.lower() == "flair":
        return preprocess_for_flair(granularity)
    else:
        raise ValueError("model_type must be 'bilstm' or 'flair'")



