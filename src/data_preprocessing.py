# data_preprocessing.py
import os
import re
import pandas as pd
import nltk
import emoji
from collections import Counter
import numpy as np

from nltk import word_tokenize
from typing import NamedTuple

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import DistilBertTokenizerFast
from datasets import Dataset

class BiLSTMPreprocessOutput(NamedTuple):
    """
    Output from BiLSTM preprocessing function.
    Contains all necessary data for training a BiLSTM model with Glove embeddings.
    """
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    tokenizer: Tokenizer
    embedding_matrix: np.ndarray
    max_len: int  # ✅ FIXED: Changed from MAX_LEN to max_len
    vocab_size: int

class FlairPreprocessOutput(NamedTuple):
    """
    Output from Flair preprocessing function.
    Contains all necessary data for training a Flair model.
    """
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    dataset_folder: str
class DistilBERTPreprocessOutput(NamedTuple):
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    df_test: pd.DataFrame
    train_dataset: any  # HuggingFace Dataset
    val_dataset: any
    test_dataset: any
    tokenizer: any  # DistilBertTokenizer

# Ensure NLTK data
nltk.download("punkt", quiet=True)

# --------------------------
# Configuration Constants
# --------------------------
TOP_N_EMOJIS = 50
PADDING_QUANTILE = 0.95
MAX_CHAR_REPETITION = 4  # Allow up to 4 character repetitions for elongations
DEFAULT_EMBEDDING_DIM = 100

# --------------------------
# Label definitions
# --------------------------
LABELS_FINE = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude",
    "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness",
    "surprise", "neutral"
]

ekman_mapping = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude",
            "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"]
}
label_to_ekman = {emo: group for group, emos in ekman_mapping.items() for emo in emos}

positive = set(ekman_mapping["joy"])
negative = set(ekman_mapping["anger"] + ekman_mapping["disgust"] + ekman_mapping["fear"] + ekman_mapping["sadness"])
neutral = {"neutral"}

# --------------------------
# Mapping helpers
# --------------------------
def map_labels_to_ekman(fine_labels, label_names=LABELS_FINE):
    """
    Map fine-grained emotion labels to Ekman's 6 basic emotions.
    
    Args:
        fine_labels (list): List of fine-grained label indices
        label_names (list): List of fine-grained label names
    
    Returns:
        list: List of Ekman emotion labels
    """
    return list({label_to_ekman[label_names[i]] for i in fine_labels if label_names[i] in label_to_ekman})


def map_to_sentiment(fine_labels, label_names=LABELS_FINE):
    """
    Map fine-grained emotion labels to sentiment (positive/negative/neutral).
    
    Args:
        fine_labels (list): List of fine-grained label indices
        label_names (list): List of fine-grained label names
    
    Returns:
        list: List of sentiment labels
    """
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
    """
    Add string labels to dataframe based on granularity level.
    
    Args:
        df (DataFrame): DataFrame with 'labels' column
        granularity (str): One of 'fine', 'ekman', or 'sentiment'
    
    Returns:
        list: List of label names for the specified granularity
    """
    if granularity == "fine":
        df['label_strs'] = df['labels'].apply(lambda idxs: [LABELS_FINE[i] for i in idxs])
        return LABELS_FINE
    elif granularity == "ekman":
        df['label_strs'] = df['labels'].apply(lambda idxs: map_labels_to_ekman(idxs, LABELS_FINE))
        return ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    elif granularity == "sentiment":
        df['label_strs'] = df['labels'].apply(lambda idxs: map_to_sentiment(idxs, LABELS_FINE))
        return ["positive", "negative", "neutral"]
    else:
        raise ValueError("granularity must be fine|ekman|sentiment")

# -------------------------------------
# Data Loading
# -------------------------------------
def load_goemotions():
    """
    Load GoEmotions dataset from HuggingFace.
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
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
def load_glove_vocab(glove_path):
    """
    Load just the vocabulary (words) from GloVe file without the vectors.
    This is faster than loading full embeddings.
    
    Args:
        glove_path (str): Path to GloVe embeddings file
    
    Returns:
        set: Set of words in GloVe vocabulary
    """
    if not os.path.exists(glove_path):
        print(f"[GloVe Vocab] File not found at {glove_path}, returning empty vocab")
        return set()
    
    vocab = set()
    print(f"[GloVe Vocab] Loading vocabulary from {glove_path}...")
    
    try:
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                word = line.split(" ", 1)[0]  # Get just the first token (the word)
                vocab.add(word)
        print(f"[GloVe Vocab] ✓ Loaded {len(vocab):,} words")
    except Exception as e:
        print(f"[GloVe Vocab] ERROR: {e}")
        return set()
    
    return vocab


def get_emoji_map(df, top_n=TOP_N_EMOJIS, glove_vocab=None):
    """
    Extract top N most common emojis and create replacement mappings.
    Checks if emojis exist in GloVe before converting to text tokens.
    
    Args:
        df (DataFrame): DataFrame with 'text' column
        top_n (int): Number of top emojis to extract
        glove_vocab (set, optional): Set of words in GloVe vocabulary
    
    Returns:
        dict: Mapping of emojis to descriptive tokens (or keeps emoji if in GloVe)
    """
    all_nonascii = df['text'].str.findall(r'[^\x00-\x7F]').explode().dropna()
    all_emojis = [ch for ch in all_nonascii if ch in emoji.EMOJI_DATA]
    top_emojis = [emo for emo, _ in Counter(all_emojis).most_common(top_n)]

    emoji_map = {}
    emojis_in_glove = 0
    emojis_converted = 0
    
    for emo in top_emojis:
        # Check if emoji itself is in GloVe vocabulary
        if glove_vocab and emo in glove_vocab:
            # Keep the emoji as-is since GloVe has an embedding for it
            emoji_map[emo] = f" {emo} "
            emojis_in_glove += 1
        else:
            # Convert to descriptive token
            desc = emoji.demojize(emo).strip(':').replace('/', '_').replace('-', '_')
            emoji_map[emo] = f" {desc}_emoji "
            emojis_converted += 1
    
    # Handle rare emojis not in top N
    for emo in set(all_emojis) - set(top_emojis):
        if glove_vocab and emo in glove_vocab:
            emoji_map[emo] = f" {emo} "
        else:
            emoji_map[emo] = " rare_emoji "
    
    # Report statistics
    print(f"[Emoji Mapping] Found {len(all_emojis):,} total emojis in dataset")
    print(f"[Emoji Mapping] Top {top_n} emojis selected for special handling")
    if glove_vocab:
        print(f"[Emoji Mapping] ✓ {emojis_in_glove}/{len(top_emojis)} top emojis found in GloVe (kept as-is)")
        print(f"[Emoji Mapping] → {emojis_converted}/{len(top_emojis)} top emojis converted to text tokens")
    else:
        print(f"[Emoji Mapping] ⚠️  GloVe vocab not provided, converting all emojis to text")
    
    return emoji_map


def clean_for_twitter_glove(text, emoji_map):
    """
    Minimal cleaning designed for GloVe Twitter embeddings.
    Preserves case, emoticons, and informal patterns to match GloVe training data.
    
    Args:
        text (str): Raw text from GoEmotions
        emoji_map (dict): Mapping of emojis to descriptive tokens
    
    Returns:
        list: Tokenized text preserving Twitter-native features
    
    Examples:
        >>> clean_for_twitter_glove("OMG I'm SOOOOO happy!!!", {})
        ['OMG', "I'm", 'SOOO', 'happy', '!!!']
    """
    # Replace known tokens
    text = text.replace("[NAME]", "NAME")
    
    # Clean HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Handle Reddit-specific patterns
    text = re.sub(r'/u/\w+', 'USER', text)
    text = re.sub(r'/r/\w+', 'SUBREDDIT', text)
    
    # Normalize URLs
    text = re.sub(r'http\S+|www\.\S+', 'URL', text)
    
    # Replace emojis with descriptive tokens
    for emo, repl in emoji_map.items():
        text = text.replace(emo, repl)
    
    # Gentle elongation normalization (allow up to MAX_CHAR_REPETITION)
    # This preserves expressive patterns like "soooo" while preventing excessive repetition
    text = re.sub(r'(.)\1{' + str(MAX_CHAR_REPETITION) + r',}', r'\1' * MAX_CHAR_REPETITION, text)
    
    # Tokenize - CRITICAL: No lowercasing, no lemmatization!
    tokens = word_tokenize(text)
    
    return tokens


def build_label_vector(lbls, classes):
    """
    Build multi-label binary vector for emotion labels.
    
    Args:
        lbls (list): List of label strings or indices
        classes (list): List of all possible class names
    
    Returns:
        np.ndarray: Binary vector of shape (len(classes),)
    """
    vec = np.zeros(len(classes), dtype=np.float32)
    for l in lbls:
        if isinstance(l, int):
            vec[l] = 1.0
        else:
            if l in classes:
                vec[classes.index(l)] = 1.0
    return vec


def build_sequences_and_embeddings(df_train, df_val, df_test,
                                   glove_path="./glove.twitter.27B.100d.txt",
                                   embedding_dim=DEFAULT_EMBEDDING_DIM):
    """
    Build sequences and load GloVe embeddings.
    
    Args:
        df_train, df_val, df_test (DataFrame): DataFrames with 'tokens' and 'label_vector' columns
        glove_path (str): Path to GloVe embeddings file
        embedding_dim (int): Dimension of embeddings
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, 
                tokenizer, embedding_matrix, MAX_LEN, vocab_size)
    
    Raises:
        FileNotFoundError: If GloVe file doesn't exist and user confirms they want to stop
    """
    # Check if GloVe file exists before processing
    if not os.path.exists(glove_path):
        print("\n" + "="*70)
        print("⚠️  WARNING: GloVe embeddings file not found!")
        print("="*70)
        print(f"Expected location: {glove_path}")
        print("\nGloVe Twitter embeddings are REQUIRED for this model.")
        print("\nDownload instructions:")
        print("  1. wget https://nlp.stanford.edu/data/glove.twitter.27B.zip")
        print("  2. unzip glove.twitter.27B.zip")
        print("  3. Ensure the file is in the correct location")
        print("\nAvailable dimensions: 25d, 50d, 100d, 200d")
        print(f"Current configuration expects: {embedding_dim}d")
        print("="*70)
        
        response = input("\nDo you want to continue with random initialization? (yes/no): ").lower()
        if response not in ['yes', 'y']:
            raise FileNotFoundError(f"GloVe embeddings not found at {glove_path}. Please download and try again.")
        else:
            print("\n⚠️  Proceeding with RANDOM initialization (NOT recommended for production)")
    
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

    MAX_LEN = int(df_train['tokens'].map(len).quantile(PADDING_QUANTILE))
    print(f"[Padding] MAX_LEN ({int(PADDING_QUANTILE*100)}th percentile) = {MAX_LEN}")

    X_train = pad_sequences(train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_val   = pad_sequences(val_seq,   maxlen=MAX_LEN, padding='post', truncating='post')
    X_test  = pad_sequences(test_seq,  maxlen=MAX_LEN, padding='post', truncating='post')

    # Load GloVe embeddings
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype="float32")
    words_found = 0
    
    if os.path.exists(glove_path):
        print(f"[GloVe] Loading embeddings from {glove_path}...")
        try:
            with open(glove_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip().split(" ")
                    word, vec = parts[0], parts[1:]
                    if word in tokenizer.word_index:
                        embedding_matrix[tokenizer.word_index[word]] = np.asarray(vec, dtype='float32')
                        words_found += 1
            
            coverage = (words_found / (vocab_size - 1)) * 100  # -1 to exclude padding token
            print(f"[GloVe] ✓ Found embeddings for {words_found:,}/{vocab_size-1:,} words ({coverage:.2f}% coverage)")
            
            if coverage < 50:
                print(f"[GloVe] ⚠️  WARNING: Low coverage ({coverage:.2f}%). Consider checking:")
                print(f"         - Is this the correct GloVe file for Twitter?")
                print(f"         - Are you using the right embedding dimension?")
            
        except Exception as e:
            print(f"[GloVe] ERROR reading file: {e}")
            print(f"[GloVe] Using random initialization")
    else:
        # This should not happen due to earlier check, but keeping for safety
        print(f"[GloVe] Using random initialization (file not found)")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            tokenizer, embedding_matrix, MAX_LEN, vocab_size)


def preprocess_for_bilstm(granularity="fine", glove_path="./glove.twitter.27B.100d.txt") -> tuple[BiLSTMPreprocessOutput, list]:
    """
    Complete preprocessing pipeline for BiLSTM model.
    
    Args:
        granularity (str): Label granularity ('fine', 'ekman', or 'sentiment')
        glove_path (str): Path to GloVe embeddings file
    
    Returns:
        tuple[BiLSTMPreprocessOutput, list]:
            - BiLSTMPreprocessOutput: All preprocessing outputs in a named structure
            - list: Class labels for the chosen granularity
    """
    df_train, df_val, df_test = load_goemotions()

    classes = None
    for df in [df_train, df_val, df_test]:
        classes = add_label_strs(df, granularity)

    # Load GloVe vocabulary first to check which emojis are in GloVe
    print("\n" + "="*70)
    print("STEP 1: Loading GloVe vocabulary (for emoji checking)")
    print("="*70)
    glove_vocab = load_glove_vocab(glove_path)
    
    # Create emoji map based on what's in GloVe
    print("\n" + "="*70)
    print("STEP 2: Creating emoji mapping")
    print("="*70)
    emoji_map = get_emoji_map(df_train, glove_vocab=glove_vocab)
    
    # Tokenize texts
    print("\n" + "="*70)
    print("STEP 3: Tokenizing texts")
    print("="*70)
    for df in [df_train, df_val, df_test]:
        df['tokens'] = df['text'].apply(lambda x: clean_for_twitter_glove(x, emoji_map))
        df['label_vector'] = df['label_strs'].apply(lambda lbls: build_label_vector(lbls, classes))
    
    # Build sequences and load full embeddings
    print("\n" + "="*70)
    print("STEP 4: Building sequences and loading embeddings")
    print("="*70)
    X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, embedding_matrix, MAX_LEN, vocab_size = \
        build_sequences_and_embeddings(df_train, df_val, df_test, glove_path)

    print(f"\n[DEBUG] {granularity.upper()} unique labels: {set().union(*df_train['label_strs'])}")
    
    # ✨ Create the NamedTuple for type-safe return
    output = BiLSTMPreprocessOutput(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        tokenizer=tokenizer,
        embedding_matrix=embedding_matrix,
        max_len=MAX_LEN,
        vocab_size=vocab_size
    )
    
    return output, classes  # ✅ FIXED: Changed from "classess" to "classes"


# --------------------------
# Flair Specific Preprocessing
# --------------------------
def clean_text_flair(text):
    """
    Clean text for Flair classifier.
    
    Args:
        text (str): Raw text
    
    Returns:
        str: Cleaned text
    """
    text = text.replace("[NAME]", "NAME")
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def df_to_flair_txt(df, out_path):
    """
    Convert DataFrame to Flair format text file.
    
    Args:
        df (DataFrame): DataFrame with 'text' and 'label_strs' columns
        out_path (str): Output file path
    """
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
    """
    Prepare dataset in Flair format.
    
    Args:
        df_train, df_val, df_test (DataFrame): DataFrames with labels
        granularity (str): Label granularity level
    
    Returns:
        str: Path to dataset folder
    """
    folder = f"flair_dataset_{granularity}"
    os.makedirs(folder, exist_ok=True)
    df_to_flair_txt(df_train, os.path.join(folder, "train.txt"))
    df_to_flair_txt(df_val, os.path.join(folder, "dev.txt"))
    df_to_flair_txt(df_test, os.path.join(folder, "test.txt"))
    print(f"[Flair] Dataset prepared in: {folder}")
    print(f"[DEBUG] {granularity.upper()} unique labels: {set().union(*df_train['label_strs'])}")
    return folder


def preprocess_for_flair(granularity="fine") -> tuple[FlairPreprocessOutput, list]:
    """
    Complete preprocessing pipeline for Flair model.
    
    Args:
        granularity (str): Label granularity ('fine', 'ekman', or 'sentiment')
    
    Returns:
        tuple[FlairPreprocessOutput, list]:
            - FlairPreprocessOutput: All preprocessing outputs in a named structure
            - list: Class labels for the chosen granularity
    """
    df_train, df_val, df_test = load_goemotions()
    classes = None
    for df in [df_train, df_val, df_test]:
        classes = add_label_strs(df, granularity)
    dataset_folder = prepare_flair_dataset(df_train, df_val, df_test, granularity)
    
    # ✨ Create the NamedTuple for type-safe return
    output = FlairPreprocessOutput(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        dataset_folder=dataset_folder
    )
    
    return output, classes


def preprocess_for_distilbert(granularity="fine", model_name="distilbert-base-cased", max_length=512) -> tuple[DistilBERTPreprocessOutput, list]:
    """Complete preprocessing pipeline for DistilBERT model."""
    
    # Load data
    df_train, df_val, df_test = load_goemotions()
    
    # Add labels
    classes = None
    for df in [df_train, df_val, df_test]:
        classes = add_label_strs(df, granularity)
        df['label_vector'] = df['label_strs'].apply(lambda lbls: build_label_vector(lbls, classes))
    
    # Initialize tokenizer
    print(f"[DistilBERT] Initializing tokenizer: {model_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(df_train[['text', 'label_vector']].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(df_val[['text', 'label_vector']].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(df_test[['text', 'label_vector']].reset_index(drop=True))
    
    # Tokenize
    print(f"[DistilBERT] Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Format labels
    def format_labels(example):
        example['labels'] = example['label_vector']
        return example
    
    train_dataset = train_dataset.map(format_labels)
    val_dataset = val_dataset.map(format_labels)
    test_dataset = test_dataset.map(format_labels)
    
    # Set PyTorch format
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    print(f"[DistilBERT] ✓ Tokenized {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    output = DistilBERTPreprocessOutput(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        tokenizer=tokenizer
    )
    
    return output, classes


# --------------------------
# Unified interface
# --------------------------
def preprocess_data(model_type, granularity="fine", **kwargs):
    """
    Unified preprocessing interface for different model types.
    
    Args:
        model_type (str): 'bilstm' or 'flair'
        granularity (str): Label granularity ('fine', 'ekman', or 'sentiment')
        **kwargs: Additional arguments passed to specific preprocessing functions
    
    Returns:
        tuple: Preprocessing outputs and class labels (format depends on model_type)
    """
    if model_type.lower() == "bilstm":
        return preprocess_for_bilstm(granularity, kwargs.get('glove_path', './glove.twitter.27B.100d.txt'))
    elif model_type.lower() == "flair":
        return preprocess_for_flair(granularity)
    elif model_type.lower() == "distilbert":  # ← NEW!
        return preprocess_for_distilbert(
            granularity, 
            kwargs.get('model_name', 'distilbert-base-cased'),
            kwargs.get('max_length', 512)
        )


# --------------------------
# Test harness (run directly)
# --------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DEMONSTRATION: Emoji Checking Against GloVe Vocabulary")
    print("="*70)
    
    # Simulate having a small GloVe vocab
    # In reality, GloVe Twitter might have some emojis like 😂
    fake_glove_vocab = {"😂", "LOL", "happy", "sad", "OMG", "URL", "NAME"}
    
    print("\nSimulated GloVe vocabulary (for demo):")
    print(f"  {fake_glove_vocab}")
    
    # Create a simple DataFrame with emojis
    import pandas as pd
    sample_df = pd.DataFrame({
        'text': [
            "I'm so happy! 😂😂😂",
            "This is sad 😭😭",
            "Amazing! 🔥🔥🔥",
            "Love it ❤️❤️"
        ]
    })
    
    print("\n--- Testing get_emoji_map with GloVe vocab checking ---")
    emoji_map_with_check = get_emoji_map(sample_df, top_n=10, glove_vocab=fake_glove_vocab)
    
    print("\nEmoji Mapping Results:")
    for emo, token in sorted(emoji_map_with_check.items()):
        status = "✓ KEPT (in GloVe)" if emo in token else "→ CONVERTED (not in GloVe)"
        print(f"  {emo} => {token.strip():30s} {status}")
    
    # Now test the actual preprocessing
    print("\n" + "="*70)
    print("TWITTER-NATIVE PREPROCESSING TEST")
    print("="*70)
    
    test_cases = [
        "OMG I'm SOOOOO happy!!! 😂😂😂",
        "i love this :) :) :)",
        "WHY would they DO that?!?!",
        "this is AMAZINGGGGG!!!",
        "Check out this link: https://example.com",
        "Posted by /u/username in /r/subreddit",
    ]
    
    # Use the emoji map we just created
    for i, text in enumerate(test_cases, 1):
        tokens = clean_for_twitter_glove(text, emoji_map_with_check)
        
        print(f"\nTest {i}:")
        print(f"  Input:  {text}")
        print(f"  Tokens: {tokens}")
        
        # Verify key features
        has_uppercase = any(t.isupper() or any(c.isupper() for c in t) for t in tokens)
        has_elongation = any(len(t) > 3 for t in tokens if t.isalpha())
        
        print(f"  ✓ Case preserved: {has_uppercase}")
        print(f"  ✓ Tokens found: {len(tokens)}")
    
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("When emoji 😂 is in GloVe vocab:")
    print("  → We KEEP it as-is (GloVe has an embedding for it)")
    print("\nWhen emoji 😭 is NOT in GloVe vocab:")
    print("  → We CONVERT to 'crying_emoji' (text token)")
    print("\nThis maximizes information preservation!")
    print("="*70)




