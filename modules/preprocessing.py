"""
Tweet preprocessing pipeline for hate speech classification.

Cleaning steps:
  1. Emoji → text description  (emoji library)
  2. Lowercase
  3. Remove URLs
  4. Remove @mentions
  5. Hashtags: strip #, keep word
  6. Remove numbers
  7. Remove punctuation / special characters
  8. Tokenize
  9. Remove English stopwords
 10. Lemmatize

Two output variants are produced for each tweet:
  - tweet_clean : fully cleaned (used for BoW, TF-IDF, GloVe)
  - tweet_bert  : lighter cleaning, preserves sentence structure
                  (used as input to DistilBERT tokenizer)
"""

import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Resource download
# ---------------------------------------------------------------------------

def download_nltk_resources() -> None:
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for r in resources:
        nltk.download(r, quiet=True)


# ---------------------------------------------------------------------------
# Atomic cleaning helpers
# ---------------------------------------------------------------------------

def convert_emojis(text: str) -> str:
    """Replace emojis with space-delimited text tokens, e.g. 😡 → ' angry_face '."""
    return emoji.demojize(text, delimiters=(" ", " "))


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)


def remove_mentions(text: str) -> str:
    return re.sub(r"@\w+", " ", text)


def handle_hashtags(text: str) -> str:
    """Strip '#' but keep the word — #HateSpeech → HateSpeech."""
    return re.sub(r"#(\w+)", r"\1", text)


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", " ", text)


def remove_special_chars(text: str) -> str:
    """Keep only ASCII letters and whitespace."""
    return re.sub(r"[^a-zA-Z\s]", " ", text)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Full cleaning pipelines
# ---------------------------------------------------------------------------

def _base_clean(text: str) -> str:
    """Steps shared by both variants."""
    text = str(text)
    text = convert_emojis(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = handle_hashtags(text)
    text = text.lower()
    text = remove_numbers(text)
    text = remove_special_chars(text)
    text = collapse_whitespace(text)
    return text


def clean_for_traditional(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """
    Full pipeline for BoW / TF-IDF / GloVe:
      base clean → tokenize → remove stopwords → lemmatize → rejoin
    """
    text = _base_clean(text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 1
    ]
    return " ".join(tokens)


def clean_for_bert(text: str) -> str:
    """
    Lighter pipeline for DistilBERT:
      base clean only (no stopword removal / lemmatization so the
      model sees natural sentence structure).
    """
    return _base_clean(text)


# ---------------------------------------------------------------------------
# DataFrame-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "tweet") -> pd.DataFrame:
    """
    Apply both cleaning variants to every row of the dataframe.

    Adds two new columns:
      - tweet_clean : for traditional ML features + GloVe
      - tweet_bert  : for DistilBERT embeddings

    Parameters
    ----------
    df       : input dataframe (must contain `text_col`)
    text_col : name of the raw text column

    Returns
    -------
    A copy of `df` with the two new columns appended.
    """
    download_nltk_resources()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    df = df.copy()

    tqdm.pandas(desc="Cleaning (traditional)")
    df["tweet_clean"] = df[text_col].progress_apply(
        lambda x: clean_for_traditional(x, lemmatizer, stop_words)
    )

    tqdm.pandas(desc="Cleaning (BERT)")
    df["tweet_bert"] = df[text_col].progress_apply(clean_for_bert)

    return df


# ---------------------------------------------------------------------------
# Quick-look helper
# ---------------------------------------------------------------------------

def show_examples(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return a side-by-side comparison of raw vs cleaned tweets."""
    cols = ["tweet", "tweet_clean", "tweet_bert", "label"]
    available = [c for c in cols if c in df.columns]
    return df[available].sample(n, random_state=42).reset_index(drop=True)
