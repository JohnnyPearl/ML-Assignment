"""
Feature extraction module for hate speech classification.

Five methods are implemented:

  Traditional (sparse, frequency-based):
    1. Bag-of-Words (BoW)          — CountVectorizer, unigram
    2. TF-IDF (word)               — TfidfVectorizer, unigram + bigram
    3. Char n-gram TF-IDF          — TfidfVectorizer, char 3-5 gram
                                     (captures slang / misspellings)

  Modern (dense, semantic):
    4. GloVe Twitter 200d          — averaged pre-trained word vectors
                                     trained on 2B tweets (domain match)
    5. DistilBERT CLS token        — contextual embeddings, 768-dim

Outputs
-------
  BoW / TF-IDF / Char-TF-IDF  → scipy sparse (.npz) — too large for dense .npy
  GloVe / DistilBERT           → numpy dense (.npy)
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

import scipy.sparse as sp
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tqdm import tqdm


# ===========================================================================
# 1. Bag-of-Words
# ===========================================================================

def extract_bow(
    texts_train,
    texts_test,
    max_features: int = 10_000,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, CountVectorizer]:
    """
    Fit CountVectorizer on training texts, transform train + test.

    Parameters
    ----------
    texts_train   : iterable of clean strings (training set)
    texts_test    : iterable of clean strings (test set)
    max_features  : vocabulary cap

    Returns
    -------
    X_train, X_test (sparse csr_matrix), fitted vectorizer
    """
    vec = CountVectorizer(
        max_features=max_features,
        min_df=2,           # ignore very rare words
        ngram_range=(1, 1),
    )
    X_train = vec.fit_transform(texts_train)
    X_test = vec.transform(texts_test)
    print(f"  BoW  → train {X_train.shape}, test {X_test.shape}")
    return X_train, X_test, vec


# ===========================================================================
# 2. TF-IDF (word unigram + bigram)
# ===========================================================================

def extract_tfidf(
    texts_train,
    texts_test,
    max_features: int = 15_000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[sp.csr_matrix, sp.csr_matrix, TfidfVectorizer]:
    """
    Fit TF-IDF vectorizer (word-level, unigram + bigram by default).
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        ngram_range=ngram_range,
        sublinear_tf=True,   # apply log(1 + tf) to dampen frequency
    )
    X_train = vec.fit_transform(texts_train)
    X_test = vec.transform(texts_test)
    print(f"  TF-IDF (word {ngram_range}) → train {X_train.shape}, test {X_test.shape}")
    return X_train, X_test, vec


# ===========================================================================
# 3. Character n-gram TF-IDF
# ===========================================================================

def extract_char_tfidf(
    texts_train,
    texts_test,
    max_features: int = 10_000,
    ngram_range: Tuple[int, int] = (3, 5),
) -> Tuple[sp.csr_matrix, sp.csr_matrix, TfidfVectorizer]:
    """
    Character-level n-gram TF-IDF.
    Effective for Twitter data: handles abbreviations, slang, typos.
    Uses 'char_wb' analyser — pads each token with spaces so n-grams
    do not cross word boundaries (better interpretability).
    """
    vec = TfidfVectorizer(
        analyzer="char_wb",
        max_features=max_features,
        min_df=2,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    X_train = vec.fit_transform(texts_train)
    X_test = vec.transform(texts_test)
    print(f"  Char TF-IDF ({ngram_range}) → train {X_train.shape}, test {X_test.shape}")
    return X_train, X_test, vec


# ===========================================================================
# 4. GloVe Twitter 200d (averaged word vectors)
# ===========================================================================

def load_glove_vectors(glove_path: str) -> dict:
    """
    Load GloVe pre-trained vectors into a word → np.ndarray dict.

    Expected file: glove.twitter.27B.200d.txt
    Download:      https://nlp.stanford.edu/data/glove.twitter.27B.zip

    Parameters
    ----------
    glove_path : path to the .txt file

    Returns
    -------
    dict  {word: np.ndarray of shape (200,)}
    """
    print(f"Loading GloVe vectors from: {glove_path}")
    embeddings = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading GloVe"):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    print(f"  Loaded {len(embeddings):,} word vectors (dim={next(iter(embeddings.values())).shape[0]})")
    return embeddings


def _text_to_glove_vector(text: str, embeddings: dict, dim: int = 200) -> np.ndarray:
    """Average GloVe vectors for all known tokens in the text."""
    tokens = text.split()
    vectors = [embeddings[t] for t in tokens if t in embeddings]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(dim, dtype="float32")


def extract_glove(
    texts_train,
    texts_test,
    glove_path: str,
    dim: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute averaged GloVe sentence vectors.

    Parameters
    ----------
    texts_train : iterable of clean strings (training set)
    texts_test  : iterable of clean strings (test set)
    glove_path  : path to glove.twitter.27B.200d.txt
    dim         : embedding dimension (200)

    Returns
    -------
    X_train, X_test — numpy arrays of shape (n_samples, dim)
    """
    embeddings = load_glove_vectors(glove_path)

    texts_train = list(texts_train)
    texts_test = list(texts_test)

    X_train = np.array(
        [_text_to_glove_vector(t, embeddings, dim) for t in tqdm(texts_train, desc="GloVe (train)")],
        dtype="float32",
    )
    X_test = np.array(
        [_text_to_glove_vector(t, embeddings, dim) for t in tqdm(texts_test, desc="GloVe (test)")],
        dtype="float32",
    )
    print(f"  GloVe → train {X_train.shape}, test {X_test.shape}")
    return X_train, X_test


# ===========================================================================
# 5. DistilBERT CLS token embeddings
# ===========================================================================

def extract_distilbert(
    texts,
    batch_size: int = 32,
    model_name: str = "distilbert-base-uncased",
    device: Optional[str] = None,
    max_length: int = 128,
) -> np.ndarray:
    """
    Extract CLS-token embeddings from DistilBERT for a list of texts.

    The CLS token (index 0 of last hidden state) serves as the sentence
    representation.  Model weights are downloaded automatically on first run
    (~260 MB from HuggingFace Hub).

    Parameters
    ----------
    texts      : iterable of strings (use tweet_bert column — lighter clean)
    batch_size : number of texts per forward pass (reduce if OOM)
    model_name : HuggingFace model identifier
    device     : 'cuda', 'cpu', or None (auto-detect)
    max_length : max token length (Twitter tweets are short, 128 is enough)

    Returns
    -------
    numpy array of shape (n_texts, 768)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  DistilBERT running on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    texts = list(texts)
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="DistilBERT"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        # CLS token = first token of last hidden state
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

    embeddings = np.vstack(all_embeddings).astype("float32")
    print(f"  DistilBERT → {embeddings.shape}")
    return embeddings


# ===========================================================================
# Save / load helpers
# ===========================================================================

def save_sparse(matrix: sp.csr_matrix, path: str) -> None:
    """Save a scipy sparse matrix to .npz."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_npz(path, matrix)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"  Saved sparse → {path}  ({size_mb:.1f} MB)")


def save_dense(array: np.ndarray, path: str) -> None:
    """Save a numpy dense array to .npy."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"  Saved dense  → {path}  ({size_mb:.1f} MB)")


def load_sparse(path: str) -> sp.csr_matrix:
    return load_npz(path)


def load_dense(path: str) -> np.ndarray:
    return np.load(path)
