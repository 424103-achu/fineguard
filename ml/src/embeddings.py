"""Phase 4: Word2Vec / GloVe embeddings and sentence averaging."""
import re

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def train_word2vec(texts, dim=200, seed=42, workers=4):
    from gensim.models import Word2Vec

    sentences = [tokenize(t) for t in texts]
    return Word2Vec(sentences=sentences, vector_size=dim, window=5, min_count=2,
                    sg=1, epochs=20, workers=workers, seed=seed)


def load_glove(name="glove-wiki-gigaword-200"):
    import gensim.downloader

    return gensim.downloader.load(name)


def average_vectors(texts, kv):
    """kv: a gensim KeyedVectors object (w2v.wv or loaded GloVe vectors)."""
    out = np.zeros((len(texts), kv.vector_size), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs = [kv[w] for w in tokenize(t) if w in kv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out
