"""Sentence encoders for flow descriptions.

* SBERTEncoder  – sentence-transformers (all-MiniLM-L6-v2, DistilBERT/BERT variants, ...) if installed
                  and the model can be downloaded/cached.  Install:  pip install sentence-transformers
* LocalEncoder  – fully offline domain encoder: TF-IDF (word 1-2 grams) + truncated SVD (LSA), fit on the
                  unlabelled sentence corpus of all datasets. Used automatically when SBERT is unavailable.

Both encode unique sentences once (flow sentences are highly repetitive) and broadcast back.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..utils import get_logger

log = get_logger("nlp")


class LocalEncoder:
    name = "Local TF-IDF + LSA (offline)"

    def __init__(self, dim: int = 64, seed: int = 0):
        self.dim, self.seed = dim, seed

    def fit(self, corpus):
        uniq = np.unique(np.asarray(corpus, dtype=object))
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(uniq)
        M = self.vec.transform(uniq)
        k = max(2, min(self.dim, M.shape[1] - 1, len(uniq) - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed).fit(M)
        return self

    def encode(self, sentences):
        s = np.asarray(sentences, dtype=object)
        uniq, inv = np.unique(s, return_inverse=True)
        E = normalize(self.svd.transform(self.vec.transform(uniq)))
        return E[inv].astype(np.float32)


class SBERTEncoder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model = SentenceTransformer(model_name)
        self.name = f"SBERT ({model_name.split('/')[-1]})"

    def fit(self, corpus):
        return self

    def encode(self, sentences):
        s = np.asarray(sentences, dtype=object)
        uniq, inv = np.unique(s, return_inverse=True)
        E = self.model.encode(list(uniq), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(E, dtype=np.float32)[inv]


def get_encoder(cfg, corpus):
    if cfg["nlp"].get("prefer_sbert", True):
        try:
            enc = SBERTEncoder(cfg["nlp"]["sbert_model"])
            log.info(f"Using {enc.name}")
            return enc.fit(corpus)
        except Exception as e:  # no package / no internet / no cached model
            log.info(f"SBERT unavailable ({type(e).__name__}) -> using offline LocalEncoder")
    return LocalEncoder(cfg["nlp"]["dim"], cfg["seed"]).fit(corpus)
