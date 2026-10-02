"""Persistent visual-word postings for candidate proposal, never pose verification."""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits


class KeyframeIndex:
    def __init__(self):
        self.vocabulary = None
        self.frames = {}
        self.histograms = {}
        self.postings = [dict() for _ in range(128)]
        self.pending = {}
        self.tokens = {}

    def upsert(self, ident, frame, descriptors):
        descriptors = np.asarray(descriptors)
        if descriptors.ndim != 2 or descriptors.shape[1] != 128 or descriptors.dtype == np.uint8:
            raise ValueError("Visual-word retrieval requires 128-dimensional float descriptors")
        token = (id(descriptors), len(descriptors))
        if self.tokens.get(ident) == token:
            return
        self.remove(ident)
        self.tokens[ident] = token
        self.frames[ident] = frame
        if self.vocabulary is None:
            self.pending[ident] = descriptors.copy()
            usable = [d for d in self.pending.values() if len(d) >= 2]
            if len(usable) < 5 or sum(len(d) for d in usable[:5]) < 128:
                return
            self.vocabulary = MiniBatchKMeans(
                n_clusters=128, random_state=0, n_init=1, batch_size=1024,
                max_iter=50, reassignment_ratio=0,
            )
            with threadpool_limits(limits=1):
                self.vocabulary.fit(np.vstack(usable[:5]).astype(np.float32))
            for key, desc in self.pending.items():
                self._insert(key, desc)
            self.pending.clear()
        else:
            self._insert(ident, descriptors)

    def _histogram(self, descriptors):
        if not len(descriptors):
            return np.zeros(128, np.float64)
        with threadpool_limits(limits=1):
            words = self.vocabulary.predict(np.asarray(descriptors, np.float32))
        counts = np.bincount(words, minlength=128).astype(np.float64)
        return counts / counts.sum()

    def _insert(self, ident, descriptors):
        histogram = self._histogram(descriptors)
        self.histograms[ident] = histogram
        for word in np.flatnonzero(histogram):
            self.postings[word][ident] = histogram[word]

    def remove(self, ident):
        histogram = self.histograms.pop(ident, None)
        if histogram is not None:
            for word in np.flatnonzero(histogram):
                self.postings[word].pop(ident, None)
        self.frames.pop(ident, None)
        self.pending.pop(ident, None)
        self.tokens.pop(ident, None)

    def query(self, descriptors, eligible, limit=20):
        """None means vocabulary startup: caller must use exhaustive retrieval."""
        if self.vocabulary is None:
            return None
        eligible = set(eligible)
        query = self._histogram(descriptors)
        idf = np.log((1 + len(self.histograms)) / (1 + np.array([len(p) for p in self.postings]))) + 1
        scores = {}
        for word in np.flatnonzero(query):
            for ident, weight in self.postings[word].items():
                if ident in eligible:
                    scores[ident] = scores.get(ident, 0.0) + weight * query[word] * idf[word] ** 2
        query_norm = np.linalg.norm(query * idf)
        ranked = [
            (score / max(query_norm * np.linalg.norm(self.histograms[ident] * idf), 1e-12), ident)
            for ident, score in scores.items()
        ]
        return [ident for _, ident in sorted(ranked, reverse=True)[:limit]]
