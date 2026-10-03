"""Persistent visual-word postings for candidate proposal, never verification."""

import numpy as np
from scipy.cluster.vq import kmeans2, vq


class KeyframeIndex:
    """Detected-SIFT visual-word index used only to propose loop candidates.

    Geometry must still verify every candidate. ``None`` from ``query`` means
    the vocabulary is still starting up and the caller should use its fallback.
    """

    def __init__(self):
        self.vocabulary = None
        self.frames = {}
        self.histograms = {}
        self.postings = [dict() for _ in range(128)]
        self.pending = {}
        self.tokens = {}
        self.residuals = {}
        # Compact residual summaries distinguish frames with similar word counts.
        self.projection = np.linalg.qr(
            np.random.default_rng(1).normal(size=(128, 16))
        )[0].astype(np.float32)

    def upsert(self, ident, frame, descriptors):
        descriptors = np.asarray(descriptors)
        if (
            descriptors.ndim != 2
            or descriptors.shape[1] != 128
            or not np.issubdtype(descriptors.dtype, np.floating)
        ):
            raise ValueError("Visual-word retrieval requires 128-dimensional float SIFT descriptors")
        token = (id(descriptors), len(descriptors))
        if self.tokens.get(ident) == token:
            self.frames[ident] = frame
            return
        self.remove(ident)
        self.tokens[ident] = token
        self.frames[ident] = frame
        if self.vocabulary is None:
            self.pending[ident] = descriptors.copy()
            usable = [values for values in self.pending.values() if len(values) >= 2]
            if len(usable) < 5 or sum(len(values) for values in usable[:5]) < 128:
                return
            # SciPy is already required by the solvers. Avoid importing another
            # ML runtime solely to train this small vocabulary.
            training = np.vstack(usable[:5]).astype(np.float32)
            self.vocabulary, _ = kmeans2(
                training, 128, iter=20, minit="++", rng=np.random.default_rng(0)
            )
            for key, values in self.pending.items():
                self._insert(key, values)
            self.pending.clear()
        else:
            self._insert(ident, descriptors)

    def _encode(self, descriptors):
        if not len(descriptors):
            return np.zeros(128, np.float64), np.zeros((128, 16), np.float32)
        descriptors = np.asarray(descriptors, np.float32)
        words, _ = vq(descriptors, self.vocabulary)
        counts = np.bincount(words, minlength=128).astype(np.float64)
        residual = np.zeros((128, 16), np.float32)
        np.add.at(
            residual,
            words,
            (descriptors - self.vocabulary[words]) @ self.projection,
        )
        residual = np.sign(residual) * np.sqrt(np.abs(residual))
        residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
        residual /= max(np.linalg.norm(residual), 1e-12)
        return counts / counts.sum(), residual

    def _insert(self, ident, descriptors):
        histogram, residual = self._encode(descriptors)
        self.histograms[ident] = histogram
        self.residuals[ident] = residual
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
        self.residuals.pop(ident, None)

    def query(self, descriptors, eligible, limit=20):
        """Return a bounded, eligible shortlist, or ``None`` before training."""
        if self.vocabulary is None:
            return None
        eligible = set(eligible)
        query, residual = self._encode(descriptors)
        idf = np.log(
            (1 + len(self.histograms))
            / (1 + np.array([len(posting) for posting in self.postings]))
        ) + 1
        scores = {}
        for word in np.flatnonzero(query):
            for ident, weight in self.postings[word].items():
                if ident in eligible:
                    scores[ident] = scores.get(ident, 0.0) + weight * query[word] * idf[word] ** 2
        query_norm = np.linalg.norm(query * idf)
        ranked = [
            (
                0.5 * score / max(query_norm * np.linalg.norm(self.histograms[ident] * idf), 1e-12)
                + 0.5 * float(np.sum(residual * self.residuals[ident])),
                ident,
            )
            for ident, score in scores.items()
        ]
        ordered = [ident for _, ident in sorted(ranked, reverse=True)]
        # Reserve half the budget for adjacent views around the best indexed
        # hit. Exact geometric verification chooses the final reference.
        limit = max(0, int(limit))
        result = ordered[:max(1, limit // 2)] if limit else []
        if result:
            chronological = sorted(self.frames, key=lambda key: (self.frames[key], key))
            position = chronological.index(result[0])
            for distance in range(1, 6):
                for neighbor in (position - distance, position + distance):
                    if 0 <= neighbor < len(chronological):
                        ident = chronological[neighbor]
                        if ident in eligible and ident not in result:
                            result.append(ident)
                if len(result) >= limit:
                    break
        result.extend(ident for ident in ordered if ident not in result)
        return result[:limit]
