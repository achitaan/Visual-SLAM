"""Deterministic descriptor sketches shortlist views; geometry alone verifies recovery."""
import numpy as np


class KeyframeRetrieval:
    def __init__(self):
        self.histograms = {}
        self.projections = {}

    def sketch(self, descriptors):
        descriptors = np.asarray(descriptors)
        if descriptors.ndim != 2 or not len(descriptors):
            return np.zeros(256)
        dimensions = descriptors.shape[1]
        if dimensions not in self.projections:
            self.projections[dimensions] = np.random.default_rng(514).normal(size=(dimensions,8))
        centered = descriptors.astype(float) - descriptors.mean(axis=1,keepdims=True)
        bits = centered @ self.projections[dimensions] >= 0
        words = bits.astype(np.uint16) @ (1 << np.arange(8))
        histogram = np.bincount(words,minlength=256).astype(float)
        return histogram / max(1,np.linalg.norm(histogram))

    def update(self, keyframes):
        for ident,frame in keyframes.items():
            if ident not in self.histograms:
                self.histograms[ident] = self.sketch(frame.descriptors)
        for ident in set(self.histograms)-set(keyframes):
            del self.histograms[ident]

    def query(self, descriptors, count=8, allowed=None):
        ids=sorted(self.histograms if allowed is None else set(allowed)&self.histograms.keys())
        if not ids or not len(descriptors):return []
        query=self.sketch(descriptors)
        matrix=np.array([self.histograms[i] for i in ids])
        weights=np.log((len(matrix)+1)/(np.count_nonzero(matrix,axis=0)+1))+1
        matrix=matrix*weights;query=query*weights
        scores=matrix @ query / np.maximum(np.linalg.norm(matrix,axis=1)*np.linalg.norm(query),1e-12)
        order=sorted(range(len(ids)),key=lambda i:(-scores[i],ids[i]))[:count]
        return [ids[i] for i in order]
