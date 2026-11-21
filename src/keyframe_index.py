"""Persistent visual-word postings for candidate proposal, never verification."""

import numpy as np


def _kmeans_pp_lloyd_blas(data, k=128, n_iter=20, rng=None):
    """Exact-parity BLAS replacement for scipy.cluster.vq.kmeans2(minit='++').

    Avoids SciPy's O(k^2 * N) repeated cdist calls during k-means++ seeding and
    unvectorized vq loops during Lloyd iterations while preserving the exact RNG
    draw sequence and float64 distance precision.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    data64 = np.ascontiguousarray(data, dtype=np.float64)
    n, d = data64.shape
    x_norm = np.einsum("ij,ij->i", data64, data64)
    init = np.empty((k, d), dtype=np.float64)
    min_d2 = np.full(n, np.inf, dtype=np.float64)

    for i in range(k):
        if i == 0:
            data_idx = int(rng.integers(0, n))
        else:
            c = init[i - 1]
            d2 = np.maximum(x_norm + float(c @ c) - 2.0 * (data64 @ c), 0.0)
            np.minimum(min_d2, d2, out=min_d2)
            total = float(min_d2.sum())
            if total > 0.0 and np.isfinite(total):
                cumprobs = (min_d2 / total).cumsum()
                r = rng.uniform()
                data_idx = min(int(np.searchsorted(cumprobs, r)), n - 1)
            else:
                rng.uniform()
                data_idx = 0
        init[i] = data64[data_idx]

    code_book = init
    neg2_data = -2.0 * data64
    for _ in range(n_iter):
        c_norm = np.einsum("ij,ij->i", code_book, code_book)
        dist = neg2_data @ code_book.T
        dist += c_norm[None, :]
        label = np.argmin(dist, axis=1)
        counts = np.bincount(label, minlength=k)
        order = np.argsort(label)
        sorted_data = data64[order]
        starts = np.zeros(k, dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        nz = np.flatnonzero(counts > 0)
        if len(nz):
            sums = np.add.reduceat(sorted_data, starts[nz], axis=0)
            new_code_book = code_book.copy()
            new_code_book[nz] = sums / counts[nz, None]
            code_book = new_code_book
    return np.ascontiguousarray(code_book, dtype=np.float32)


class KeyframeIndex:
    """Detected-SIFT visual-word index used only to propose loop candidates.

    Geometry must still verify every candidate. ``None`` from ``query`` means
    the vocabulary is still starting up and the caller should use its fallback.
    """

    def __init__(self):
        self.vocabulary = None
        self._vocab_norm = None
        self._vocab_proj = None
        self.frames = {}
        self.histograms = {}
        self.postings = [dict() for _ in range(128)]
        self.pending = {}
        self.tokens = {}
        self.residuals = {}
        # Compact residual summaries distinguish frames with similar word counts.
        self.projection = np.ascontiguousarray(
            np.linalg.qr(np.random.default_rng(1).normal(size=(128, 16)))[0],
            dtype=np.float32,
        )
        # Synchronized dense tables for vectorized query scoring.
        self._dense_cap = 64
        self._dense_size = 0
        self._dense_ids = []
        self._id_to_row = {}
        self._hist_matrix = np.empty((self._dense_cap, 128), dtype=np.float64)
        self._res_matrix = np.empty((self._dense_cap, 128 * 16), dtype=np.float32)
        self._df = np.zeros(128, dtype=np.int32)
        # Cache read-only array tokens -> ident so querying the just-upserted
        # current keyframe avoids redundant descriptor quantization.
        self._ro_token_to_ident = {}

    def _grow_dense(self):
        new_cap = self._dense_cap * 2
        new_hist = np.empty((new_cap, 128), dtype=np.float64)
        new_res = np.empty((new_cap, 128 * 16), dtype=np.float32)
        new_hist[: self._dense_size] = self._hist_matrix[: self._dense_size]
        new_res[: self._dense_size] = self._res_matrix[: self._dense_size]
        self._hist_matrix = new_hist
        self._res_matrix = new_res
        self._dense_cap = new_cap

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
        if not descriptors.flags.writeable:
            self._ro_token_to_ident[token] = ident
        self.frames[ident] = frame
        if self.vocabulary is None:
            self.pending[ident] = descriptors.copy()
            usable = [values for values in self.pending.values() if len(values) >= 2]
            if len(usable) < 5 or sum(len(values) for values in usable[:5]) < 128:
                return
            training = np.vstack(usable[:5]).astype(np.float32)
            self.vocabulary = _kmeans_pp_lloyd_blas(
                training, k=128, n_iter=20, rng=np.random.default_rng(0)
            )
            self._vocab_norm = np.einsum("ij,ij->i", self.vocabulary, self.vocabulary)
            self._vocab_proj = np.ascontiguousarray(
                self.vocabulary @ self.projection, dtype=np.float32
            )
            for key, values in self.pending.items():
                self._insert(key, values)
            self.pending.clear()
        else:
            self._insert(ident, descriptors)

    def _encode(self, descriptors):
        if not len(descriptors):
            return np.zeros(128, np.float64), np.zeros((128, 16), np.float32)
        desc = np.ascontiguousarray(descriptors, dtype=np.float32)
        dist = np.multiply(desc, np.float32(-2.0)) @ self.vocabulary.T
        dist += self._vocab_norm[None, :]
        words = np.argmin(dist, axis=1)
        counts = np.bincount(words, minlength=128).astype(np.float64)
        residual = np.zeros((128, 16), np.float32)
        diff_proj = (desc @ self.projection) - self._vocab_proj[words]
        np.add.at(residual, words, diff_proj)
        residual = np.sign(residual) * np.sqrt(np.abs(residual))
        residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
        residual /= max(np.linalg.norm(residual), 1e-12)
        return counts / counts.sum(), residual

    def _insert(self, ident, descriptors):
        histogram, residual = self._encode(descriptors)
        self.histograms[ident] = histogram
        self.residuals[ident] = residual
        nz = np.flatnonzero(histogram)
        for word in nz:
            self.postings[word][ident] = histogram[word]
        self._df[nz] += 1
        if self._dense_size >= self._dense_cap:
            self._grow_dense()
        row = self._dense_size
        self._dense_size += 1
        self._dense_ids.append(ident)
        self._id_to_row[ident] = row
        self._hist_matrix[row] = histogram
        self._res_matrix[row] = residual.ravel()

    def remove(self, ident):
        old_token = self.tokens.pop(ident, None)
        if old_token is not None and self._ro_token_to_ident.get(old_token) == ident:
            self._ro_token_to_ident.pop(old_token, None)
        histogram = self.histograms.pop(ident, None)
        if histogram is not None:
            nz = np.flatnonzero(histogram)
            for word in nz:
                if self.postings[word].pop(ident, None) is not None:
                    self._df[word] -= 1
        row = self._id_to_row.pop(ident, None)
        if row is not None:
            last = self._dense_size - 1
            if row != last:
                moved_id = self._dense_ids[last]
                self._dense_ids[row] = moved_id
                self._hist_matrix[row] = self._hist_matrix[last]
                self._res_matrix[row] = self._res_matrix[last]
                self._id_to_row[moved_id] = row
            self._dense_ids.pop()
            self._dense_size = last
        self.frames.pop(ident, None)
        self.pending.pop(ident, None)
        self.residuals.pop(ident, None)

    def query(self, descriptors, eligible, limit=20):
        """Return a bounded, eligible shortlist, or ``None`` before training."""
        if self.vocabulary is None:
            return None
        eligible = set(eligible)
        cached_ident = None
        if isinstance(descriptors, np.ndarray) and not descriptors.flags.writeable:
            token = (id(descriptors), len(descriptors))
            cand_id = self._ro_token_to_ident.get(token)
            if cand_id is not None and cand_id in self.histograms:
                cached_ident = cand_id
        if cached_ident is not None:
            query = self.histograms[cached_ident]
            residual = self.residuals[cached_ident]
        else:
            query, residual = self._encode(descriptors)

        limit = max(0, int(limit))
        if not limit or not eligible or not self._dense_size:
            return []

        id_to_row = self._id_to_row
        elig_rows = []
        elig_ids = []
        for ident in eligible:
            row = id_to_row.get(ident)
            if row is not None:
                elig_rows.append(row)
                elig_ids.append(ident)
        if not elig_rows:
            return []

        row_arr = np.asarray(elig_rows, dtype=np.intp)
        idf = np.log((1 + len(self.histograms)) / (1 + self._df)) + 1.0
        q_idf = query * idf
        query_norm = float(np.linalg.norm(q_idf))
        h_sub = self._hist_matrix[row_arr]
        h_idf = h_sub * idf[None, :]
        dots = h_idf @ q_idf
        # Match original postings-intersection semantics: only keyframes sharing at least
        # one non-zero visual word with the query are ranked.
        active = np.flatnonzero(dots > 0.0)
        if not len(active):
            return []

        act_rows = row_arr[active]
        h_norms = np.linalg.norm(h_idf[active], axis=1)
        bow = 0.5 * dots[active] / np.maximum(query_norm * h_norms, 1e-12)
        vlad = 0.5 * (self._res_matrix[act_rows] @ residual.ravel()).astype(np.float64)
        total = bow + vlad

        ranked = [(float(total[idx]), elig_ids[int(active[idx])]) for idx in range(len(active))]
        ordered = [ident for _, ident in sorted(ranked, reverse=True)]
        # Reserve half the budget for adjacent views around the best indexed
        # hit. Exact geometric verification chooses the final reference.
        result = ordered[:max(1, limit // 2)]
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
