// Fast exact analytic sparse block-6x6 SE(3) Levenberg-Marquardt pose graph solver.
// Uses Minimum-Degree (MD) node ordering + symbolic/numeric block-6x6 sparse Cholesky (LL^T).
// Matches src/pose_graph.py:optimize_sparse_se3 objective, Jacobians, Huber loss, and LM update.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

struct Mat3 {
    double m[9];
};

struct Vec3 {
    double v[3];
};

struct Mat6 {
    double m[36];
};

inline void mat3_mul(const double* A, const double* B, double* C) {
    for (int i = 0; i < 3; ++i) {
        const double a0 = A[i * 3 + 0];
        const double a1 = A[i * 3 + 1];
        const double a2 = A[i * 3 + 2];
        C[i * 3 + 0] = a0 * B[0] + a1 * B[3] + a2 * B[6];
        C[i * 3 + 1] = a0 * B[1] + a1 * B[4] + a2 * B[7];
        C[i * 3 + 2] = a0 * B[2] + a1 * B[5] + a2 * B[8];
    }
}

inline void mat3_mul_T_left(const double* A, const double* B, double* C) {
    // C = A^T * B
    for (int i = 0; i < 3; ++i) {
        const double a0 = A[0 * 3 + i];
        const double a1 = A[1 * 3 + i];
        const double a2 = A[2 * 3 + i];
        C[i * 3 + 0] = a0 * B[0] + a1 * B[3] + a2 * B[6];
        C[i * 3 + 1] = a0 * B[1] + a1 * B[4] + a2 * B[7];
        C[i * 3 + 2] = a0 * B[2] + a1 * B[5] + a2 * B[8];
    }
}

inline void mat3_mul_T_right(const double* A, const double* B, double* C) {
    // C = A * B^T
    for (int i = 0; i < 3; ++i) {
        const double a0 = A[i * 3 + 0];
        const double a1 = A[i * 3 + 1];
        const double a2 = A[i * 3 + 2];
        C[i * 3 + 0] = a0 * B[0] + a1 * B[1] + a2 * B[2];
        C[i * 3 + 1] = a0 * B[3] + a1 * B[4] + a2 * B[5];
        C[i * 3 + 2] = a0 * B[6] + a1 * B[7] + a2 * B[8];
    }
}

inline void mat3_vec(const double* A, const double* x, double* y) {
    y[0] = A[0] * x[0] + A[1] * x[1] + A[2] * x[2];
    y[1] = A[3] * x[0] + A[4] * x[1] + A[5] * x[2];
    y[2] = A[6] * x[0] + A[7] * x[1] + A[8] * x[2];
}

inline void mat3_T_vec(const double* A, const double* x, double* y) {
    y[0] = A[0] * x[0] + A[3] * x[1] + A[6] * x[2];
    y[1] = A[1] * x[0] + A[4] * x[1] + A[7] * x[2];
    y[2] = A[2] * x[0] + A[5] * x[1] + A[8] * x[2];
}

inline void so3_log(const double* R, double* phi) {
    // Robust SO(3) logarithm matching scipy Rotation.from_matrix().as_rotvec().
    double tr = R[0] + R[4] + R[8];
    double cos_t = 0.5 * (tr - 1.0);
    if (cos_t > 1.0) cos_t = 1.0;
    if (cos_t < -1.0) cos_t = -1.0;
    double theta = std::acos(cos_t);
    double w0 = R[7] - R[5];
    double w1 = R[2] - R[6];
    double w2 = R[3] - R[1];
    if (theta < 1e-6) {
        double scale = 0.5 + (theta * theta) / 12.0;
        phi[0] = scale * w0;
        phi[1] = scale * w1;
        phi[2] = scale * w2;
        return;
    }
    if (theta < 3.14159265358979323846 - 1e-4) {
        double scale = 0.5 * theta / std::sin(theta);
        phi[0] = scale * w0;
        phi[1] = scale * w1;
        phi[2] = scale * w2;
        return;
    }
    // Near-pi fallback via quaternion extraction.
    double q[4];
    if (R[0] >= R[4] && R[0] >= R[8]) {
        double s = 2.0 * std::sqrt(std::max(1.0 + R[0] - R[4] - R[8], 1e-18));
        q[0] = (R[7] - R[5]) / s;
        q[1] = 0.25 * s;
        q[2] = (R[1] + R[3]) / s;
        q[3] = (R[2] + R[6]) / s;
    } else if (R[4] >= R[8]) {
        double s = 2.0 * std::sqrt(std::max(1.0 + R[4] - R[0] - R[8], 1e-18));
        q[0] = (R[2] - R[6]) / s;
        q[1] = (R[1] + R[3]) / s;
        q[2] = 0.25 * s;
        q[3] = (R[5] + R[7]) / s;
    } else {
        double s = 2.0 * std::sqrt(std::max(1.0 + R[8] - R[0] - R[4], 1e-18));
        q[0] = (R[3] - R[1]) / s;
        q[1] = (R[2] + R[6]) / s;
        q[2] = (R[5] + R[7]) / s;
        q[3] = 0.25 * s;
    }
    if (q[0] < 0.0) {
        q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3];
    }
    double vn = std::sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    double ang = 2.0 * std::atan2(vn, q[0]);
    double inv_vn = (vn > 1e-15) ? (ang / vn) : 2.0;
    phi[0] = q[1] * inv_vn;
    phi[1] = q[2] * inv_vn;
    phi[2] = q[3] * inv_vn;
}

inline void so3_exp(const double* phi, double* R) {
    double t2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    double A, B;
    if (t2 < 1e-10) {
        A = 1.0 - t2 / 6.0 + (t2 * t2) / 120.0;
        B = 0.5 - t2 / 24.0 + (t2 * t2) / 720.0;
    } else {
        double t = std::sqrt(t2);
        A = std::sin(t) / t;
        B = (1.0 - std::cos(t)) / t2;
    }
    double wx = phi[0], wy = phi[1], wz = phi[2];
    double xx = wx * wx, yy = wy * wy, zz = wz * wz;
    double xy = wx * wy, xz = wx * wz, yz = wy * wz;
    R[0] = 1.0 - B * (yy + zz);
    R[1] = B * xy - A * wz;
    R[2] = B * xz + A * wy;
    R[3] = B * xy + A * wz;
    R[4] = 1.0 - B * (xx + zz);
    R[5] = B * yz - A * wx;
    R[6] = B * xz - A * wy;
    R[7] = B * yz + A * wx;
    R[8] = 1.0 - B * (xx + yy);
}

inline void inv_right_jacobian(const double* phi, double* Jr_inv) {
    double t2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    double c2;
    if (t2 < 1e-10) {
        c2 = 1.0 / 12.0;
    } else {
        double t = std::sqrt(t2);
        double half_t = 0.5 * t;
        double s = std::sin(half_t);
        if (s < 1e-12) s = 1e-12;
        double cot_half = std::cos(half_t) / s;
        c2 = (1.0 - half_t * cot_half) / t2;
    }
    double wx = phi[0], wy = phi[1], wz = phi[2];
    double xx = wx * wx, yy = wy * wy, zz = wz * wz;
    double xy = wx * wy, xz = wx * wz, yz = wy * wz;
    // I + 0.5*[phi]_x + c2*[phi]_x^2
    Jr_inv[0] = 1.0 - c2 * (yy + zz);
    Jr_inv[1] = -0.5 * wz + c2 * xy;
    Jr_inv[2] =  0.5 * wy + c2 * xz;
    Jr_inv[3] =  0.5 * wz + c2 * xy;
    Jr_inv[4] = 1.0 - c2 * (xx + zz);
    Jr_inv[5] = -0.5 * wx + c2 * yz;
    Jr_inv[6] = -0.5 * wy + c2 * xz;
    Jr_inv[7] =  0.5 * wx + c2 * yz;
    Jr_inv[8] = 1.0 - c2 * (xx + yy);
}

// 6x6 dense lower-triangular Cholesky: A = L * L^T (writes L into out_L).
inline bool chol6(const double* A, double* L) {
    std::memset(L, 0, 36 * sizeof(double));
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[i * 6 + j];
            for (int k = 0; k < j; ++k) {
                s -= L[i * 6 + k] * L[j * 6 + k];
            }
            if (i == j) {
                if (!(s > 1e-15)) return false;
                L[i * 6 + j] = std::sqrt(s);
            } else {
                L[i * 6 + j] = s / L[j * 6 + j];
            }
        }
    }
    return true;
}

// Solve X * L^T = B where L is 6x6 lower-triangular, B is 6x6, in-place on B -> X.
inline void solve_right_LT_6(const double* L, double* B) {
    for (int r = 0; r < 6; ++r) {
        double* row = B + r * 6;
        for (int c = 0; c < 6; ++c) {
            double s = row[c];
            for (int k = 0; k < c; ++k) {
                s -= row[k] * L[c * 6 + k];
            }
            row[c] = s / L[c * 6 + c];
        }
    }
}

// C -= A * B^T (6x6)
inline void sub_mul_T_6(double* C, const double* A, const double* B) {
    for (int i = 0; i < 6; ++i) {
        const double* ai = A + i * 6;
        double* ci = C + i * 6;
        for (int j = 0; j < 6; ++j) {
            const double* bj = B + j * 6;
            ci[j] -= ai[0] * bj[0] + ai[1] * bj[1] + ai[2] * bj[2]
                   + ai[3] * bj[3] + ai[4] * bj[4] + ai[5] * bj[5];
        }
    }
}

// C -= A * A^T (6x6, symmetric lower triangle + mirror)
inline void sub_syrk_6(double* C, const double* A) {
    for (int i = 0; i < 6; ++i) {
        const double* ai = A + i * 6;
        for (int j = 0; j <= i; ++j) {
            const double* aj = A + j * 6;
            double s = ai[0] * aj[0] + ai[1] * aj[1] + ai[2] * aj[2]
                     + ai[3] * aj[3] + ai[4] * aj[4] + ai[5] * aj[5];
            C[i * 6 + j] -= s;
            if (i != j) C[j * 6 + i] -= s;
        }
    }
}

struct SymbolicFactor {
    int n_free;
    std::vector<int> perm;       // perm[new_idx] = orig_free_idx
    std::vector<int> inv_perm;   // inv_perm[orig_free_idx] = new_idx
    std::vector<int> col_ptr;    // size n_free + 1 (off-diagonal lower-tri blocks per col)
    std::vector<int> row_idx;    // sorted row > col for each off-diagonal block

    int find_block(int col, int row) const {
        int lo = col_ptr[col];
        int hi = col_ptr[col + 1];
        auto it = std::lower_bound(row_idx.begin() + lo, row_idx.begin() + hi, row);
        if (it != row_idx.begin() + hi && *it == row) {
            return static_cast<int>(it - row_idx.begin());
        }
        return -1;
    }
};

SymbolicFactor build_symbolic_md(int n_free, int n_edges, const int32_t* fi, const int32_t* si) {
    SymbolicFactor sym;
    sym.n_free = n_free;
    sym.perm.resize(n_free);
    sym.inv_perm.assign(n_free, -1);

    std::vector<std::vector<int>> adj(n_free);
    for (int e = 0; e < n_edges; ++e) {
        int u = fi[e] - 1;
        int v = si[e] - 1;
        if (u >= 0 && v >= 0 && u != v) {
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
    }
    for (int u = 0; u < n_free; ++u) {
        std::sort(adj[u].begin(), adj[u].end());
        adj[u].erase(std::unique(adj[u].begin(), adj[u].end()), adj[u].end());
    }

    // Minimum-Degree elimination.
    std::vector<uint8_t> eliminated(n_free, 0);
    std::vector<std::vector<int>> orig_fill_neighbors(n_free);
    for (int step = 0; step < n_free; ++step) {
        int best_u = -1;
        size_t best_deg = static_cast<size_t>(-1);
        for (int u = 0; u < n_free; ++u) {
            if (eliminated[u]) continue;
            size_t d = adj[u].size();
            if (d < best_deg) {
                best_deg = d;
                best_u = u;
                if (d <= 1) break;
            }
        }
        eliminated[best_u] = 1;
        sym.perm[step] = best_u;
        sym.inv_perm[best_u] = step;

        const auto& nbrs = adj[best_u];
        orig_fill_neighbors[step] = nbrs;

        // Remove best_u from neighbors and connect all pairs in nbrs (clique fill).
        for (int w : nbrs) {
            auto& aw = adj[w];
            aw.erase(std::remove(aw.begin(), aw.end(), best_u), aw.end());
        }
        const size_t k = nbrs.size();
        for (size_t a = 0; a < k; ++a) {
            int wa = nbrs[a];
            auto& awa = adj[wa];
            for (size_t b = a + 1; b < k; ++b) {
                int wb = nbrs[b];
                auto it = std::lower_bound(awa.begin(), awa.end(), wb);
                if (it == awa.end() || *it != wb) {
                    awa.insert(it, wb);
                    auto& awb = adj[wb];
                    auto it2 = std::lower_bound(awb.begin(), awb.end(), wa);
                    awb.insert(it2, wa);
                }
            }
        }
        adj[best_u].clear();
    }

    // Convert fill neighbors to sorted eliminated-order row indices (> col).
    sym.col_ptr.resize(n_free + 1, 0);
    for (int c = 0; c < n_free; ++c) {
        std::vector<int> rows;
        rows.reserve(orig_fill_neighbors[c].size());
        for (int w : orig_fill_neighbors[c]) {
            int r = sym.inv_perm[w];
            if (r > c) rows.push_back(r);
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        sym.col_ptr[c] = static_cast<int>(sym.row_idx.size());
        sym.row_idx.insert(sym.row_idx.end(), rows.begin(), rows.end());
    }
    sym.col_ptr[n_free] = static_cast<int>(sym.row_idx.size());
    return sym;
}

bool factor_and_solve(
    const SymbolicFactor& sym,
    const std::vector<Mat6>& H_diag,
    const std::vector<Mat6>& H_sub,
    const std::vector<double>& g_perm,
    double lam,
    std::vector<Mat6>& L_diag,
    std::vector<Mat6>& L_sub,
    std::vector<double>& step_orig
) {
    const int n = sym.n_free;
    L_diag = H_diag;
    L_sub = H_sub;

    for (int c = 0; c < n; ++c) {
        double* D = L_diag[c].m;
        for (int k = 0; k < 6; ++k) {
            double dk = D[k * 6 + k];
            double damp = lam * (dk > 1e-6 ? dk : 1e-6);
            D[k * 6 + k] = dk + damp;
        }
    }

    // Right-looking sparse block-6x6 Cholesky.
    for (int c = 0; c < n; ++c) {
        double Lcc[36];
        if (!chol6(L_diag[c].m, Lcc)) {
            return false;
        }
        std::memcpy(L_diag[c].m, Lcc, 36 * sizeof(double));

        const int start = sym.col_ptr[c];
        const int end = sym.col_ptr[c + 1];
        for (int idx = start; idx < end; ++idx) {
            solve_right_LT_6(Lcc, L_sub[idx].m);
        }

        // Schur complement update to trailing columns.
        for (int a = start; a < end; ++a) {
            const int ra = sym.row_idx[a];
            const double* Lac = L_sub[a].m;
            sub_syrk_6(L_diag[ra].m, Lac);
            for (int b = a + 1; b < end; ++b) {
                const int rb = sym.row_idx[b];
                const int target = sym.find_block(ra, rb);
                if (target >= 0) {
                    sub_mul_T_6(L_sub[target].m, L_sub[b].m, Lac);
                }
            }
        }
    }

    // Forward substitution: L * y = -g_perm.
    std::vector<double> y(n * 6);
    for (int i = 0; i < n * 6; ++i) y[i] = -g_perm[i];
    for (int c = 0; c < n; ++c) {
        const double* Lcc = L_diag[c].m;
        double* yc = y.data() + c * 6;
        for (int r = 0; r < 6; ++r) {
            double s = yc[r];
            for (int k = 0; k < r; ++k) s -= Lcc[r * 6 + k] * yc[k];
            yc[r] = s / Lcc[r * 6 + r];
        }
        const int start = sym.col_ptr[c];
        const int end = sym.col_ptr[c + 1];
        for (int idx = start; idx < end; ++idx) {
            const int r = sym.row_idx[idx];
            const double* Lrc = L_sub[idx].m;
            double* yr = y.data() + r * 6;
            for (int i = 0; i < 6; ++i) {
                const double* row = Lrc + i * 6;
                yr[i] -= row[0] * yc[0] + row[1] * yc[1] + row[2] * yc[2]
                       + row[3] * yc[3] + row[4] * yc[4] + row[5] * yc[5];
            }
        }
    }

    // Backward substitution: L^T * z = y.
    for (int c = n - 1; c >= 0; --c) {
        double* zc = y.data() + c * 6;
        const int start = sym.col_ptr[c];
        const int end = sym.col_ptr[c + 1];
        for (int idx = start; idx < end; ++idx) {
            const int r = sym.row_idx[idx];
            const double* Lrc = L_sub[idx].m;
            const double* zr = y.data() + r * 6;
            for (int k = 0; k < 6; ++k) {
                zc[k] -= Lrc[0 * 6 + k] * zr[0] + Lrc[1 * 6 + k] * zr[1]
                       + Lrc[2 * 6 + k] * zr[2] + Lrc[3 * 6 + k] * zr[3]
                       + Lrc[4 * 6 + k] * zr[4] + Lrc[5 * 6 + k] * zr[5];
            }
        }
        const double* Lcc = L_diag[c].m;
        for (int r = 5; r >= 0; --r) {
            double s = zc[r];
            for (int k = r + 1; k < 6; ++k) s -= Lcc[k * 6 + r] * zc[k];
            zc[r] = s / Lcc[r * 6 + r];
        }
    }

    // Unpermute into original free-node order.
    for (int c = 0; c < n; ++c) {
        int orig = sym.perm[c];
        std::memcpy(step_orig.data() + orig * 6, y.data() + c * 6, 6 * sizeof(double));
    }
    return true;
}

struct EdgeEval {
    double R_hat[9];
    double t_hat[3];
    double dR[9];
    double phi[3];
    double res[6];
};

double eval_graph(
    int n_edges,
    const int32_t* fi,
    const int32_t* si,
    const std::vector<Mat3>& Rm_T,
    const std::vector<Vec3>& tm,
    const double* W,
    const std::vector<Mat3>& R_cur,
    const std::vector<Vec3>& t_cur,
    double delta_h,
    std::vector<EdgeEval>& ev
) {
    double total_cost = 0.0;
    for (int e = 0; e < n_edges; ++e) {
        const int i = fi[e];
        const int j = si[e];
        const double* Ri = R_cur[i].m;
        const double* ti = t_cur[i].v;
        const double* Rj = R_cur[j].m;
        const double* tj = t_cur[j].v;

        EdgeEval& out = ev[e];
        mat3_mul_T_left(Ri, Rj, out.R_hat);
        double d_tij[3] = {tj[0] - ti[0], tj[1] - ti[1], tj[2] - ti[2]};
        mat3_T_vec(Ri, d_tij, out.t_hat);

        mat3_mul(Rm_T[e].m, out.R_hat, out.dR);
        double t_diff[3] = {
            out.t_hat[0] - tm[e].v[0],
            out.t_hat[1] - tm[e].v[1],
            out.t_hat[2] - tm[e].v[2]
        };
        double dt[3];
        mat3_vec(Rm_T[e].m, t_diff, dt);
        so3_log(out.dR, out.phi);

        double err[6] = {out.phi[0], out.phi[1], out.phi[2], dt[0], dt[1], dt[2]};
        const double* We = W + e * 36;
        for (int r = 0; r < 6; ++r) {
            const double* wr = We + r * 6;
            double v = wr[0] * err[0] + wr[1] * err[1] + wr[2] * err[2]
                     + wr[3] * err[3] + wr[4] * err[4] + wr[5] * err[5];
            out.res[r] = v;
            double a = std::fabs(v);
            total_cost += (a <= delta_h) ? (0.5 * v * v) : (delta_h * (a - 0.5 * delta_h));
        }
    }
    return total_cost;
}

}  // namespace

extern "C" int solve_sparse_se3_pgo(
    int n_poses,
    double* poses,          // [n_poses, 4, 4] row-major, in/out
    int n_edges,
    const int32_t* fi,      // [n_edges]
    const int32_t* si,      // [n_edges]
    const double* meas,     // [n_edges, 4, 4] row-major
    const double* W,        // [n_edges, 6, 6] row-major
    double huber_scale,
    int max_iter,
    double* out_stats       // [5]: initial_cost, final_cost, optimality, nfev, njev
) {
    if (n_poses <= 1 || n_edges <= 0) {
        if (out_stats) {
            out_stats[0] = 0.0; out_stats[1] = 0.0; out_stats[2] = 0.0;
            out_stats[3] = 0.0; out_stats[4] = 0.0;
        }
        return 1;
    }

    const int n_free = n_poses - 1;
    const double delta_h = (huber_scale > 1e-6) ? huber_scale : 1e-6;
    const int it_cap = std::max(1, std::min(max_iter, 60));

    // Unpack poses & measurements into compact 3x3 + 3 arrays.
    std::vector<Mat3> R_cur(n_poses), R_cand(n_poses);
    std::vector<Vec3> t_cur(n_poses), t_cand(n_poses);
    for (int k = 0; k < n_poses; ++k) {
        const double* P = poses + k * 16;
        R_cur[k].m[0] = P[0];  R_cur[k].m[1] = P[1];  R_cur[k].m[2] = P[2];
        R_cur[k].m[3] = P[4];  R_cur[k].m[4] = P[5];  R_cur[k].m[5] = P[6];
        R_cur[k].m[6] = P[8];  R_cur[k].m[7] = P[9];  R_cur[k].m[8] = P[10];
        t_cur[k].v[0] = P[3];  t_cur[k].v[1] = P[7];  t_cur[k].v[2] = P[11];
    }
    R_cand[0] = R_cur[0];
    t_cand[0] = t_cur[0];

    std::vector<Mat3> Rm_T(n_edges);
    std::vector<Vec3> tm(n_edges);
    for (int e = 0; e < n_edges; ++e) {
        const double* M = meas + e * 16;
        // Transpose of M[:3, :3]
        Rm_T[e].m[0] = M[0];  Rm_T[e].m[1] = M[4];  Rm_T[e].m[2] = M[8];
        Rm_T[e].m[3] = M[1];  Rm_T[e].m[4] = M[5];  Rm_T[e].m[5] = M[9];
        Rm_T[e].m[6] = M[2];  Rm_T[e].m[7] = M[6];  Rm_T[e].m[8] = M[10];
        tm[e].v[0] = M[3];    tm[e].v[1] = M[7];    tm[e].v[2] = M[11];
    }

    // Build symbolic MD Cholesky structure once for the whole solve.
    SymbolicFactor sym = build_symbolic_md(n_free, n_edges, fi, si);

    // Pre-map each edge's off-diagonal block index in sym (for O(1) assembly).
    std::vector<int> edge_offdiag_idx(n_edges, -1);
    std::vector<uint8_t> edge_swapped(n_edges, 0);
    for (int e = 0; e < n_edges; ++e) {
        int u = fi[e] - 1;
        int v = si[e] - 1;
        if (u >= 0 && v >= 0) {
            int pu = sym.inv_perm[u];
            int pv = sym.inv_perm[v];
            if (pu < pv) {
                edge_offdiag_idx[e] = sym.find_block(pu, pv);
                edge_swapped[e] = 0;
            } else {
                edge_offdiag_idx[e] = sym.find_block(pv, pu);
                edge_swapped[e] = 1;
            }
        }
    }

    std::vector<EdgeEval> ev(n_edges), ev_cand(n_edges);
    double cur_cost = eval_graph(n_edges, fi, si, Rm_T, tm, W, R_cur, t_cur, delta_h, ev);
    const double initial_cost = cur_cost;

    std::vector<Mat6> H_diag(n_free), L_diag(n_free);
    std::vector<Mat6> H_sub(sym.row_idx.size()), L_sub(sym.row_idx.size());
    std::vector<double> g_perm(n_free * 6), step_orig(n_free * 6);

    double lam = 1e-3;
    int nfev = 1;
    int njev = 0;
    bool converged = false;
    double optimality = 1e300;

    for (int it = 0; it < it_cap; ++it) {
        njev += 1;
        std::memset(H_diag.data(), 0, H_diag.size() * sizeof(Mat6));
        if (!H_sub.empty()) {
            std::memset(H_sub.data(), 0, H_sub.size() * sizeof(Mat6));
        }
        std::fill(g_perm.begin(), g_perm.end(), 0.0);

        for (int e = 0; e < n_edges; ++e) {
            const int i = fi[e];
            const int j = si[e];
            const EdgeEval& ee = ev[e];
            const double* We = W + e * 36;

            double irls[6];
            for (int r = 0; r < 6; ++r) {
                double ar = std::fabs(ee.res[r]);
                irls[r] = (ar <= delta_h) ? 1.0 : (delta_h / std::max(ar, 1e-12));
            }

            double Jr_inv[9];
            inv_right_jacobian(ee.phi, Jr_inv);

            double WJi[36], WJj[36];
            const bool has_i = (i > 0);
            const bool has_j = (j > 0);

            if (has_i) {
                double Ji[36] = {};
                // Ji[0:3, 0:3] = -Jr_inv @ R_hat^T
                double tmp[9];
                mat3_mul_T_right(Jr_inv, ee.R_hat, tmp);
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) Ji[r * 6 + c] = -tmp[r * 3 + c];
                }
                // Ji[3:6, 0:3] = Rm_T @ skew(t_hat)
                double sk[9] = {
                    0.0,          -ee.t_hat[2],  ee.t_hat[1],
                    ee.t_hat[2],   0.0,         -ee.t_hat[0],
                   -ee.t_hat[1],   ee.t_hat[0],  0.0
                };
                mat3_mul(Rm_T[e].m, sk, tmp);
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        Ji[(r + 3) * 6 + c] = tmp[r * 3 + c];
                        Ji[(r + 3) * 6 + (c + 3)] = -Rm_T[e].m[r * 3 + c];
                    }
                }
                for (int r = 0; r < 6; ++r) {
                    const double* wr = We + r * 6;
                    for (int c = 0; c < 6; ++c) {
                        WJi[r * 6 + c] = wr[0] * Ji[0 * 6 + c] + wr[1] * Ji[1 * 6 + c]
                                       + wr[2] * Ji[2 * 6 + c] + wr[3] * Ji[3 * 6 + c]
                                       + wr[4] * Ji[4 * 6 + c] + wr[5] * Ji[5 * 6 + c];
                    }
                }
                const int pu = sym.inv_perm[i - 1];
                double* Huu = H_diag[pu].m;
                double* gu = g_perm.data() + pu * 6;
                for (int a = 0; a < 6; ++a) {
                    double ga = 0.0;
                    for (int r = 0; r < 6; ++r) ga += WJi[r * 6 + a] * irls[r] * ee.res[r];
                    gu[a] += ga;
                    for (int b = 0; b <= a; ++b) {
                        double s = 0.0;
                        for (int r = 0; r < 6; ++r) s += WJi[r * 6 + a] * irls[r] * WJi[r * 6 + b];
                        Huu[a * 6 + b] += s;
                        if (a != b) Huu[b * 6 + a] += s;
                    }
                }
            }

            if (has_j) {
                double Jj[36] = {};
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        Jj[r * 6 + c] = Jr_inv[r * 3 + c];
                        Jj[(r + 3) * 6 + (c + 3)] = ee.dR[r * 3 + c];
                    }
                }
                for (int r = 0; r < 6; ++r) {
                    const double* wr = We + r * 6;
                    for (int c = 0; c < 6; ++c) {
                        WJj[r * 6 + c] = wr[0] * Jj[0 * 6 + c] + wr[1] * Jj[1 * 6 + c]
                                       + wr[2] * Jj[2 * 6 + c] + wr[3] * Jj[3 * 6 + c]
                                       + wr[4] * Jj[4 * 6 + c] + wr[5] * Jj[5 * 6 + c];
                    }
                }
                const int pv = sym.inv_perm[j - 1];
                double* Hvv = H_diag[pv].m;
                double* gv = g_perm.data() + pv * 6;
                for (int a = 0; a < 6; ++a) {
                    double ga = 0.0;
                    for (int r = 0; r < 6; ++r) ga += WJj[r * 6 + a] * irls[r] * ee.res[r];
                    gv[a] += ga;
                    for (int b = 0; b <= a; ++b) {
                        double s = 0.0;
                        for (int r = 0; r < 6; ++r) s += WJj[r * 6 + a] * irls[r] * WJj[r * 6 + b];
                        Hvv[a * 6 + b] += s;
                        if (a != b) Hvv[b * 6 + a] += s;
                    }
                }
            }

            if (has_i && has_j) {
                const int bidx = edge_offdiag_idx[e];
                double* Hrc = H_sub[bidx].m;
                if (!edge_swapped[e]) {
                    // pu < pv -> row = pv, col = pu -> H[pv, pu] += WJj^T * diag(irls) * WJi
                    for (int a = 0; a < 6; ++a) {
                        for (int b = 0; b < 6; ++b) {
                            double s = 0.0;
                            for (int r = 0; r < 6; ++r) s += WJj[r * 6 + a] * irls[r] * WJi[r * 6 + b];
                            Hrc[a * 6 + b] += s;
                        }
                    }
                } else {
                    // pv < pu -> row = pu, col = pv -> H[pu, pv] += WJi^T * diag(irls) * WJj
                    for (int a = 0; a < 6; ++a) {
                        for (int b = 0; b < 6; ++b) {
                            double s = 0.0;
                            for (int r = 0; r < 6; ++r) s += WJi[r * 6 + a] * irls[r] * WJj[r * 6 + b];
                            Hrc[a * 6 + b] += s;
                        }
                    }
                }
            }
        }

        optimality = 0.0;
        bool finite_g = true;
        for (int k = 0; k < n_free * 6; ++k) {
            double ak = std::fabs(g_perm[k]);
            if (!std::isfinite(ak)) { finite_g = false; break; }
            if (ak > optimality) optimality = ak;
        }
        if (!finite_g) break;
        if (optimality < 1e-6) {
            converged = true;
            break;
        }

        bool accepted = false;
        for (int lm_try = 0; lm_try < 8; ++lm_try) {
            if (!factor_and_solve(sym, H_diag, H_sub, g_perm, lam, L_diag, L_sub, step_orig)) {
                lam = std::min(lam * 10.0, 1e8);
                continue;
            }
            bool finite_step = true;
            double step_norm2 = 0.0;
            for (int k = 0; k < n_free * 6; ++k) {
                double sk = step_orig[k];
                if (!std::isfinite(sk)) { finite_step = false; break; }
                step_norm2 += sk * sk;
            }
            if (!finite_step) {
                lam = std::min(lam * 10.0, 1e8);
                continue;
            }

            // Apply SE(3) right-perturbation update.
            for (int k = 0; k < n_free; ++k) {
                const double* s = step_orig.data() + k * 6;
                double dR_s[9];
                so3_exp(s, dR_s);
                double dt_w[3];
                mat3_vec(R_cur[k + 1].m, s + 3, dt_w);
                t_cand[k + 1].v[0] = t_cur[k + 1].v[0] + dt_w[0];
                t_cand[k + 1].v[1] = t_cur[k + 1].v[1] + dt_w[1];
                t_cand[k + 1].v[2] = t_cur[k + 1].v[2] + dt_w[2];
                mat3_mul(R_cur[k + 1].m, dR_s, R_cand[k + 1].m);
            }

            double cost_c = eval_graph(n_edges, fi, si, Rm_T, tm, W, R_cand, t_cand, delta_h, ev_cand);
            nfev += 1;
            if (std::isfinite(cost_c) && cost_c < cur_cost) {
                double rel_impr = (cur_cost - cost_c) / std::max(cur_cost, 1e-12);
                R_cur.swap(R_cand);
                t_cur.swap(t_cand);
                R_cand[0] = R_cur[0];
                t_cand[0] = t_cur[0];
                ev.swap(ev_cand);
                cur_cost = cost_c;
                lam = std::max(lam * 0.3, 1e-7);
                accepted = true;
                if (rel_impr < 1e-8 || std::sqrt(step_norm2) < 1e-6) {
                    converged = true;
                }
                break;
            }
            lam = std::min(lam * 5.0, 1e8);
        }

        if (converged || !accepted) {
            if (!accepted) converged = true;
            break;
        }
    }

    // Write back updated poses.
    for (int k = 0; k < n_poses; ++k) {
        double* P = poses + k * 16;
        P[0] = R_cur[k].m[0];  P[1] = R_cur[k].m[1];  P[2] = R_cur[k].m[2];  P[3] = t_cur[k].v[0];
        P[4] = R_cur[k].m[3];  P[5] = R_cur[k].m[4];  P[6] = R_cur[k].m[5];  P[7] = t_cur[k].v[1];
        P[8] = R_cur[k].m[6];  P[9] = R_cur[k].m[7];  P[10] = R_cur[k].m[8]; P[11] = t_cur[k].v[2];
        P[12] = 0.0; P[13] = 0.0; P[14] = 0.0; P[15] = 1.0;
    }

    if (out_stats) {
        out_stats[0] = initial_cost;
        out_stats[1] = cur_cost;
        out_stats[2] = optimality;
        out_stats[3] = static_cast<double>(nfev);
        out_stats[4] = static_cast<double>(njev);
    }
    return converged ? 1 : 0;
}
