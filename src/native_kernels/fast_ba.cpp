#include "portable_export.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// Exact Rodrigues formula R = exp([w]_x) matching scipy Rotation.from_rotvec
inline void rotvec_to_mat(const double w[3], double R[9]) {
    const double wx = w[0], wy = w[1], wz = w[2];
    const double th2 = wx * wx + wy * wy + wz * wz;
    double A, B;
    if (th2 < 1e-14) {
        A = 1.0 - th2 / 6.0;
        B = 0.5 - th2 / 24.0;
    } else {
        const double th = std::sqrt(th2);
        A = std::sin(th) / th;
        B = (1.0 - std::cos(th)) / th2;
    }
    const double xx = wx * wx, yy = wy * wy, zz = wz * wz;
    const double xy = wx * wy, xz = wx * wz, yz = wy * wz;
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

// Exact SO(3) right Jacobian J_r(w) = I - c1*[w]_x + c2*[w]_x^2
inline void right_jacobian(const double w[3], double Jr[9]) {
    const double wx = w[0], wy = w[1], wz = w[2];
    const double th2 = wx * wx + wy * wy + wz * wz;
    double c1, c2;
    if (th2 < 1e-14) {
        c1 = 0.5;
        c2 = 1.0 / 6.0;
    } else {
        const double th = std::sqrt(th2);
        c1 = (1.0 - std::cos(th)) / th2;
        c2 = (th - std::sin(th)) / (th2 * th);
    }
    const double xx = wx * wx, yy = wy * wy, zz = wz * wz;
    const double xy = wx * wy, xz = wx * wz, yz = wy * wz;
    // [w]_x = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]
    // [w]_x^2 = [[-(yy+zz), xy, xz], [xy, -(xx+zz), yz], [xz, yz, -(xx+yy)]]
    Jr[0] = 1.0 - c2 * (yy + zz);
    Jr[1] = c1 * wz + c2 * xy;
    Jr[2] = -c1 * wy + c2 * xz;
    Jr[3] = -c1 * wz + c2 * xy;
    Jr[4] = 1.0 - c2 * (xx + zz);
    Jr[5] = c1 * wx + c2 * yz;
    Jr[6] = c1 * wy + c2 * xz;
    Jr[7] = -c1 * wx + c2 * yz;
    Jr[8] = 1.0 - c2 * (xx + yy);
}

inline bool invert_3x3(const double M[9], double inv[9]) {
    const double c00 = M[4] * M[8] - M[5] * M[7];
    const double c01 = M[5] * M[6] - M[3] * M[8];
    const double c02 = M[3] * M[7] - M[4] * M[6];
    const double det = M[0] * c00 + M[1] * c01 + M[2] * c02;
    if (!std::isfinite(det) || std::abs(det) < 1e-18) return false;
    const double idet = 1.0 / det;
    inv[0] = c00 * idet;
    inv[1] = (M[2] * M[7] - M[1] * M[8]) * idet;
    inv[2] = (M[1] * M[5] - M[2] * M[4]) * idet;
    inv[3] = c01 * idet;
    inv[4] = (M[0] * M[8] - M[2] * M[6]) * idet;
    inv[5] = (M[2] * M[3] - M[0] * M[5]) * idet;
    inv[6] = c02 * idet;
    inv[7] = (M[1] * M[6] - M[0] * M[7]) * idet;
    inv[8] = (M[0] * M[4] - M[1] * M[3]) * idet;
    return true;
}

// Dense Gaussian elimination with partial pivoting for small N x N system (N <= 60).
inline bool solve_dense(int32_t n, std::vector<double>& A, std::vector<double>& b, std::vector<double>& x) {
    x.resize(n);
    for (int32_t k = 0; k < n; ++k) {
        int32_t piv = k;
        double max_v = std::abs(A[k * n + k]);
        for (int32_t r = k + 1; r < n; ++r) {
            double v = std::abs(A[r * n + k]);
            if (v > max_v) { max_v = v; piv = r; }
        }
        if (!std::isfinite(max_v) || max_v < 1e-15) return false;
        if (piv != k) {
            for (int32_t c = k; c < n; ++c) std::swap(A[k * n + c], A[piv * n + c]);
            std::swap(b[k], b[piv]);
        }
        const double inv_diag = 1.0 / A[k * n + k];
        for (int32_t r = k + 1; r < n; ++r) {
            const double f = A[r * n + k] * inv_diag;
            if (f == 0.0) continue;
            for (int32_t c = k + 1; c < n; ++c) A[r * n + c] -= f * A[k * n + c];
            b[r] -= f * b[k];
        }
    }
    for (int32_t r = n - 1; r >= 0; --r) {
        double sum = b[r];
        for (int32_t c = r + 1; c < n; ++c) sum -= A[r * n + c] * x[c];
        double val = sum / A[r * n + r];
        if (!std::isfinite(val)) return false;
        x[r] = val;
    }
    return true;
}

inline void eval_obs_block(
    int32_t m,
    const double* pts_w,       // either indexed via pt_idx or direct [m, 3]
    const int32_t* pt_idx,     // nullptr if direct
    const double* cams_R,      // [n_cams, 9]
    const double* cams_t,      // [n_cams, 3]
    const int32_t* cam_idx,    // [m]
    const double* pixels,      // [m, 2]
    const double* right_u,     // [m]
    const uint8_t* mask,       // [m, 3]
    const double* weights,     // [m, 3] or nullptr (all 1.0)
    const double K[9],
    double fb,
    double doff,
    int32_t metric,
    double delta_h,
    double* out_pc,            // [m, 3]
    double* out_raw_left,      // [m, 2]
    double* out_wres,          // [m, 3]
    double& out_cost
) {
    double cost = 0.0;
    for (int32_t i = 0; i < m; ++i) {
        const double* pw = pt_idx ? (pts_w + static_cast<size_t>(pt_idx[i]) * 3)
                                  : (pts_w + static_cast<size_t>(i) * 3);
        const int32_t c = cam_idx[i];
        const double* R = cams_R + static_cast<size_t>(c) * 9;
        const double* t = cams_t + static_cast<size_t>(c) * 3;
        const double dx = pw[0] - t[0];
        const double dy = pw[1] - t[1];
        const double dz = pw[2] - t[2];
        // pc = R^T @ (pw - t)  (in Python: delta @ R)
        const double px = dx * R[0] + dy * R[3] + dz * R[6];
        const double py = dx * R[1] + dy * R[4] + dz * R[7];
        const double pz = dx * R[2] + dy * R[5] + dz * R[8];
        out_pc[3 * i + 0] = px;
        out_pc[3 * i + 1] = py;
        out_pc[3 * i + 2] = pz;

        const double h0 = K[0] * px + K[1] * py + K[2] * pz;
        const double h1 = K[3] * px + K[4] * py + K[5] * pz;
        const double h2 = K[6] * px + K[7] * py + K[8] * pz;
        const double inv_h2 = 1.0 / std::max(h2, 1e-9);
        const double u = h0 * inv_h2;
        const double v = h1 * inv_h2;
        const double raw_u = u - pixels[2 * i + 0];
        const double raw_v = v - pixels[2 * i + 1];
        out_raw_left[2 * i + 0] = raw_u;
        out_raw_left[2 * i + 1] = raw_v;

        double r0, r1, r2 = 0.0;
        if (pz <= 0.0) {
            r0 = 1e4;
            r1 = 1e4;
            r2 = metric ? 1e4 : 0.0;
        } else {
            r0 = std::max(-1e4, std::min(1e4, raw_u));
            r1 = std::max(-1e4, std::min(1e4, raw_v));
            if (metric) {
                r2 = u - fb / std::max(pz, 1e-9) - doff - right_u[i];
            }
        }
        const double res3[3] = {r0, r1, r2};
        const uint8_t* mi = mask + 3 * i;
        const double* wi = weights ? (weights + 3 * i) : nullptr;
        for (int32_t d = 0; d < 3; ++d) {
            if (mi[d]) {
                const double wr = wi ? (res3[d] * wi[d]) : res3[d];
                out_wres[3 * i + d] = wr;
                const double a = std::abs(wr);
                cost += (a <= delta_h) ? (0.5 * wr * wr) : (delta_h * (a - 0.5 * delta_h));
            } else {
                out_wres[3 * i + d] = 0.0;
            }
        }
    }
    out_cost = cost;
}

inline void proj_jac_cam(
    const double pc[3],
    const double K[9],
    double fb,
    const double raw_left[2],
    double J_pix[9]  // [3, 3]
) {
    std::memset(J_pix, 0, 9 * sizeof(double));
    const double pz = pc[2];
    if (pz <= 0.0) return;

    const double h0 = K[0] * pc[0] + K[1] * pc[1] + K[2] * pc[2];
    const double h1 = K[3] * pc[0] + K[4] * pc[1] + K[5] * pc[2];
    const double h2 = K[6] * pc[0] + K[7] * pc[1] + K[8] * pc[2];

    double lp0[3], lp1[3];
    if (h2 > 1e-9) {
        const double inv_h2_sq = 1.0 / (h2 * h2);
        for (int32_t j = 0; j < 3; ++j) {
            lp0[j] = (h2 * K[0 + j] - h0 * K[6 + j]) * inv_h2_sq;
            lp1[j] = (h2 * K[3 + j] - h1 * K[6 + j]) * inv_h2_sq;
        }
    } else {
        for (int32_t j = 0; j < 3; ++j) {
            lp0[j] = K[0 + j] / 1e-9;
            lp1[j] = K[3 + j] / 1e-9;
        }
    }

    if (std::abs(raw_left[0]) < 1e4) {
        J_pix[0] = lp0[0]; J_pix[1] = lp0[1]; J_pix[2] = lp0[2];
    }
    if (std::abs(raw_left[1]) < 1e4) {
        J_pix[3] = lp1[0]; J_pix[4] = lp1[1]; J_pix[5] = lp1[2];
    }
    J_pix[6] = lp0[0]; J_pix[7] = lp0[1]; J_pix[8] = lp0[2];
    if (pz > 1e-9 && fb != 0.0) {
        J_pix[8] += fb / (pz * pz);
    }
}

}  // namespace

extern "C" {

// Exact C++ Schur-complement Huber-IRLS Levenberg-Marquardt local BA solver.
// Modifies `x_free_poses` [n_free * 6] and `x_points` [n_pt * 3] in place.
VSLAM_EXPORT int32_t solve_schur_local_bundle_cpp(
    int32_t n_cams,
    int32_t n_free,
    int32_t n_pt,
    int32_t n_rec,
    int32_t n_held,
    const double* base_cams_R,     // [n_cams * 9]
    const double* base_cams_t,     // [n_cams * 3]
    const int32_t* free_cam_idx,   // [n_free]
    double* x_free_poses,          // [n_free * 6] (rotvec(3), trans(3)) in/out
    double* x_points,              // [n_pt * 3] in/out
    const int32_t* rec_cams,       // [n_rec]
    const int32_t* rec_pts,        // [n_rec]
    const double* rec_pixels,      // [n_rec * 2]
    const double* rec_right,       // [n_rec]
    const uint8_t* rec_mask,       // [n_rec * 3]
    const double* rec_weights,     // [n_rec * 3]
    const int32_t* held_cams,      // [n_held]
    const double* held_pts,        // [n_held * 3]
    const double* held_pixels,     // [n_held * 2]
    const double* held_right,      // [n_held]
    const uint8_t* held_mask,      // [n_held * 3]
    const double* K,               // [9]
    double baseline,
    double disparity_offset,
    int32_t metric,
    double f_scale,
    int32_t max_iter,
    int32_t* out_status,
    int32_t* out_nfev,
    double* out_cost,
    double* out_optimality
) {
    if (n_free <= 0 || n_pt <= 0) {
        *out_status = 1;
        *out_nfev = 1;
        *out_cost = 0.0;
        *out_optimality = 0.0;
        return 1;
    }

    const double fb = metric ? (K[0] * baseline) : 0.0;
    const double delta_h = std::max(f_scale, 1e-6);

    std::vector<int32_t> cam_to_slot(n_cams, -1);
    for (int32_t s = 0; s < n_free; ++s) cam_to_slot[free_cam_idx[s]] = s;

    // Build per-point sparse list of free-camera slots that observe point p (deduplicated)
    // and flat W storage indexed by (slot * n_pt + p).
    std::vector<double> cams_R(base_cams_R, base_cams_R + static_cast<size_t>(n_cams) * 9);
    std::vector<double> cams_t(base_cams_t, base_cams_t + static_cast<size_t>(n_cams) * 3);
    std::vector<double> cams_R_cand = cams_R;
    std::vector<double> cams_t_cand = cams_t;

    auto update_cams = [&](const double* fp, std::vector<double>& cR, std::vector<double>& ct) {
        for (int32_t s = 0; s < n_free; ++s) {
            const int32_t ci = free_cam_idx[s];
            rotvec_to_mat(fp + 6 * s, cR.data() + static_cast<size_t>(ci) * 9);
            ct[3 * ci + 0] = fp[6 * s + 3];
            ct[3 * ci + 1] = fp[6 * s + 4];
            ct[3 * ci + 2] = fp[6 * s + 5];
        }
    };

    std::vector<double> pc_r(n_rec * 3), raw_r(n_rec * 2), wr_r(n_rec * 3);
    std::vector<double> pc_h(n_held * 3), raw_h(n_held * 2), wr_h(n_held * 3);
    std::vector<double> pc_rc(n_rec * 3), raw_rc(n_rec * 2), wr_rc(n_rec * 3);
    std::vector<double> pc_hc(n_held * 3), raw_hc(n_held * 2), wr_hc(n_held * 3);

    update_cams(x_free_poses, cams_R, cams_t);
    double c_r = 0.0, c_h = 0.0;
    eval_obs_block(n_rec, x_points, rec_pts, cams_R.data(), cams_t.data(), rec_cams,
                   rec_pixels, rec_right, rec_mask, rec_weights, K, fb, disparity_offset,
                   metric, delta_h, pc_r.data(), raw_r.data(), wr_r.data(), c_r);
    if (n_held > 0) {
        eval_obs_block(n_held, held_pts, nullptr, cams_R.data(), cams_t.data(), held_cams,
                       held_pixels, held_right, held_mask, nullptr, K, fb, disparity_offset,
                       metric, delta_h, pc_h.data(), raw_h.data(), wr_h.data(), c_h);
    }
    double cur_cost = c_r + c_h;
    int32_t nfev = 1;
    double lam = 1e-3;
    double optimality = 1e300;
    int32_t status = 0;
    bool converged = false;

    const int32_t n6 = 6 * n_free;
    std::vector<double> Jr_free(n_free * 9);
    std::vector<double> U(n_free * 36);
    std::vector<double> V(n_pt * 9);
    std::vector<double> W(static_cast<size_t>(n_free) * n_pt * 18);
    std::vector<uint8_t> W_nz(static_cast<size_t>(n_free) * n_pt);
    std::vector<double> g_c(n6);
    std::vector<double> g_p(n_pt * 3);
    std::vector<double> U_diag(n6);
    std::vector<double> V_diag(n_pt * 3);
    std::vector<double> V_inv(n_pt * 9);
    std::vector<double> S(n6 * n6);
    std::vector<double> rhs_c(n6);
    std::vector<double> dx_c(n6);
    std::vector<double> dx_p(n_pt * 3);
    std::vector<double> x_fp_cand(n6);
    std::vector<double> x_pt_cand(n_pt * 3);
    std::vector<int32_t> pt_slots_buf;
    pt_slots_buf.reserve(n_free);

    // Precompute which free slots touch each point (static across iterations!)
    std::vector<int32_t> pt_slot_head(n_pt + 1, 0);
    std::vector<int32_t> pt_slot_list;
    {
        std::vector<uint8_t> seen(static_cast<size_t>(n_pt) * n_free, 0);
        for (int32_t i = 0; i < n_rec; ++i) {
            int32_t s = cam_to_slot[rec_cams[i]];
            if (s >= 0) seen[static_cast<size_t>(rec_pts[i]) * n_free + s] = 1;
        }
        pt_slot_list.reserve(n_rec);
        for (int32_t p = 0; p < n_pt; ++p) {
            pt_slot_head[p] = static_cast<int32_t>(pt_slot_list.size());
            for (int32_t s = 0; s < n_free; ++s) {
                if (seen[static_cast<size_t>(p) * n_free + s]) pt_slot_list.push_back(s);
            }
        }
        pt_slot_head[n_pt] = static_cast<int32_t>(pt_slot_list.size());
    }

    const int32_t iters = std::max(1, max_iter);
    for (int32_t it = 0; it < iters; ++it) {
        for (int32_t s = 0; s < n_free; ++s) {
            right_jacobian(x_free_poses + 6 * s, Jr_free.data() + 9 * s);
        }
        std::fill(U.begin(), U.end(), 0.0);
        std::fill(V.begin(), V.end(), 0.0);
        std::fill(W.begin(), W.end(), 0.0);
        std::fill(g_c.begin(), g_c.end(), 0.0);
        std::fill(g_p.begin(), g_p.end(), 0.0);

        // 1. Accumulate optimized observations
        for (int32_t i = 0; i < n_rec; ++i) {
            const int32_t c = rec_cams[i];
            const int32_t p = rec_pts[i];
            const int32_t s = cam_to_slot[c];
            const double* pc = pc_r.data() + 3 * i;
            if (pc[2] <= 0.0) continue;

            double J_pix[9];
            proj_jac_cam(pc, K, fb, raw_r.data() + 2 * i, J_pix);

            const uint8_t* mi = rec_mask + 3 * i;
            const double* wi = rec_weights + 3 * i;
            const double* wri = wr_r.data() + 3 * i;
            double irls[3];
            for (int32_t d = 0; d < 3; ++d) {
                if (mi[d]) {
                    const double ww = wi[d];
                    J_pix[3 * d + 0] *= ww;
                    J_pix[3 * d + 1] *= ww;
                    J_pix[3 * d + 2] *= ww;
                    const double a = std::abs(wri[d]);
                    irls[d] = (a <= delta_h) ? 1.0 : (delta_h / std::max(a, 1e-12));
                } else {
                    J_pix[3 * d + 0] = 0.0;
                    J_pix[3 * d + 1] = 0.0;
                    J_pix[3 * d + 2] = 0.0;
                    irls[d] = 0.0;
                }
            }

            // J_pt[d, k] = sum_j R[k, j] * J_pix[d, j]  (R_wc @ J_pix^T transposed)
            const double* R = cams_R.data() + static_cast<size_t>(c) * 9;
            double J_pt[9];
            for (int32_t d = 0; d < 3; ++d) {
                const double j0 = J_pix[3 * d + 0], j1 = J_pix[3 * d + 1], j2 = J_pix[3 * d + 2];
                J_pt[3 * d + 0] = R[0] * j0 + R[1] * j1 + R[2] * j2;
                J_pt[3 * d + 1] = R[3] * j0 + R[4] * j1 + R[5] * j2;
                J_pt[3 * d + 2] = R[6] * j0 + R[7] * j1 + R[8] * j2;
            }

            double* Vp = V.data() + 9 * p;
            double* gpp = g_p.data() + 3 * p;
            for (int32_t d = 0; d < 3; ++d) {
                const double iw = irls[d];
                if (iw == 0.0) continue;
                const double a0 = J_pt[3 * d + 0], a1 = J_pt[3 * d + 1], a2 = J_pt[3 * d + 2];
                const double w0 = a0 * iw, w1 = a1 * iw, w2 = a2 * iw;
                Vp[0] += a0 * w0; Vp[1] += a0 * w1; Vp[2] += a0 * w2;
                Vp[3] += a1 * w0; Vp[4] += a1 * w1; Vp[5] += a1 * w2;
                Vp[6] += a2 * w0; Vp[7] += a2 * w1; Vp[8] += a2 * w2;
                const double rval = wri[d];
                gpp[0] += w0 * rval;
                gpp[1] += w1 * rval;
                gpp[2] += w2 * rval;
            }

            if (s >= 0) {
                const double* Jr = Jr_free.data() + 9 * s;
                double J_pose[18];  // [3, 6]
                for (int32_t d = 0; d < 3; ++d) {
                    const double j0 = J_pix[3 * d + 0], j1 = J_pix[3 * d + 1], j2 = J_pix[3 * d + 2];
                    // cross(J_pix[d], pc)
                    const double vc0 = j1 * pc[2] - j2 * pc[1];
                    const double vc1 = j2 * pc[0] - j0 * pc[2];
                    const double vc2 = j0 * pc[1] - j1 * pc[0];
                    // J_rot[d, :] = vc @ Jr
                    J_pose[6 * d + 0] = vc0 * Jr[0] + vc1 * Jr[3] + vc2 * Jr[6];
                    J_pose[6 * d + 1] = vc0 * Jr[1] + vc1 * Jr[4] + vc2 * Jr[7];
                    J_pose[6 * d + 2] = vc0 * Jr[2] + vc1 * Jr[5] + vc2 * Jr[8];
                    // J_tr[d, :] = -J_pt[d, :]
                    J_pose[6 * d + 3] = -J_pt[3 * d + 0];
                    J_pose[6 * d + 4] = -J_pt[3 * d + 1];
                    J_pose[6 * d + 5] = -J_pt[3 * d + 2];
                }

                double* Us = U.data() + 36 * s;
                double* gcs = g_c.data() + 6 * s;
                double* Wsp = W.data() + (static_cast<size_t>(s) * n_pt + p) * 18;
                for (int32_t d = 0; d < 3; ++d) {
                    const double iw = irls[d];
                    if (iw == 0.0) continue;
                    const double* jp = J_pose + 6 * d;
                    const double* jpt = J_pt + 3 * d;
                    const double rval = wri[d];
                    for (int32_t a = 0; a < 6; ++a) {
                        const double wja = jp[a] * iw;
                        gcs[a] += wja * rval;
                        for (int32_t b = 0; b < 6; ++b) Us[6 * a + b] += jp[a] * (jp[b] * iw);
                        Wsp[3 * a + 0] += wja * jpt[0];
                        Wsp[3 * a + 1] += wja * jpt[1];
                        Wsp[3 * a + 2] += wja * jpt[2];
                    }
                }
            }
        }

        // 2. Held-out fixed landmarks block
        for (int32_t i = 0; i < n_held; ++i) {
            const int32_t c = held_cams[i];
            const int32_t s = cam_to_slot[c];
            if (s < 0) continue;
            const double* pc = pc_h.data() + 3 * i;
            if (pc[2] <= 0.0) continue;

            double J_pix[9];
            proj_jac_cam(pc, K, fb, raw_h.data() + 2 * i, J_pix);
            const uint8_t* mi = held_mask + 3 * i;
            const double* wri = wr_h.data() + 3 * i;
            double irls[3];
            for (int32_t d = 0; d < 3; ++d) {
                if (mi[d]) {
                    const double a = std::abs(wri[d]);
                    irls[d] = (a <= delta_h) ? 1.0 : (delta_h / std::max(a, 1e-12));
                } else {
                    J_pix[3 * d + 0] = J_pix[3 * d + 1] = J_pix[3 * d + 2] = 0.0;
                    irls[d] = 0.0;
                }
            }
            const double* R = cams_R.data() + static_cast<size_t>(c) * 9;
            const double* Jr = Jr_free.data() + 9 * s;
            double J_pose[18];
            for (int32_t d = 0; d < 3; ++d) {
                const double j0 = J_pix[3 * d + 0], j1 = J_pix[3 * d + 1], j2 = J_pix[3 * d + 2];
                const double vc0 = j1 * pc[2] - j2 * pc[1];
                const double vc1 = j2 * pc[0] - j0 * pc[2];
                const double vc2 = j0 * pc[1] - j1 * pc[0];
                J_pose[6 * d + 0] = vc0 * Jr[0] + vc1 * Jr[3] + vc2 * Jr[6];
                J_pose[6 * d + 1] = vc0 * Jr[1] + vc1 * Jr[4] + vc2 * Jr[7];
                J_pose[6 * d + 2] = vc0 * Jr[2] + vc1 * Jr[5] + vc2 * Jr[8];
                J_pose[6 * d + 3] = -(R[0] * j0 + R[1] * j1 + R[2] * j2);
                J_pose[6 * d + 4] = -(R[3] * j0 + R[4] * j1 + R[5] * j2);
                J_pose[6 * d + 5] = -(R[6] * j0 + R[7] * j1 + R[8] * j2);
            }
            double* Us = U.data() + 36 * s;
            double* gcs = g_c.data() + 6 * s;
            for (int32_t d = 0; d < 3; ++d) {
                const double iw = irls[d];
                if (iw == 0.0) continue;
                const double* jp = J_pose + 6 * d;
                const double rval = wri[d];
                for (int32_t a = 0; a < 6; ++a) {
                    const double wja = jp[a] * iw;
                    gcs[a] += wja * rval;
                    for (int32_t b = 0; b < 6; ++b) Us[6 * a + b] += jp[a] * (jp[b] * iw);
                }
            }
        }

        double max_g = 0.0;
        bool finite_g = true;
        for (int32_t i = 0; i < n6; ++i) {
            if (!std::isfinite(g_c[i])) { finite_g = false; break; }
            max_g = std::max(max_g, std::abs(g_c[i]));
        }
        for (int32_t i = 0; i < n_pt * 3 && finite_g; ++i) {
            if (!std::isfinite(g_p[i])) { finite_g = false; break; }
            max_g = std::max(max_g, std::abs(g_p[i]));
        }
        optimality = finite_g ? max_g : NAN;
        if (!finite_g) {
            status = -1;
            break;
        }
        if (optimality < 1e-7) {
            converged = true;
            status = 1;
            break;
        }

        for (int32_t s = 0; s < n_free; ++s) {
            for (int32_t a = 0; a < 6; ++a) U_diag[6 * s + a] = std::max(U[36 * s + 6 * a + a], 1e-6);
        }
        for (int32_t p = 0; p < n_pt; ++p) {
            V_diag[3 * p + 0] = std::max(V[9 * p + 0], 1e-6);
            V_diag[3 * p + 1] = std::max(V[9 * p + 4], 1e-6);
            V_diag[3 * p + 2] = std::max(V[9 * p + 8], 1e-6);
        }

        bool accepted = false;
        for (int32_t lm_try = 0; lm_try < 8; ++lm_try) {
            bool inv_ok = true;
            for (int32_t p = 0; p < n_pt; ++p) {
                double Vd[9];
                std::memcpy(Vd, V.data() + 9 * p, 9 * sizeof(double));
                Vd[0] += lam * V_diag[3 * p + 0];
                Vd[4] += lam * V_diag[3 * p + 1];
                Vd[8] += lam * V_diag[3 * p + 2];
                if (!invert_3x3(Vd, V_inv.data() + 9 * p)) {
                    inv_ok = false;
                    break;
                }
            }
            if (!inv_ok) {
                lam = std::min(lam * 10.0, 1e8);
                continue;
            }

            // Build Schur complement S = U_damp - sum_p W_p V_inv_p W_p^T and rhs_c = -g_c + sum_p W_p V_inv_p g_p
            std::fill(S.begin(), S.end(), 0.0);
            for (int32_t s = 0; s < n_free; ++s) {
                const double* Us = U.data() + 36 * s;
                const int32_t s0 = 6 * s;
                for (int32_t a = 0; a < 6; ++a) {
                    for (int32_t b = 0; b < 6; ++b) {
                        S[(s0 + a) * n6 + (s0 + b)] = Us[6 * a + b];
                    }
                    S[(s0 + a) * n6 + (s0 + a)] += lam * U_diag[s0 + a];
                    rhs_c[s0 + a] = -g_c[s0 + a];
                }
            }

            for (int32_t p = 0; p < n_pt; ++p) {
                const int32_t beg = pt_slot_head[p];
                const int32_t end = pt_slot_head[p + 1];
                if (beg == end) continue;
                const double* Vi = V_inv.data() + 9 * p;
                const double* gpp = g_p.data() + 3 * p;
                const double vig0 = Vi[0] * gpp[0] + Vi[1] * gpp[1] + Vi[2] * gpp[2];
                const double vig1 = Vi[3] * gpp[0] + Vi[4] * gpp[1] + Vi[5] * gpp[2];
                const double vig2 = Vi[6] * gpp[0] + Vi[7] * gpp[1] + Vi[8] * gpp[2];

                for (int32_t ii = beg; ii < end; ++ii) {
                    const int32_t s1 = pt_slot_list[ii];
                    const double* W1 = W.data() + (static_cast<size_t>(s1) * n_pt + p) * 18;
                    double WV[18];  // [6, 3] = W1 @ Vi
                    const int32_t r0 = 6 * s1;
                    for (int32_t a = 0; a < 6; ++a) {
                        const double w0 = W1[3 * a + 0], w1 = W1[3 * a + 1], w2 = W1[3 * a + 2];
                        WV[3 * a + 0] = w0 * Vi[0] + w1 * Vi[3] + w2 * Vi[6];
                        WV[3 * a + 1] = w0 * Vi[1] + w1 * Vi[4] + w2 * Vi[7];
                        WV[3 * a + 2] = w0 * Vi[2] + w1 * Vi[5] + w2 * Vi[8];
                        rhs_c[r0 + a] += w0 * vig0 + w1 * vig1 + w2 * vig2;
                    }
                    for (int32_t jj = beg; jj < end; ++jj) {
                        const int32_t s2 = pt_slot_list[jj];
                        const double* W2 = W.data() + (static_cast<size_t>(s2) * n_pt + p) * 18;
                        const int32_t c0 = 6 * s2;
                        for (int32_t a = 0; a < 6; ++a) {
                            const double q0 = WV[3 * a + 0], q1 = WV[3 * a + 1], q2 = WV[3 * a + 2];
                            double* Srow = S.data() + (r0 + a) * n6 + c0;
                            for (int32_t b = 0; b < 6; ++b) {
                                Srow[b] -= q0 * W2[3 * b + 0] + q1 * W2[3 * b + 1] + q2 * W2[3 * b + 2];
                            }
                        }
                    }
                }
            }

            if (!solve_dense(n6, S, rhs_c, dx_c)) {
                lam = std::min(lam * 10.0, 1e8);
                continue;
            }

            // Back-substitute points: dx_p[p] = V_inv[p] @ (-g_p[p] - sum_s W[s,p]^T dx_c[s])
            bool step_ok = true;
            double step_sq = 0.0;
            for (int32_t i = 0; i < n6; ++i) step_sq += dx_c[i] * dx_c[i];
            for (int32_t p = 0; p < n_pt; ++p) {
                double b0 = -g_p[3 * p + 0], b1 = -g_p[3 * p + 1], b2 = -g_p[3 * p + 2];
                for (int32_t ii = pt_slot_head[p]; ii < pt_slot_head[p + 1]; ++ii) {
                    const int32_t s = pt_slot_list[ii];
                    const double* Wsp = W.data() + (static_cast<size_t>(s) * n_pt + p) * 18;
                    const double* dxc_s = dx_c.data() + 6 * s;
                    for (int32_t a = 0; a < 6; ++a) {
                        const double da = dxc_s[a];
                        b0 -= Wsp[3 * a + 0] * da;
                        b1 -= Wsp[3 * a + 1] * da;
                        b2 -= Wsp[3 * a + 2] * da;
                    }
                }
                const double* Vi = V_inv.data() + 9 * p;
                const double d0 = Vi[0] * b0 + Vi[1] * b1 + Vi[2] * b2;
                const double d1 = Vi[3] * b0 + Vi[4] * b1 + Vi[5] * b2;
                const double d2 = Vi[6] * b0 + Vi[7] * b1 + Vi[8] * b2;
                if (!std::isfinite(d0) || !std::isfinite(d1) || !std::isfinite(d2)) {
                    step_ok = false;
                    break;
                }
                dx_p[3 * p + 0] = d0;
                dx_p[3 * p + 1] = d1;
                dx_p[3 * p + 2] = d2;
                step_sq += d0 * d0 + d1 * d1 + d2 * d2;
            }
            if (!step_ok) {
                lam = std::min(lam * 10.0, 1e8);
                continue;
            }

            for (int32_t i = 0; i < n6; ++i) x_fp_cand[i] = x_free_poses[i] + dx_c[i];
            for (int32_t i = 0; i < n_pt * 3; ++i) x_pt_cand[i] = x_points[i] + dx_p[i];

            update_cams(x_fp_cand.data(), cams_R_cand, cams_t_cand);
            double c_rc = 0.0, c_hc = 0.0;
            eval_obs_block(n_rec, x_pt_cand.data(), rec_pts, cams_R_cand.data(), cams_t_cand.data(), rec_cams,
                           rec_pixels, rec_right, rec_mask, rec_weights, K, fb, disparity_offset,
                           metric, delta_h, pc_rc.data(), raw_rc.data(), wr_rc.data(), c_rc);
            if (n_held > 0) {
                eval_obs_block(n_held, held_pts, nullptr, cams_R_cand.data(), cams_t_cand.data(), held_cams,
                               held_pixels, held_right, held_mask, nullptr, K, fb, disparity_offset,
                               metric, delta_h, pc_hc.data(), raw_hc.data(), wr_hc.data(), c_hc);
            }
            const double cost_c = c_rc + c_hc;
            ++nfev;

            if (std::isfinite(cost_c) && cost_c < cur_cost) {
                const double rel_impr = (cur_cost - cost_c) / std::max(cur_cost, 1e-12);
                const double step_norm = std::sqrt(step_sq);
                std::memcpy(x_free_poses, x_fp_cand.data(), n6 * sizeof(double));
                std::memcpy(x_points, x_pt_cand.data(), n_pt * 3 * sizeof(double));
                std::swap(cams_R, cams_R_cand);
                std::swap(cams_t, cams_t_cand);
                std::swap(pc_r, pc_rc);
                std::swap(raw_r, raw_rc);
                std::swap(wr_r, wr_rc);
                std::swap(pc_h, pc_hc);
                std::swap(raw_h, raw_hc);
                std::swap(wr_h, wr_hc);
                cur_cost = cost_c;
                lam = std::max(lam * 0.3, 1e-7);
                accepted = true;
                if (rel_impr < 1e-8 || step_norm < 1e-7) {
                    converged = true;
                    status = 2;
                }
                break;
            }
            lam = std::min(lam * 5.0, 1e8);
        }

        if (converged || !accepted) {
            if (!accepted) {
                converged = true;
                status = 3;
            }
            break;
        }
    }

    *out_status = status;
    *out_nfev = nfev;
    *out_cost = cur_cost;
    *out_optimality = optimality;
    return 1;
}

static inline bool solve_6x6(const double H_in[36], const double g_in[6], double step[6]) {
    double A[6][7];
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) A[r][c] = H_in[r * 6 + c];
        A[r][6] = -g_in[r];
    }
    for (int k = 0; k < 6; ++k) {
        int piv = k;
        double max_v = std::abs(A[k][k]);
        for (int r = k + 1; r < 6; ++r) {
            double v = std::abs(A[r][k]);
            if (v > max_v) { max_v = v; piv = r; }
        }
        if (!std::isfinite(max_v) || max_v < 1e-18) return false;
        if (piv != k) {
            for (int c = k; c < 7; ++c) std::swap(A[k][c], A[piv][c]);
        }
        const double inv_p = 1.0 / A[k][k];
        for (int r = k + 1; r < 6; ++r) {
            const double f = A[r][k] * inv_p;
            for (int c = k + 1; c < 7; ++c) A[r][c] -= f * A[k][c];
        }
    }
    for (int r = 5; r >= 0; --r) {
        double s = A[r][6];
        for (int c = r + 1; c < 6; ++c) s -= A[r][c] * step[c];
        step[r] = s / A[r][r];
        if (!std::isfinite(step[r])) return false;
    }
    return true;
}

static inline void mat3_mul(const double A[9], const double B[9], double C[9]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            C[r * 3 + c] = A[r * 3 + 0] * B[0 * 3 + c]
                         + A[r * 3 + 1] * B[1 * 3 + c]
                         + A[r * 3 + 2] * B[2 * 3 + c];
        }
    }
}

VSLAM_EXPORT int32_t refine_stereo_pose_lm_f64(
    double* R_cw,                  // [9] in/out
    double* t_cw,                  // [3] in/out
    const double* pts_w,           // [n * 3]
    const double* obs,             // [n * 3] (u, v, uR)
    const double* weights,         // [n * 3]
    const uint8_t* mask,           // [n * 3]
    int32_t n,
    double fx, double fy, double cx, double cy,
    double fb, double doff,
    double delta_huber,
    int32_t max_iter,
    double* out_initial_cost,
    double* out_final_cost,
    int32_t* out_converged,
    double* out_H_final            // [36]
) {
    if (n <= 0) {
        *out_initial_cost = 0.0;
        *out_final_cost = 0.0;
        *out_converged = 0;
        std::memset(out_H_final, 0, 36 * sizeof(double));
        return 1;
    }

    auto eval_cost = [&](const double R[9], const double t[3]) -> double {
        double cost = 0.0;
        for (int32_t i = 0; i < n; ++i) {
            const double* p = pts_w + i * 3;
            const double x = R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0];
            const double y = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1];
            const double z = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2];
            const bool valid = z > 1e-3;
            const double inv_z = valid ? (1.0 / z) : 0.0;
            const double u = fx * x * inv_z + cx;
            const double v = fy * y * inv_z + cy;
            const double uR = u - fb * inv_z - doff;
            const double* o = obs + i * 3;
            const double* w = weights + i * 3;
            const uint8_t* m = mask + i * 3;
            const double raw[3] = {
                valid ? std::clamp(u - o[0], -1e4, 1e4) : 1e4,
                valid ? std::clamp(v - o[1], -1e4, 1e4) : 1e4,
                valid ? std::clamp(uR - o[2], -1e4, 1e4) : 1e4
            };
            for (int c = 0; c < 3; ++c) {
                if (!m[c]) continue;
                const double wr = raw[c] * w[c];
                const double a = std::abs(wr);
                cost += (a <= delta_huber) ? (0.5 * wr * wr) : (delta_huber * (a - 0.5 * delta_huber));
            }
        }
        return cost;
    };

    double cur_cost = eval_cost(R_cw, t_cw);
    *out_initial_cost = cur_cost;
    double lam = 1e-3;
    int32_t converged = 0;
    double H_final[36];
    std::memset(H_final, 0, sizeof(H_final));

    for (int32_t it = 0; it < max_iter; ++it) {
        double H[36];
        double g[6];
        std::memset(H, 0, sizeof(H));
        std::memset(g, 0, sizeof(g));

        for (int32_t i = 0; i < n; ++i) {
            const double* p = pts_w + i * 3;
            const double x = R_cw[0] * p[0] + R_cw[1] * p[1] + R_cw[2] * p[2] + t_cw[0];
            const double y = R_cw[3] * p[0] + R_cw[4] * p[1] + R_cw[5] * p[2] + t_cw[1];
            const double z = R_cw[6] * p[0] + R_cw[7] * p[1] + R_cw[8] * p[2] + t_cw[2];
            if (z <= 1e-3) continue;
            const double iz = 1.0 / z;
            const double u = fx * x * iz + cx;
            const double v = fy * y * iz + cy;
            const double uR = u - fb * iz - doff;
            const double* o = obs + i * 3;
            const double r0 = std::clamp(u - o[0], -1e4, 1e4);
            const double r1 = std::clamp(v - o[1], -1e4, 1e4);
            if (std::abs(r0) >= 1e4 || std::abs(r1) >= 1e4) continue;
            const double r2 = std::clamp(uR - o[2], -1e4, 1e4);
            const double iz2 = iz * iz;
            const double* w = weights + i * 3;
            const uint8_t* m = mask + i * 3;

            const double raw[3] = {r0, r1, r2};
            const double J[3][6] = {
                {-fx * x * y * iz2, fx * (1.0 + x * x * iz2), -fx * y * iz, fx * iz, 0.0, -fx * x * iz2},
                {-fy * (1.0 + y * y * iz2), fy * x * y * iz2, fy * x * iz, 0.0, fy * iz, -fy * y * iz2},
                {(-fx * x + fb) * y * iz2, fx + (fx * x - fb) * x * iz2, -fx * y * iz, fx * iz, 0.0, (-fx * x + fb) * iz2}
            };
            for (int c = 0; c < 3; ++c) {
                if (!m[c]) continue;
                const double wc = w[c];
                if (wc <= 0.0) continue;
                const double wr = raw[c] * wc;
                const double a = std::abs(wr);
                const double irls = (a <= delta_huber) ? 1.0 : (delta_huber / std::max(a, 1e-12));
                const double ww = wc * wc * irls;
                const double wgr = wc * irls * wr;
                const double* Jc = J[c];
                for (int r = 0; r < 6; ++r) {
                    const double Jr = Jc[r];
                    g[r] += Jr * wgr;
                    const double Jrw = Jr * ww;
                    for (int k = r; k < 6; ++k) {
                        H[r * 6 + k] += Jrw * Jc[k];
                    }
                }
            }
        }
        for (int r = 0; r < 6; ++r) {
            for (int k = 0; k < r; ++k) H[r * 6 + k] = H[k * 6 + r];
        }
        std::memcpy(H_final, H, sizeof(H));

        bool finite_hg = true;
        double max_g = 0.0;
        for (int r = 0; r < 6; ++r) {
            if (!std::isfinite(g[r])) finite_hg = false;
            max_g = std::max(max_g, std::abs(g[r]));
            for (int k = 0; k < 6; ++k) {
                if (!std::isfinite(H[r * 6 + k])) finite_hg = false;
            }
        }
        if (!finite_hg) break;
        if (max_g < 1e-9) { converged = 1; break; }

        double diag[6];
        for (int r = 0; r < 6; ++r) diag[r] = std::max(H[r * 6 + r], 1e-6);

        bool accepted = false;
        for (int lm_try = 0; lm_try < 6; ++lm_try) {
            double H_d[36];
            std::memcpy(H_d, H, sizeof(H));
            for (int r = 0; r < 6; ++r) H_d[r * 6 + r] += lam * diag[r];
            double step[6];
            if (!solve_6x6(H_d, g, step)) {
                lam = std::min(lam * 10.0, 1e8);
                continue;
            }
            double dR[9], R_cand[9], t_cand[3];
            rotvec_to_mat(step, dR);
            mat3_mul(dR, R_cw, R_cand);
            t_cand[0] = dR[0] * t_cw[0] + dR[1] * t_cw[1] + dR[2] * t_cw[2] + step[3];
            t_cand[1] = dR[3] * t_cw[0] + dR[4] * t_cw[1] + dR[5] * t_cw[2] + step[4];
            t_cand[2] = dR[6] * t_cw[0] + dR[7] * t_cw[1] + dR[8] * t_cw[2] + step[5];
            const double cost_c = eval_cost(R_cand, t_cand);
            if (std::isfinite(cost_c) && cost_c < cur_cost) {
                const double rel_impr = (cur_cost - cost_c) / std::max(cur_cost, 1e-12);
                std::memcpy(R_cw, R_cand, 9 * sizeof(double));
                std::memcpy(t_cw, t_cand, 3 * sizeof(double));
                cur_cost = cost_c;
                lam = std::max(lam * 0.3, 1e-7);
                accepted = true;
                double sn2 = 0.0;
                for (int r = 0; r < 6; ++r) sn2 += step[r] * step[r];
                if (std::sqrt(sn2) < 1e-7 || rel_impr < 1e-9) converged = 1;
                break;
            }
            lam = std::min(lam * 5.0, 1e8);
        }
        if (converged || !accepted) {
            if (!accepted) {
                double gn2 = 0.0;
                for (int r = 0; r < 6; ++r) gn2 += g[r] * g[r];
                if (std::sqrt(gn2) < 1e-5) converged = 1;
            }
            break;
        }
    }

    *out_final_cost = cur_cost;
    *out_converged = converged;
    std::memcpy(out_H_final, H_final, sizeof(H_final));
    return 1;
}

VSLAM_EXPORT int32_t refine_bidirectional_stereo_lm_f64(
    double* R_io,                  // [9] in/out (T_s_t rotation)
    double* t_io,                  // [3] in/out (T_s_t translation)
    const double* src_pts,         // [n1 * 3]
    const double* tgt_pix,         // [n1 * 2]
    int32_t n1,
    const double* tgt_pts,         // [n2 * 3]
    const double* src_pix,         // [n2 * 2]
    int32_t n2,
    double fx, double fy, double cx, double cy,
    double f_scale,
    int32_t max_iter,
    double* out_initial_cost,
    double* out_final_cost
) {
    auto eval_cost = [&](const double R[9], const double t[3]) -> double {
        double cost = 0.0;
        for (int32_t i = 0; i < n1; ++i) {
            const double dx = src_pts[i * 3 + 0] - t[0];
            const double dy = src_pts[i * 3 + 1] - t[1];
            const double dz = src_pts[i * 3 + 2] - t[2];
            // c1 = R^T @ d1
            const double x = R[0] * dx + R[3] * dy + R[6] * dz;
            const double y = R[1] * dx + R[4] * dy + R[7] * dz;
            const double z = R[2] * dx + R[5] * dy + R[8] * dz;
            double eu = 1e4, ev = 1e4;
            if (z > 0.0) {
                const double iz = 1.0 / std::max(z, 1e-9);
                eu = std::clamp(fx * x * iz + cx - tgt_pix[i * 2 + 0], -1e4, 1e4);
                ev = std::clamp(fy * y * iz + cy - tgt_pix[i * 2 + 1], -1e4, 1e4);
            }
            const double au = std::abs(eu), av = std::abs(ev);
            cost += (au <= f_scale) ? (0.5 * eu * eu) : (f_scale * (au - 0.5 * f_scale));
            cost += (av <= f_scale) ? (0.5 * ev * ev) : (f_scale * (av - 0.5 * f_scale));
        }
        for (int32_t i = 0; i < n2; ++i) {
            const double px = tgt_pts[i * 3 + 0];
            const double py = tgt_pts[i * 3 + 1];
            const double pz = tgt_pts[i * 3 + 2];
            // c2 = R @ X_t + t
            const double x = R[0] * px + R[1] * py + R[2] * pz + t[0];
            const double y = R[3] * px + R[4] * py + R[5] * pz + t[1];
            const double z = R[6] * px + R[7] * py + R[8] * pz + t[2];
            double eu = 1e4, ev = 1e4;
            if (z > 0.0) {
                const double iz = 1.0 / std::max(z, 1e-9);
                eu = std::clamp(fx * x * iz + cx - src_pix[i * 2 + 0], -1e4, 1e4);
                ev = std::clamp(fy * y * iz + cy - src_pix[i * 2 + 1], -1e4, 1e4);
            }
            const double au = std::abs(eu), av = std::abs(ev);
            cost += (au <= f_scale) ? (0.5 * eu * eu) : (f_scale * (au - 0.5 * f_scale));
            cost += (av <= f_scale) ? (0.5 * ev * ev) : (f_scale * (av - 0.5 * f_scale));
        }
        return cost;
    };

    double cur_cost = eval_cost(R_io, t_io);
    *out_initial_cost = cur_cost;
    double lam = 1e-3;

    for (int32_t it = 0; it < max_iter; ++it) {
        double H[36];
        double g[6];
        std::memset(H, 0, sizeof(H));
        std::memset(g, 0, sizeof(g));

        // Forward terms
        for (int32_t i = 0; i < n1; ++i) {
            const double dx = src_pts[i * 3 + 0] - t_io[0];
            const double dy = src_pts[i * 3 + 1] - t_io[1];
            const double dz = src_pts[i * 3 + 2] - t_io[2];
            const double x = R_io[0] * dx + R_io[3] * dy + R_io[6] * dz;
            const double y = R_io[1] * dx + R_io[4] * dy + R_io[7] * dz;
            const double z = R_io[2] * dx + R_io[5] * dy + R_io[8] * dz;
            if (z <= 1e-9) continue;
            const double iz = 1.0 / z;
            const double eu = fx * x * iz + cx - tgt_pix[i * 2 + 0];
            const double ev = fy * y * iz + cy - tgt_pix[i * 2 + 1];
            if (std::abs(eu) >= 1e4 || std::abs(ev) >= 1e4) continue;
            const double iz2 = iz * iz;
            const double au = std::abs(eu), av = std::abs(ev);
            const double wu = (au <= f_scale) ? 1.0 : (f_scale / std::max(au, 1e-12));
            const double wv = (av <= f_scale) ? 1.0 : (f_scale / std::max(av, 1e-12));
            const double Ju[6] = {fx * x * y * iz2, -fx * (1.0 + x * x * iz2), fx * y * iz, -fx * iz, 0.0, fx * x * iz2};
            const double Jv[6] = {fy * (1.0 + y * y * iz2), -fy * x * y * iz2, -fy * x * iz, 0.0, -fy * iz, fy * y * iz2};
            for (int r = 0; r < 6; ++r) {
                g[r] += Ju[r] * wu * eu + Jv[r] * wv * ev;
                const double Jur = Ju[r] * wu, Jvr = Jv[r] * wv;
                for (int k = r; k < 6; ++k) {
                    H[r * 6 + k] += Jur * Ju[k] + Jvr * Jv[k];
                }
            }
        }

        // Reverse terms
        for (int32_t i = 0; i < n2; ++i) {
            const double px = tgt_pts[i * 3 + 0];
            const double py = tgt_pts[i * 3 + 1];
            const double pz = tgt_pts[i * 3 + 2];
            const double x = R_io[0] * px + R_io[1] * py + R_io[2] * pz + t_io[0];
            const double y = R_io[3] * px + R_io[4] * py + R_io[5] * pz + t_io[1];
            const double z = R_io[6] * px + R_io[7] * py + R_io[8] * pz + t_io[2];
            if (z <= 1e-9) continue;
            const double iz = 1.0 / z;
            const double eu = fx * x * iz + cx - src_pix[i * 2 + 0];
            const double ev = fy * y * iz + cy - src_pix[i * 2 + 1];
            if (std::abs(eu) >= 1e4 || std::abs(ev) >= 1e4) continue;
            const double iz2 = iz * iz;
            const double au = std::abs(eu), av = std::abs(ev);
            const double wu = (au <= f_scale) ? 1.0 : (f_scale / std::max(au, 1e-12));
            const double wv = (av <= f_scale) ? 1.0 : (f_scale / std::max(av, 1e-12));
            const double du0 = fx * iz, du2 = -fx * x * iz2;
            const double dv1 = fy * iz, dv2 = -fy * y * iz2;
            const double dut0 = du0 * R_io[0] + du2 * R_io[6];
            const double dut1 = du0 * R_io[1] + du2 * R_io[7];
            const double dut2 = du0 * R_io[2] + du2 * R_io[8];
            const double dvt0 = dv1 * R_io[3] + dv2 * R_io[6];
            const double dvt1 = dv1 * R_io[4] + dv2 * R_io[7];
            const double dvt2 = dv1 * R_io[5] + dv2 * R_io[8];
            const double Ju2[6] = {
                py * dut2 - pz * dut1,
                pz * dut0 - px * dut2,
                px * dut1 - py * dut0,
                dut0, dut1, dut2
            };
            const double Jv2[6] = {
                py * dvt2 - pz * dvt1,
                pz * dvt0 - px * dvt2,
                px * dvt1 - py * dvt0,
                dvt0, dvt1, dvt2
            };
            for (int r = 0; r < 6; ++r) {
                g[r] += Ju2[r] * wu * eu + Jv2[r] * wv * ev;
                const double Jur = Ju2[r] * wu, Jvr = Jv2[r] * wv;
                for (int k = r; k < 6; ++k) {
                    H[r * 6 + k] += Jur * Ju2[k] + Jvr * Jv2[k];
                }
            }
        }
        for (int r = 0; r < 6; ++r) {
            for (int k = 0; k < r; ++k) H[r * 6 + k] = H[k * 6 + r];
        }

        double max_g = 0.0;
        for (int r = 0; r < 6; ++r) max_g = std::max(max_g, std::abs(g[r]));
        if (max_g < 1e-9) break;

        double diag[6];
        for (int r = 0; r < 6; ++r) diag[r] = std::max(H[r * 6 + r], 1e-6);

        bool accepted = false;
        for (int lm_try = 0; lm_try < 6; ++lm_try) {
            double H_d[36];
            std::memcpy(H_d, H, sizeof(H));
            for (int r = 0; r < 6; ++r) H_d[r * 6 + r] += lam * diag[r];
            double step[6];
            if (!solve_6x6(H_d, g, step)) {
                lam *= 10.0;
                continue;
            }
            double dR[9], R_cand[9], t_cand[3];
            rotvec_to_mat(step, dR);
            mat3_mul(R_io, dR, R_cand);
            t_cand[0] = t_io[0] + R_io[0] * step[3] + R_io[1] * step[4] + R_io[2] * step[5];
            t_cand[1] = t_io[1] + R_io[3] * step[3] + R_io[4] * step[4] + R_io[5] * step[5];
            t_cand[2] = t_io[2] + R_io[6] * step[3] + R_io[7] * step[4] + R_io[8] * step[5];
            const double new_cost = eval_cost(R_cand, t_cand);
            if (new_cost < cur_cost) {
                const double gain = cur_cost - new_cost;
                std::memcpy(R_io, R_cand, 9 * sizeof(double));
                std::memcpy(t_io, t_cand, 3 * sizeof(double));
                cur_cost = new_cost;
                lam = std::max(lam * 0.3, 1e-7);
                accepted = true;
                double sn2 = 0.0;
                for (int r = 0; r < 6; ++r) sn2 += step[r] * step[r];
                if (gain <= 1e-8 * std::max(1.0, cur_cost) && std::sqrt(sn2) < 1e-6) {
                    lam = -1.0;
                }
                break;
            }
            lam *= 5.0;
        }
        if (!accepted || lam < 0.0) break;
    }

    *out_final_cost = cur_cost;
    return 1;
}

}  // extern "C"
