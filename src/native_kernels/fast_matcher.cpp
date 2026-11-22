#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

inline int32_t hsum_epi32(__m512i v) {
    return _mm512_reduce_add_epi32(v);
}

// Branchless AVX-512 update of 16 parallel backward top-2 trackers (for j0..j0+15)
// given 16 distances `d` from row `row_idx`.
inline void update_b_top2_16(
    __m512i d,
    __m512i row_v,
    __m512i& bd1,
    __m512i& bd2,
    __m512i& bbest
) {
    __mmask16 lt1 = _mm512_cmp_epi32_mask(d, bd1, _MM_CMPINT_LT);
    __mmask16 lt2 = _mm512_cmp_epi32_mask(d, bd2, _MM_CMPINT_LT);
    // Where d < bd1: bd2 = bd1, bd1 = d, bbest = row_v
    // Where !(d < bd1) && (d < bd2): bd2 = d
    __m512i new_bd2 = _mm512_mask_mov_epi32(bd2, lt2, d);
    bd2 = _mm512_mask_mov_epi32(new_bd2, lt1, bd1);
    bd1 = _mm512_mask_mov_epi32(bd1, lt1, d);
    bbest = _mm512_mask_mov_epi32(bbest, lt1, row_v);
}

// Reduce a 16-lane vector of distances (for indices j0..j0+15) into scalar top-2 (d1, d2, best).
inline void reduce_f_top2_16(
    __m512i d,
    int32_t j0,
    int32_t& d1,
    int32_t& d2,
    int32_t& best
) {
    // Fast skip: if min of all 16 lanes >= d2, none can enter top-2!
    int32_t block_min = _mm512_reduce_min_epi32(d);
    if (block_min >= d2) {
        return;
    }
    alignas(64) int32_t buf[16];
    _mm512_store_si512(buf, d);
    for (int32_t lane = 0; lane < 16; ++lane) {
        int32_t val = buf[lane];
        if (val < d1) {
            d2 = d1;
            d1 = val;
            best = j0 + lane;
        } else if (val < d2) {
            d2 = val;
        }
    }
}

// High-throughput packed-16 AVX-512 VNNI kernel with zero horizontal reductions.
// B is packed into blocks of 16 rows: b_pack[jb * 32 * 16 + k * 16 + lane] holds
// 4 consecutive int8 bytes (dims 4k..4k+3) of row (jb*16 + lane).
// Each vpdpbusd computes 16 (i, j..j+15) 4-byte dot-product accumulations in parallel!
void match_u8_128_vnni_packed16(
    const uint32_t* a_u32,   // [m, 32] (each uint32 = 4 uint8 dims)
    const int32_t* b_pack,   // [n_blocks, 32, 16] (each int32 = 4 int8 dims)
    const int32_t* ca,
    const int32_t* nb_pad,   // [n_pad] padded with INF
    int32_t m,
    int32_t n,
    int32_t n_pad,
    int32_t* best_f,
    int32_t* d1_f,
    int32_t* d2_f,
    int32_t* best_b,
    int32_t* d1_b,
    int32_t* d2_b
) {
    const int32_t INF = std::numeric_limits<int32_t>::max();
    for (int32_t i = 0; i < m; ++i) {
        best_f[i] = -1;
        d1_f[i] = INF;
        d2_f[i] = INF;
    }

    const int32_t n_blocks = n_pad / 16;
    const __m512i v_inf = _mm512_set1_epi32(INF);
    const __m512i v_neg1 = _mm512_set1_epi32(-1);

    auto process_row_vec = [&](int32_t row, __m512i acc, __m512i v_nb, int32_t j0,
                               __mmask16 valid_j, __m512i& bd1, __m512i& bd2, __m512i& bbest) __attribute__((always_inline)) {
        __m512i d = _mm512_sub_epi32(
            _mm512_add_epi32(_mm512_set1_epi32(ca[row]), v_nb),
            _mm512_slli_epi32(acc, 1)
        );
        if (valid_j != static_cast<__mmask16>(0xFFFF)) {
            d = _mm512_mask_mov_epi32(v_inf, valid_j, d);
        }
        reduce_f_top2_16(d, j0, d1_f[row], d2_f[row], best_f[row]);
        update_b_top2_16(d, _mm512_set1_epi32(row), bd1, bd2, bbest);
    };

    // Process 32 columns of B (2 blocks of 16) x 6 rows of A in 12 ZMM registers (zero spills!)
    int32_t jb = 0;
    for (; jb + 1 < n_blocks; jb += 2) {
        const int32_t j0 = jb * 16;
        const int32_t j1 = (jb + 1) * 16;
        const int32_t* bp0 = b_pack + static_cast<size_t>(jb) * 32 * 16;
        const int32_t* bp1 = b_pack + static_cast<size_t>(jb + 1) * 32 * 16;
        const __m512i v_nb0 = _mm512_loadu_si512(nb_pad + j0);
        const __m512i v_nb1 = _mm512_loadu_si512(nb_pad + j1);
        const __mmask16 valid_j0 = static_cast<__mmask16>(0xFFFF);
        const __mmask16 valid_j1 = (j1 + 16 <= n)
            ? static_cast<__mmask16>(0xFFFF)
            : static_cast<__mmask16>((1u << (n - j1)) - 1u);

        __m512i bd1_0 = v_inf, bd2_0 = v_inf, bbest_0 = v_neg1;
        __m512i bd1_1 = v_inf, bd2_1 = v_inf, bbest_1 = v_neg1;

        int32_t i = 0;
        for (; i + 5 < m; i += 6) {
            const uint32_t* pa0 = a_u32 + static_cast<size_t>(i + 0) * 32;
            const uint32_t* pa1 = a_u32 + static_cast<size_t>(i + 1) * 32;
            const uint32_t* pa2 = a_u32 + static_cast<size_t>(i + 2) * 32;
            const uint32_t* pa3 = a_u32 + static_cast<size_t>(i + 3) * 32;
            const uint32_t* pa4 = a_u32 + static_cast<size_t>(i + 4) * 32;
            const uint32_t* pa5 = a_u32 + static_cast<size_t>(i + 5) * 32;

            __m512i c00 = _mm512_setzero_si512(), c01 = _mm512_setzero_si512();
            __m512i c10 = _mm512_setzero_si512(), c11 = _mm512_setzero_si512();
            __m512i c20 = _mm512_setzero_si512(), c21 = _mm512_setzero_si512();
            __m512i c30 = _mm512_setzero_si512(), c31 = _mm512_setzero_si512();
            __m512i c40 = _mm512_setzero_si512(), c41 = _mm512_setzero_si512();
            __m512i c50 = _mm512_setzero_si512(), c51 = _mm512_setzero_si512();

            for (int32_t k = 0; k < 32; ++k) {
                __m512i b0 = _mm512_load_si512(bp0 + k * 16);
                __m512i b1 = _mm512_load_si512(bp1 + k * 16);
                __m512i a0 = _mm512_set1_epi32(static_cast<int32_t>(pa0[k]));
                __m512i a1 = _mm512_set1_epi32(static_cast<int32_t>(pa1[k]));
                __m512i a2 = _mm512_set1_epi32(static_cast<int32_t>(pa2[k]));
                __m512i a3 = _mm512_set1_epi32(static_cast<int32_t>(pa3[k]));
                __m512i a4 = _mm512_set1_epi32(static_cast<int32_t>(pa4[k]));
                __m512i a5 = _mm512_set1_epi32(static_cast<int32_t>(pa5[k]));
                c00 = _mm512_dpbusd_epi32(c00, a0, b0); c01 = _mm512_dpbusd_epi32(c01, a0, b1);
                c10 = _mm512_dpbusd_epi32(c10, a1, b0); c11 = _mm512_dpbusd_epi32(c11, a1, b1);
                c20 = _mm512_dpbusd_epi32(c20, a2, b0); c21 = _mm512_dpbusd_epi32(c21, a2, b1);
                c30 = _mm512_dpbusd_epi32(c30, a3, b0); c31 = _mm512_dpbusd_epi32(c31, a3, b1);
                c40 = _mm512_dpbusd_epi32(c40, a4, b0); c41 = _mm512_dpbusd_epi32(c41, a4, b1);
                c50 = _mm512_dpbusd_epi32(c50, a5, b0); c51 = _mm512_dpbusd_epi32(c51, a5, b1);
            }

            process_row_vec(i + 0, c00, v_nb0, j0, valid_j0, bd1_0, bd2_0, bbest_0);
            process_row_vec(i + 0, c01, v_nb1, j1, valid_j1, bd1_1, bd2_1, bbest_1);
            process_row_vec(i + 1, c10, v_nb0, j0, valid_j0, bd1_0, bd2_0, bbest_0);
            process_row_vec(i + 1, c11, v_nb1, j1, valid_j1, bd1_1, bd2_1, bbest_1);
            process_row_vec(i + 2, c20, v_nb0, j0, valid_j0, bd1_0, bd2_0, bbest_0);
            process_row_vec(i + 2, c21, v_nb1, j1, valid_j1, bd1_1, bd2_1, bbest_1);
            process_row_vec(i + 3, c30, v_nb0, j0, valid_j0, bd1_0, bd2_0, bbest_0);
            process_row_vec(i + 3, c31, v_nb1, j1, valid_j1, bd1_1, bd2_1, bbest_1);
            process_row_vec(i + 4, c40, v_nb0, j0, valid_j0, bd1_0, bd2_0, bbest_0);
            process_row_vec(i + 4, c41, v_nb1, j1, valid_j1, bd1_1, bd2_1, bbest_1);
            process_row_vec(i + 5, c50, v_nb0, j0, valid_j0, bd1_0, bd2_0, bbest_0);
            process_row_vec(i + 5, c51, v_nb1, j1, valid_j1, bd1_1, bd2_1, bbest_1);
        }

        for (; i < m; ++i) {
            const uint32_t* pa = a_u32 + static_cast<size_t>(i) * 32;
            __m512i c0 = _mm512_setzero_si512(), c1 = _mm512_setzero_si512();
            for (int32_t k = 0; k < 32; ++k) {
                __m512i a0 = _mm512_set1_epi32(static_cast<int32_t>(pa[k]));
                c0 = _mm512_dpbusd_epi32(c0, a0, _mm512_load_si512(bp0 + k * 16));
                c1 = _mm512_dpbusd_epi32(c1, a0, _mm512_load_si512(bp1 + k * 16));
            }
            process_row_vec(i, c0, v_nb0, j0, valid_j0, bd1_0, bd2_0, bbest_0);
            process_row_vec(i, c1, v_nb1, j1, valid_j1, bd1_1, bd2_1, bbest_1);
        }

        alignas(64) int32_t s_bd1[16], s_bd2[16], s_bb[16];
        _mm512_store_si512(s_bd1, bd1_0); _mm512_store_si512(s_bd2, bd2_0); _mm512_store_si512(s_bb, bbest_0);
        for (int32_t lane = 0; lane < 16; ++lane) {
            d1_b[j0 + lane] = s_bd1[lane]; d2_b[j0 + lane] = s_bd2[lane]; best_b[j0 + lane] = s_bb[lane];
        }
        _mm512_store_si512(s_bd1, bd1_1); _mm512_store_si512(s_bd2, bd2_1); _mm512_store_si512(s_bb, bbest_1);
        const int32_t lim1 = (j1 + 16 <= n) ? 16 : (n - j1);
        for (int32_t lane = 0; lane < lim1; ++lane) {
            d1_b[j1 + lane] = s_bd1[lane]; d2_b[j1 + lane] = s_bd2[lane]; best_b[j1 + lane] = s_bb[lane];
        }
    }

    for (; jb < n_blocks; ++jb) {
        const int32_t j0 = jb * 16;
        const int32_t* bp = b_pack + static_cast<size_t>(jb) * 32 * 16;
        const __m512i v_nb = _mm512_loadu_si512(nb_pad + j0);
        const __mmask16 valid_j = (j0 + 16 <= n)
            ? static_cast<__mmask16>(0xFFFF)
            : static_cast<__mmask16>((1u << (n - j0)) - 1u);
        __m512i bd1 = v_inf, bd2 = v_inf, bbest = v_neg1;
        for (int32_t i = 0; i < m; ++i) {
            const uint32_t* pa = a_u32 + static_cast<size_t>(i) * 32;
            __m512i acc = _mm512_setzero_si512();
            for (int32_t k = 0; k < 32; ++k) {
                acc = _mm512_dpbusd_epi32(
                    acc,
                    _mm512_set1_epi32(static_cast<int32_t>(pa[k])),
                    _mm512_load_si512(bp + k * 16)
                );
            }
            process_row_vec(i, acc, v_nb, j0, valid_j, bd1, bd2, bbest);
        }
        alignas(64) int32_t s_bd1[16], s_bd2[16], s_bb[16];
        _mm512_store_si512(s_bd1, bd1); _mm512_store_si512(s_bd2, bd2); _mm512_store_si512(s_bb, bbest);
        const int32_t lim = (j0 + 16 <= n) ? 16 : (n - j0);
        for (int32_t lane = 0; lane < lim; ++lane) {
            d1_b[j0 + lane] = s_bd1[lane]; d2_b[j0 + lane] = s_bd2[lane]; best_b[j0 + lane] = s_bb[lane];
        }
    }
}

}  // namespace

extern "C" {

int32_t match_sift_mutual_f32(
    const float* a_f32,
    int32_t m,
    const float* b_f32,
    int32_t n,
    double ratio,
    int32_t* out_pairs,
    int32_t* out_count,
    int32_t* out_ambig_a,
    int32_t* out_ambig_a_cnt,
    int32_t* out_ambig_b,
    int32_t* out_ambig_b_cnt
) {
    if (m < 2 || n < 2) {
        *out_count = 0;
        *out_ambig_a_cnt = 0;
        *out_ambig_b_cnt = 0;
        return 1;
    }

    const int32_t n_pad = ((n + 15) / 16) * 16;
    const int32_t n_blocks = n_pad / 16;

    std::vector<uint8_t> a_u8(static_cast<size_t>(m) * 128);
    std::vector<int32_t> ca(m);
    std::vector<int32_t> nb_pad(n_pad, 0);
    // 64-byte aligned packed B buffer
    int32_t* b_pack = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&b_pack), 64,
                       static_cast<size_t>(n_blocks) * 32 * 16 * sizeof(int32_t)) != 0) {
        return 0;
    }
    std::memset(b_pack, 0, static_cast<size_t>(n_blocks) * 32 * 16 * sizeof(int32_t));

    const __m512 v_zero = _mm512_setzero_ps();
    const __m512 v_255 = _mm512_set1_ps(255.0f);
    const __m512i v_128 = _mm512_set1_epi32(128);

    // Convert A -> uint8 + ca[i]
    for (int32_t i = 0; i < m; ++i) {
        const float* row = a_f32 + static_cast<size_t>(i) * 128;
        uint8_t* dst = a_u8.data() + static_cast<size_t>(i) * 128;
        __m512i sum_v = _mm512_setzero_si512();
        __m512i sq_v = _mm512_setzero_si512();
        for (int32_t k = 0; k < 128; k += 16) {
            __m512 vf = _mm512_loadu_ps(row + k);
            __mmask16 bad_lo = _mm512_cmp_ps_mask(vf, v_zero, _CMP_LT_OQ);
            __mmask16 bad_hi = _mm512_cmp_ps_mask(vf, v_255, _CMP_GT_OQ);
            __mmask16 bad_nan = _mm512_cmp_ps_mask(vf, vf, _CMP_NEQ_UQ);
            __m512i vi = _mm512_cvttps_epi32(vf);
            __m512 vr = _mm512_cvtepi32_ps(vi);
            __mmask16 bad_frac = _mm512_cmp_ps_mask(vf, vr, _CMP_NEQ_OQ);
            if (bad_lo | bad_hi | bad_nan | bad_frac) {
                free(b_pack);
                return 0;
            }
            sum_v = _mm512_add_epi32(sum_v, vi);
            sq_v = _mm512_add_epi32(sq_v, _mm512_mullo_epi32(vi, vi));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + k), _mm512_cvtusepi32_epi8(vi));
        }
        ca[i] = hsum_epi32(sq_v) - 256 * hsum_epi32(sum_v);
    }

    // Convert B -> shifted int8 directly into packed-16 layout + nb_pad[j]
    alignas(64) int8_t tmp_b[128];
    for (int32_t j = 0; j < n; ++j) {
        const float* row = b_f32 + static_cast<size_t>(j) * 128;
        __m512i sq_v = _mm512_setzero_si512();
        for (int32_t k = 0; k < 128; k += 16) {
            __m512 vf = _mm512_loadu_ps(row + k);
            __mmask16 bad_lo = _mm512_cmp_ps_mask(vf, v_zero, _CMP_LT_OQ);
            __mmask16 bad_hi = _mm512_cmp_ps_mask(vf, v_255, _CMP_GT_OQ);
            __mmask16 bad_nan = _mm512_cmp_ps_mask(vf, vf, _CMP_NEQ_UQ);
            __m512i vi = _mm512_cvttps_epi32(vf);
            __m512 vr = _mm512_cvtepi32_ps(vi);
            __mmask16 bad_frac = _mm512_cmp_ps_mask(vf, vr, _CMP_NEQ_OQ);
            if (bad_lo | bad_hi | bad_nan | bad_frac) {
                free(b_pack);
                return 0;
            }
            sq_v = _mm512_add_epi32(sq_v, _mm512_mullo_epi32(vi, vi));
            __m512i vs = _mm512_sub_epi32(vi, v_128);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp_b + k), _mm512_cvtsepi32_epi8(vs));
        }
        nb_pad[j] = hsum_epi32(sq_v);
        const int32_t jb = j >> 4;
        const int32_t lane = j & 15;
        int32_t* bp = b_pack + static_cast<size_t>(jb) * 32 * 16 + lane;
        const int32_t* src32 = reinterpret_cast<const int32_t*>(tmp_b);
        for (int32_t k = 0; k < 32; ++k) {
            bp[k * 16] = src32[k];
        }
    }

    std::vector<int32_t> best_f(m), d1_f(m), d2_f(m);
    std::vector<int32_t> best_b(n), d1_b(n), d2_b(n);
    match_u8_128_vnni_packed16(
        reinterpret_cast<const uint32_t*>(a_u8.data()),
        b_pack,
        ca.data(),
        nb_pad.data(),
        m,
        n,
        n_pad,
        best_f.data(),
        d1_f.data(),
        d2_f.data(),
        best_b.data(),
        d1_b.data(),
        d2_b.data()
    );
    free(b_pack);

    int32_t amb_a = 0;
    int32_t amb_b = 0;
    std::vector<uint8_t> ok_f(m, 0);
    std::vector<uint8_t> ok_b(n, 0);

    // Every integer squared L2 distance for 128-D [0..255] descriptors is at most
    // 128 * 255^2 = 8,323,200 < 2^24 = 16,777,216, so float(d) is bit-exact and
    // sqrtf(float(d)) matches OpenCV cv.BFMatcher(NORM_L2)'s float32 distance identically.
    for (int32_t i = 0; i < m; ++i) {
        if (d1_f[i] < d2_f[i]) {
            const double s1 = static_cast<double>(std::sqrt(static_cast<float>(d1_f[i])));
            const double s2 = static_cast<double>(std::sqrt(static_cast<float>(d2_f[i])));
            if (s1 < ratio * s2) {
                ok_f[i] = 1;
            }
        }
    }
    for (int32_t j = 0; j < n; ++j) {
        if (d1_b[j] < d2_b[j]) {
            const double s1 = static_cast<double>(std::sqrt(static_cast<float>(d1_b[j])));
            const double s2 = static_cast<double>(std::sqrt(static_cast<float>(d2_b[j])));
            if (s1 < ratio * s2) {
                ok_b[j] = 1;
            }
        }
    }

    int32_t cnt = 0;
    for (int32_t i = 0; i < m; ++i) {
        if (ok_f[i]) {
            const int32_t j = best_f[i];
            if (j >= 0 && ok_b[j] && best_b[j] == i) {
                out_pairs[2 * cnt + 0] = i;
                out_pairs[2 * cnt + 1] = j;
                ++cnt;
            }
        }
    }
    *out_count = cnt;
    *out_ambig_a_cnt = amb_a;
    *out_ambig_b_cnt = amb_b;
    return 1;
}

int32_t match_projected_landmarks_f32(
    const float* proj_uv,       // [n_lm, 2]
    const float* lm_descs,      // [n_lm, 128]
    const int16_t* lm_oct,      // [n_lm] or nullptr
    int32_t n_lm,
    const float* f_pixels,      // [n_feat, 2]
    const float* f_descs,       // [n_feat, 128]
    const int16_t* f_oct,       // [n_feat] or nullptr
    int32_t n_feat,
    float radius_px,
    float ratio,
    int32_t max_octave_diff,
    float max_l2_distance,
    float max_single_l2_distance,
    int32_t* out_pairs,         // [min(n_lm, n_feat), 2] -> (lm_idx, feat_idx)
    int32_t* out_count
) {
    *out_count = 0;
    if (n_lm <= 0 || n_feat <= 0) {
        return 1;
    }
    const float rad = (radius_px > 1.0f) ? radius_px : 1.0f;
    const float inv_cell = 1.0f / rad;
    const float rad_sq = rad * rad;
    const float ratio_sq = ratio * ratio;
    const float max_dist_sq = max_l2_distance * max_l2_distance;
    const float max_single_sq = max_single_l2_distance * max_single_l2_distance;
    const bool use_oct = (lm_oct != nullptr && f_oct != nullptr && max_octave_diff >= 0);

    // Compute spatial grid bounds over frame features
    int32_t min_gx = 2147483647, max_gx = -2147483647;
    int32_t min_gy = 2147483647, max_gy = -2147483647;
    std::vector<int32_t> fgx(n_feat), fgy(n_feat);
    for (int32_t f = 0; f < n_feat; ++f) {
        const float fx = f_pixels[2 * f + 0];
        const float fy = f_pixels[2 * f + 1];
        if (!std::isfinite(fx) || !std::isfinite(fy)) {
            return 0;
        }
        const int32_t gx = static_cast<int32_t>(std::floor(fx * inv_cell));
        const int32_t gy = static_cast<int32_t>(std::floor(fy * inv_cell));
        fgx[f] = gx;
        fgy[f] = gy;
        if (gx < min_gx) min_gx = gx;
        if (gx > max_gx) max_gx = gx;
        if (gy < min_gy) min_gy = gy;
        if (gy > max_gy) max_gy = gy;
    }
    const int64_t gw = static_cast<int64_t>(max_gx) - static_cast<int64_t>(min_gx) + 1;
    const int64_t gh = static_cast<int64_t>(max_gy) - static_cast<int64_t>(min_gy) + 1;
    if (gw <= 0 || gh <= 0 || gw * gh > 65536) {
        return 0;
    }
    const int32_t n_cells = static_cast<int32_t>(gw * gh);
    std::vector<int32_t> cell_offsets(n_cells + 1, 0);
    for (int32_t f = 0; f < n_feat; ++f) {
        const int32_t cid = (fgy[f] - min_gy) * static_cast<int32_t>(gw) + (fgx[f] - min_gx);
        cell_offsets[cid + 1] += 1;
    }
    for (int32_t c = 0; c < n_cells; ++c) {
        cell_offsets[c + 1] += cell_offsets[c];
    }
    std::vector<int32_t> cell_cursor(cell_offsets.begin(), cell_offsets.begin() + n_cells);
    std::vector<int32_t> grid_feats(n_feat);
    for (int32_t f = 0; f < n_feat; ++f) {
        const int32_t cid = (fgy[f] - min_gy) * static_cast<int32_t>(gw) + (fgx[f] - min_gx);
        grid_feats[cell_cursor[cid]++] = f;
    }

    std::vector<float> best_feat_d2(n_feat, 1e30f);
    std::vector<int32_t> best_feat_lm(n_feat, -1);

    const int32_t gw_i = static_cast<int32_t>(gw);
    const int32_t gh_i = static_cast<int32_t>(gh);

    for (int32_t i = 0; i < n_lm; ++i) {
        const float u = proj_uv[2 * i + 0];
        const float v = proj_uv[2 * i + 1];
        if (!std::isfinite(u) || !std::isfinite(v)) {
            continue;
        }
        const int32_t cx = static_cast<int32_t>(std::floor(u * inv_cell)) - min_gx;
        const int32_t cy = static_cast<int32_t>(std::floor(v * inv_cell)) - min_gy;
        if (cx < -1 || cx > gw_i || cy < -1 || cy > gh_i) {
            continue;
        }
        const int32_t y0 = (cy > 0) ? (cy - 1) : 0;
        const int32_t y1 = (cy + 1 < gh_i) ? (cy + 1) : (gh_i - 1);
        const int32_t x0 = (cx > 0) ? (cx - 1) : 0;
        const int32_t x1 = (cx + 1 < gw_i) ? (cx + 1) : (gw_i - 1);

        const float* lmd = lm_descs + static_cast<size_t>(i) * 128;
        const int32_t lo = use_oct ? static_cast<int32_t>(lm_oct[i]) : 0;

        float best_d2 = 1e30f;
        float second_d2 = 1e30f;
        int32_t best_f = -1;
        int32_t cand_cnt = 0;

        for (int32_t gy = y0; gy <= y1; ++gy) {
            const int32_t row_base = gy * gw_i;
            for (int32_t gx = x0; gx <= x1; ++gx) {
                const int32_t cid = row_base + gx;
                const int32_t beg = cell_offsets[cid];
                const int32_t end = cell_offsets[cid + 1];
                for (int32_t p = beg; p < end; ++p) {
                    const int32_t f = grid_feats[p];
                    const float dx = f_pixels[2 * f + 0] - u;
                    const float dy = f_pixels[2 * f + 1] - v;
                    if (dx * dx + dy * dy > rad_sq) {
                        continue;
                    }
                    if (use_oct) {
                        const int32_t diff_o = static_cast<int32_t>(f_oct[f]) - lo;
                        if (diff_o > max_octave_diff || diff_o < -max_octave_diff) {
                            continue;
                        }
                    }
                    const float* fd = f_descs + static_cast<size_t>(f) * 128;
                    __m512 acc0 = _mm512_setzero_ps();
                    __m512 acc1 = _mm512_setzero_ps();
                    for (int32_t k = 0; k < 128; k += 32) {
                        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(lmd + k), _mm512_loadu_ps(fd + k));
                        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(lmd + k + 16), _mm512_loadu_ps(fd + k + 16));
                        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
                        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
                    }
                    const float d2 = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
                    ++cand_cnt;
                    if (d2 < best_d2) {
                        second_d2 = best_d2;
                        best_d2 = d2;
                        best_f = f;
                    } else if (d2 < second_d2) {
                        second_d2 = d2;
                    }
                }
            }
        }
        if (cand_cnt == 0 || best_f < 0) {
            continue;
        }
        if (cand_cnt == 1) {
            if (best_d2 > max_single_sq) {
                continue;
            }
        } else {
            if (best_d2 > max_dist_sq) {
                continue;
            }
            if (second_d2 > 1e-12f && best_d2 >= ratio_sq * second_d2) {
                continue;
            }
        }
        if (best_feat_lm[best_f] < 0 || best_d2 < best_feat_d2[best_f]) {
            best_feat_d2[best_f] = best_d2;
            best_feat_lm[best_f] = i;
        }
    }

    int32_t cnt = 0;
    for (int32_t f = 0; f < n_feat; ++f) {
        if (best_feat_lm[f] >= 0) {
            out_pairs[2 * cnt + 0] = best_feat_lm[f];
            out_pairs[2 * cnt + 1] = f;
            ++cnt;
        }
    }
    *out_count = cnt;
    return 1;
}

}  // extern "C"
