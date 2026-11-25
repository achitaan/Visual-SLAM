#include "portable_export.h"
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int32_t PYR_PAD = 12;

#if defined(__AVX2__) && !defined(__AVX512F__)
inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(s);
    __m128 sums = _mm_add_ps(s, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

struct LkPyramid {
    int32_t max_level = 0;
    std::vector<int32_t> h;
    std::vector<int32_t> w;
    std::vector<int32_t> stride;
    std::vector<std::vector<float>> img;  // [level][(h + 2*PYR_PAD) * stride]
};

inline void fill_border(float* base, int32_t h, int32_t w, int32_t stride) {
    for (int32_t y = 0; y < h; ++y) {
        float* row = base + static_cast<size_t>(y + PYR_PAD) * stride + PYR_PAD;
        const float lv = row[0];
        const float rv = row[w - 1];
        for (int32_t k = 1; k <= PYR_PAD; ++k) {
            row[-k] = lv;
            row[w - 1 + k] = rv;
        }
    }
    const float* top_src = base + static_cast<size_t>(PYR_PAD) * stride;
    const float* bot_src = base + static_cast<size_t>(PYR_PAD + h - 1) * stride;
    for (int32_t k = 0; k < PYR_PAD; ++k) {
        std::memcpy(base + static_cast<size_t>(k) * stride, top_src, static_cast<size_t>(stride) * sizeof(float));
        std::memcpy(base + static_cast<size_t>(PYR_PAD + h + k) * stride, bot_src, static_cast<size_t>(stride) * sizeof(float));
    }
}

void pyr_down_simd(
    const float* VSLAM_RESTRICT src, int32_t sh, int32_t sw, int32_t sstride,
    float* VSLAM_RESTRICT dst, int32_t dh, int32_t dw, int32_t dstride
) {
    (void)sw;
    thread_local std::vector<float> tmp;
    const size_t need = static_cast<size_t>(sh) * dw;
    if (tmp.size() < need) tmp.resize(need);
#if defined(__AVX512F__)
    const __m512 v1 = _mm512_set1_ps(1.0f / 16.0f);
    const __m512 v4 = _mm512_set1_ps(4.0f / 16.0f);
    const __m512 v6 = _mm512_set1_ps(6.0f / 16.0f);
    const __m512i even_idx = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
#elif defined(__AVX2__)
    const __m256 v1 = _mm256_set1_ps(1.0f / 16.0f);
    const __m256 v4 = _mm256_set1_ps(4.0f / 16.0f);
    const __m256 v6 = _mm256_set1_ps(6.0f / 16.0f);
#endif

    for (int32_t y = 0; y < sh; ++y) {
        const float* sr = src + static_cast<size_t>(y + PYR_PAD) * sstride + PYR_PAD;
        float* tr = tmp.data() + static_cast<size_t>(y) * dw;
        int32_t dx = 0;
#if defined(__AVX512F__)
        for (; dx + 15 < dw; dx += 16) {
            const int32_t sx = 2 * dx;
            auto load_even = [&](int32_t off) -> __m512 {
                __m512 a = _mm512_loadu_ps(sr + sx + off);
                __m512 b = _mm512_loadu_ps(sr + sx + off + 16);
                return _mm512_permutex2var_ps(a, even_idx, b);
            };
            __m512 acc = _mm512_mul_ps(v6, load_even(0));
            acc = _mm512_fmadd_ps(v4, _mm512_add_ps(load_even(-1), load_even(1)), acc);
            acc = _mm512_fmadd_ps(v1, _mm512_add_ps(load_even(-2), load_even(2)), acc);
            _mm512_storeu_ps(tr + dx, acc);
        }
#endif
        for (; dx < dw; ++dx) {
            const int32_t sx = 2 * dx;
            tr[dx] = (sr[sx - 2] + 4.0f * sr[sx - 1] + 6.0f * sr[sx]
                    + 4.0f * sr[sx + 1] + sr[sx + 2]) * (1.0f / 16.0f);
        }
    }

    auto row_y = [&](int32_t sy) -> const float* {
        if (sy < 0) sy = 0;
        if (sy >= sh) sy = sh - 1;
        return tmp.data() + static_cast<size_t>(sy) * dw;
    };

    for (int32_t dy = 0; dy < dh; ++dy) {
        const int32_t sy = 2 * dy;
        const float* rm2 = row_y(sy - 2);
        const float* rm1 = row_y(sy - 1);
        const float* r0  = row_y(sy);
        const float* rp1 = row_y(sy + 1);
        const float* rp2 = row_y(sy + 2);
        float* dr = dst + static_cast<size_t>(dy + PYR_PAD) * dstride + PYR_PAD;
        int32_t x = 0;
#if defined(__AVX512F__)
        for (; x + 15 < dw; x += 16) {
            __m512 acc = _mm512_mul_ps(v6, _mm512_loadu_ps(r0 + x));
            acc = _mm512_fmadd_ps(v4, _mm512_add_ps(_mm512_loadu_ps(rm1 + x), _mm512_loadu_ps(rp1 + x)), acc);
            acc = _mm512_fmadd_ps(v1, _mm512_add_ps(_mm512_loadu_ps(rm2 + x), _mm512_loadu_ps(rp2 + x)), acc);
            _mm512_storeu_ps(dr + x, acc);
        }
#elif defined(__AVX2__)
        for (; x + 7 < dw; x += 8) {
            __m256 acc = _mm256_mul_ps(v6, _mm256_loadu_ps(r0 + x));
            acc = _mm256_fmadd_ps(v4, _mm256_add_ps(_mm256_loadu_ps(rm1 + x), _mm256_loadu_ps(rp1 + x)), acc);
            acc = _mm256_fmadd_ps(v1, _mm256_add_ps(_mm256_loadu_ps(rm2 + x), _mm256_loadu_ps(rp2 + x)), acc);
            _mm256_storeu_ps(dr + x, acc);
        }
#endif
        for (; x < dw; ++x) {
            dr[x] = (rm2[x] + 4.0f * rm1[x] + 6.0f * r0[x] + 4.0f * rp1[x] + rp2[x]) * (1.0f / 16.0f);
        }
    }
    fill_border(dst, dh, dw, dstride);
}

}  // namespace

extern "C" {

VSLAM_EXPORT void* lk_pyramid_create(const uint8_t* gray, int32_t h, int32_t w, int32_t max_level) {
    if (!gray || h < 8 || w < 8 || max_level < 0 || max_level > 6) return nullptr;
    auto* pyr = new LkPyramid();
    pyr->max_level = max_level;
    const int32_t n_lvls = max_level + 1;
    pyr->h.resize(n_lvls);
    pyr->w.resize(n_lvls);
    pyr->stride.resize(n_lvls);
    pyr->img.resize(n_lvls);

    for (int32_t lvl = 0; lvl < n_lvls; ++lvl) {
        const int32_t lh = (lvl == 0) ? h : ((pyr->h[lvl - 1] + 1) / 2);
        const int32_t lw = (lvl == 0) ? w : ((pyr->w[lvl - 1] + 1) / 2);
        const int32_t lstride = ((lw + 2 * PYR_PAD + 15) & ~15) + 16;
        pyr->h[lvl] = lh;
        pyr->w[lvl] = lw;
        pyr->stride[lvl] = lstride;
        pyr->img[lvl].resize(static_cast<size_t>(lh + 2 * PYR_PAD) * lstride);

        if (lvl == 0) {
            float* dst = pyr->img[0].data();
            for (int32_t y = 0; y < h; ++y) {
                const uint8_t* sr = gray + static_cast<size_t>(y) * w;
                float* dr = dst + static_cast<size_t>(y + PYR_PAD) * lstride + PYR_PAD;
                int32_t x = 0;
#if defined(__AVX512F__)
                for (; x + 15 < w; x += 16) {
                    __m512 vf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                        _mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + x))));
                    _mm512_storeu_ps(dr + x, vf);
                }
#elif defined(__AVX2__)
                for (; x + 7 < w; x += 8) {
                    __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(sr + x))));
                    _mm256_storeu_ps(dr + x, vf);
                }
#endif
                for (; x < w; ++x) dr[x] = static_cast<float>(sr[x]);
            }
            fill_border(dst, h, w, lstride);
        } else {
            pyr_down_simd(
                pyr->img[lvl - 1].data(), pyr->h[lvl - 1], pyr->w[lvl - 1], pyr->stride[lvl - 1],
                pyr->img[lvl].data(), lh, lw, lstride
            );
        }
    }
    return pyr;
}

VSLAM_EXPORT void lk_pyramid_destroy(void* handle) {
    delete static_cast<LkPyramid*>(handle);
}

VSLAM_EXPORT int32_t lk_track_pyramids_avx512(
    const void* prev_handle,
    const void* next_handle,
    const float* prev_pts,
    float* next_pts,
    uint8_t* status,
    int32_t n_pts,
    int32_t half_win,
    int32_t max_level,
    int32_t use_initial_flow,
    int32_t max_iter,
    float eps_sq,
    float min_eig_thresh
) {
    if (!prev_handle || !next_handle || n_pts <= 0 || half_win != 10) return 0;
    const auto* pyr0 = static_cast<const LkPyramid*>(prev_handle);
    const auto* pyr1 = static_cast<const LkPyramid*>(next_handle);
    const int32_t lvls = std::min(max_level, std::min(pyr0->max_level, pyr1->max_level));
    constexpr int32_t WIN = 21;
    constexpr int32_t HALF = 10;
    constexpr int32_t ROW_PAD = 32;

    alignas(64) float I_patch[WIN * ROW_PAD];
    alignas(64) float dIx_patch[WIN * ROW_PAD];
    alignas(64) float dIy_patch[WIN * ROW_PAD];

#if defined(__AVX512F__)
    const __mmask16 mask5 = 0x1Fu;
#endif
    const float inv_area = 1.0f / static_cast<float>(WIN * WIN);

    for (int32_t i = 0; i < n_pts; ++i) {
        const float px0 = prev_pts[2 * i + 0];
        const float py0 = prev_pts[2 * i + 1];
        float nx, ny;
        if (use_initial_flow) {
            nx = next_pts[2 * i + 0];
            ny = next_pts[2 * i + 1];
        } else {
            nx = px0;
            ny = py0;
        }
        if (!std::isfinite(px0) || !std::isfinite(py0) || !std::isfinite(nx) || !std::isfinite(ny)) {
            status[i] = 0;
            continue;
        }

        const float top_scale = 1.0f / static_cast<float>(1 << lvls);
        nx *= top_scale;
        ny *= top_scale;
        uint8_t ok = 1;

        for (int32_t lvl = lvls; lvl >= 0; --lvl) {
            const float scale = 1.0f / static_cast<float>(1 << lvl);
            const float px = px0 * scale;
            const float py = py0 * scale;

            if (lvl < lvls) {
                nx *= 2.0f;
                ny *= 2.0f;
            }

            const int32_t lh = pyr0->h[lvl];
            const int32_t lw = pyr0->w[lvl];
            const int32_t st0 = pyr0->stride[lvl];
            const int32_t st1 = pyr1->stride[lvl];
            const float* img0 = pyr0->img[lvl].data() + static_cast<size_t>(PYR_PAD) * st0 + PYR_PAD;
            const float* img1 = pyr1->img[lvl].data() + static_cast<size_t>(PYR_PAD) * st1 + PYR_PAD;

            const float tx0 = px - static_cast<float>(HALF);
            const float ty0 = py - static_cast<float>(HALF);
            const int32_t itx = static_cast<int32_t>(std::floor(tx0));
            const int32_t ity = static_cast<int32_t>(std::floor(ty0));
            if (itx < -HALF || ity < -HALF || itx + WIN > lw + HALF || ity + WIN > lh + HALF) {
                if (lvl == 0) ok = 0;
                continue;
            }

            const float ax = tx0 - static_cast<float>(itx);
            const float ay = ty0 - static_cast<float>(ity);
            const float sw00 = (1.0f - ax) * (1.0f - ay);
            const float sw01 = ax * (1.0f - ay);
            const float sw10 = (1.0f - ax) * ay;
            const float sw11 = ax * ay;

#if defined(__AVX512F__)
            const __m512 w00 = _mm512_set1_ps(sw00);
            const __m512 w01 = _mm512_set1_ps(sw01);
            const __m512 w10 = _mm512_set1_ps(sw10);
            const __m512 w11 = _mm512_set1_ps(sw11);
            const __m512 v05 = _mm512_set1_ps(0.5f);
            __m512 acc_xx = _mm512_setzero_ps();
            __m512 acc_xy = _mm512_setzero_ps();
            __m512 acc_yy = _mm512_setzero_ps();
#elif defined(__AVX2__)
            const __m256 w00 = _mm256_set1_ps(sw00);
            const __m256 w01 = _mm256_set1_ps(sw01);
            const __m256 w10 = _mm256_set1_ps(sw10);
            const __m256 w11 = _mm256_set1_ps(sw11);
            const __m256 v05 = _mm256_set1_ps(0.5f);
            __m256 acc_xx_v = _mm256_setzero_ps();
            __m256 acc_xy_v = _mm256_setzero_ps();
            __m256 acc_yy_v = _mm256_setzero_ps();
            float acc_xx_s = 0.0f, acc_xy_s = 0.0f, acc_yy_s = 0.0f;
#else
            float acc_xx_s = 0.0f, acc_xy_s = 0.0f, acc_yy_s = 0.0f;
#endif

            for (int32_t r = 0; r < WIN; ++r) {
                const float* row0 = img0 + static_cast<size_t>(ity + r) * st0 + itx;
                const float* row1 = row0 + st0;
                const float* rowm1 = row0 - st0;
                const float* row2 = row1 + st0;
                float* ip = I_patch + r * ROW_PAD;
                float* dxp = dIx_patch + r * ROW_PAD;
                float* dyp = dIy_patch + r * ROW_PAD;

#if defined(__AVX512F__)
                {
                    const __m512 p00 = _mm512_loadu_ps(row0);
                    const __m512 p01 = _mm512_loadu_ps(row0 + 1);
                    const __m512 p10 = _mm512_loadu_ps(row1);
                    const __m512 p11 = _mm512_loadu_ps(row1 + 1);
                    const __m512 val = _mm512_fmadd_ps(w00, p00,
                                       _mm512_fmadd_ps(w01, p01,
                                       _mm512_fmadd_ps(w10, p10, _mm512_mul_ps(w11, p11))));
                    _mm512_store_ps(ip, val);

                    const __m512 gx00 = _mm512_mul_ps(v05, _mm512_sub_ps(p01, _mm512_loadu_ps(row0 - 1)));
                    const __m512 gx01 = _mm512_mul_ps(v05, _mm512_sub_ps(_mm512_loadu_ps(row0 + 2), p00));
                    const __m512 gx10 = _mm512_mul_ps(v05, _mm512_sub_ps(p11, _mm512_loadu_ps(row1 - 1)));
                    const __m512 gx11 = _mm512_mul_ps(v05, _mm512_sub_ps(_mm512_loadu_ps(row1 + 2), p10));
                    const __m512 gx = _mm512_fmadd_ps(w00, gx00,
                                      _mm512_fmadd_ps(w01, gx01,
                                      _mm512_fmadd_ps(w10, gx10, _mm512_mul_ps(w11, gx11))));
                    _mm512_store_ps(dxp, gx);

                    const __m512 gy00 = _mm512_mul_ps(v05, _mm512_sub_ps(p10, _mm512_loadu_ps(rowm1)));
                    const __m512 gy01 = _mm512_mul_ps(v05, _mm512_sub_ps(p11, _mm512_loadu_ps(rowm1 + 1)));
                    const __m512 gy10 = _mm512_mul_ps(v05, _mm512_sub_ps(_mm512_loadu_ps(row2), p00));
                    const __m512 gy11 = _mm512_mul_ps(v05, _mm512_sub_ps(_mm512_loadu_ps(row2 + 1), p01));
                    const __m512 gy = _mm512_fmadd_ps(w00, gy00,
                                      _mm512_fmadd_ps(w01, gy01,
                                      _mm512_fmadd_ps(w10, gy10, _mm512_mul_ps(w11, gy11))));
                    _mm512_store_ps(dyp, gy);

                    acc_xx = _mm512_fmadd_ps(gx, gx, acc_xx);
                    acc_xy = _mm512_fmadd_ps(gx, gy, acc_xy);
                    acc_yy = _mm512_fmadd_ps(gy, gy, acc_yy);
                }
                {
                    const __m512 p00 = _mm512_maskz_loadu_ps(mask5, row0 + 16);
                    const __m512 p01 = _mm512_maskz_loadu_ps(mask5, row0 + 17);
                    const __m512 p10 = _mm512_maskz_loadu_ps(mask5, row1 + 16);
                    const __m512 p11 = _mm512_maskz_loadu_ps(mask5, row1 + 17);
                    const __m512 val = _mm512_fmadd_ps(w00, p00,
                                       _mm512_fmadd_ps(w01, p01,
                                       _mm512_fmadd_ps(w10, p10, _mm512_mul_ps(w11, p11))));
                    _mm512_mask_storeu_ps(ip + 16, mask5, val);

                    const __m512 gx00 = _mm512_mul_ps(v05, _mm512_sub_ps(p01, _mm512_maskz_loadu_ps(mask5, row0 + 15)));
                    const __m512 gx01 = _mm512_mul_ps(v05, _mm512_sub_ps(_mm512_maskz_loadu_ps(mask5, row0 + 18), p00));
                    const __m512 gx10 = _mm512_mul_ps(v05, _mm512_sub_ps(p11, _mm512_maskz_loadu_ps(mask5, row1 + 15)));
                    const __m512 gx11 = _mm512_mul_ps(v05, _mm512_sub_ps(_mm512_maskz_loadu_ps(mask5, row1 + 18), p10));
                    const __m512 gx = _mm512_maskz_add_ps(
                        mask5,
                        _mm512_fmadd_ps(w00, gx00, _mm512_mul_ps(w01, gx01)),
                        _mm512_fmadd_ps(w10, gx10, _mm512_mul_ps(w11, gx11))
                    );
                    _mm512_mask_storeu_ps(dxp + 16, mask5, gx);

                    const __m512 gy00 = _mm512_mul_ps(v05, _mm512_sub_ps(p10, _mm512_maskz_loadu_ps(mask5, rowm1 + 16)));
                    const __m512 gy01 = _mm512_mul_ps(v05, _mm512_sub_ps(p11, _mm512_maskz_loadu_ps(mask5, rowm1 + 17)));
                    const __m512 gy10 = _mm512_mul_ps(v05, _mm512_sub_ps(_mm512_maskz_loadu_ps(mask5, row2 + 16), p00));
                    const __m512 gy11 = _mm512_mul_ps(v05, _mm512_sub_ps(_mm512_maskz_loadu_ps(mask5, row2 + 17), p01));
                    const __m512 gy = _mm512_maskz_add_ps(
                        mask5,
                        _mm512_fmadd_ps(w00, gy00, _mm512_mul_ps(w01, gy01)),
                        _mm512_fmadd_ps(w10, gy10, _mm512_mul_ps(w11, gy11))
                    );
                    _mm512_mask_storeu_ps(dyp + 16, mask5, gy);

                    acc_xx = _mm512_fmadd_ps(gx, gx, acc_xx);
                    acc_xy = _mm512_fmadd_ps(gx, gy, acc_xy);
                    acc_yy = _mm512_fmadd_ps(gy, gy, acc_yy);
                }
#elif defined(__AVX2__)
                for (int32_t c = 0; c < 16; c += 8) {
                    const __m256 p00 = _mm256_loadu_ps(row0 + c);
                    const __m256 p01 = _mm256_loadu_ps(row0 + c + 1);
                    const __m256 p10 = _mm256_loadu_ps(row1 + c);
                    const __m256 p11 = _mm256_loadu_ps(row1 + c + 1);
                    const __m256 val = _mm256_fmadd_ps(w00, p00,
                                       _mm256_fmadd_ps(w01, p01,
                                       _mm256_fmadd_ps(w10, p10, _mm256_mul_ps(w11, p11))));
                    _mm256_store_ps(ip + c, val);

                    const __m256 gx00 = _mm256_mul_ps(v05, _mm256_sub_ps(p01, _mm256_loadu_ps(row0 + c - 1)));
                    const __m256 gx01 = _mm256_mul_ps(v05, _mm256_sub_ps(_mm256_loadu_ps(row0 + c + 2), p00));
                    const __m256 gx10 = _mm256_mul_ps(v05, _mm256_sub_ps(p11, _mm256_loadu_ps(row1 + c - 1)));
                    const __m256 gx11 = _mm256_mul_ps(v05, _mm256_sub_ps(_mm256_loadu_ps(row1 + c + 2), p10));
                    const __m256 gx = _mm256_fmadd_ps(w00, gx00,
                                      _mm256_fmadd_ps(w01, gx01,
                                      _mm256_fmadd_ps(w10, gx10, _mm256_mul_ps(w11, gx11))));
                    _mm256_store_ps(dxp + c, gx);

                    const __m256 gy00 = _mm256_mul_ps(v05, _mm256_sub_ps(p10, _mm256_loadu_ps(rowm1 + c)));
                    const __m256 gy01 = _mm256_mul_ps(v05, _mm256_sub_ps(p11, _mm256_loadu_ps(rowm1 + c + 1)));
                    const __m256 gy10 = _mm256_mul_ps(v05, _mm256_sub_ps(_mm256_loadu_ps(row2 + c), p00));
                    const __m256 gy11 = _mm256_mul_ps(v05, _mm256_sub_ps(_mm256_loadu_ps(row2 + c + 1), p01));
                    const __m256 gy = _mm256_fmadd_ps(w00, gy00,
                                      _mm256_fmadd_ps(w01, gy01,
                                      _mm256_fmadd_ps(w10, gy10, _mm256_mul_ps(w11, gy11))));
                    _mm256_store_ps(dyp + c, gy);

                    acc_xx_v = _mm256_fmadd_ps(gx, gx, acc_xx_v);
                    acc_xy_v = _mm256_fmadd_ps(gx, gy, acc_xy_v);
                    acc_yy_v = _mm256_fmadd_ps(gy, gy, acc_yy_v);
                }
                for (int32_t c = 16; c < WIN; ++c) {
                    const float p00 = row0[c], p01 = row0[c + 1], p10 = row1[c], p11 = row1[c + 1];
                    ip[c] = sw00 * p00 + sw01 * p01 + sw10 * p10 + sw11 * p11;
                    const float gx00 = 0.5f * (p01 - row0[c - 1]);
                    const float gx01 = 0.5f * (row0[c + 2] - p00);
                    const float gx10 = 0.5f * (p11 - row1[c - 1]);
                    const float gx11 = 0.5f * (row1[c + 2] - p10);
                    const float gx = sw00 * gx00 + sw01 * gx01 + sw10 * gx10 + sw11 * gx11;
                    dxp[c] = gx;
                    const float gy00 = 0.5f * (p10 - rowm1[c]);
                    const float gy01 = 0.5f * (p11 - rowm1[c + 1]);
                    const float gy10 = 0.5f * (row2[c] - p00);
                    const float gy11 = 0.5f * (row2[c + 1] - p01);
                    const float gy = sw00 * gy00 + sw01 * gy01 + sw10 * gy10 + sw11 * gy11;
                    dyp[c] = gy;
                    acc_xx_s += gx * gx;
                    acc_xy_s += gx * gy;
                    acc_yy_s += gy * gy;
                }
#else
                for (int32_t c = 0; c < WIN; ++c) {
                    const float p00 = row0[c], p01 = row0[c + 1], p10 = row1[c], p11 = row1[c + 1];
                    ip[c] = sw00 * p00 + sw01 * p01 + sw10 * p10 + sw11 * p11;
                    const float gx00 = 0.5f * (p01 - row0[c - 1]);
                    const float gx01 = 0.5f * (row0[c + 2] - p00);
                    const float gx10 = 0.5f * (p11 - row1[c - 1]);
                    const float gx11 = 0.5f * (row1[c + 2] - p10);
                    const float gx = sw00 * gx00 + sw01 * gx01 + sw10 * gx10 + sw11 * gx11;
                    dxp[c] = gx;
                    const float gy00 = 0.5f * (p10 - rowm1[c]);
                    const float gy01 = 0.5f * (p11 - rowm1[c + 1]);
                    const float gy10 = 0.5f * (row2[c] - p00);
                    const float gy11 = 0.5f * (row2[c + 1] - p01);
                    const float gy = sw00 * gy00 + sw01 * gy01 + sw10 * gy10 + sw11 * gy11;
                    dyp[c] = gy;
                    acc_xx_s += gx * gx;
                    acc_xy_s += gx * gy;
                    acc_yy_s += gy * gy;
                }
#endif
            }

#if defined(__AVX512F__)
            const double A11 = static_cast<double>(_mm512_reduce_add_ps(acc_xx));
            const double A12 = static_cast<double>(_mm512_reduce_add_ps(acc_xy));
            const double A22 = static_cast<double>(_mm512_reduce_add_ps(acc_yy));
#elif defined(__AVX2__)
            const double A11 = static_cast<double>(hsum256_ps(acc_xx_v) + acc_xx_s);
            const double A12 = static_cast<double>(hsum256_ps(acc_xy_v) + acc_xy_s);
            const double A22 = static_cast<double>(hsum256_ps(acc_yy_v) + acc_yy_s);
#else
            const double A11 = static_cast<double>(acc_xx_s);
            const double A12 = static_cast<double>(acc_xy_s);
            const double A22 = static_cast<double>(acc_yy_s);
#endif
            const double D = A11 * A22 - A12 * A12;
            const double min_eig = 0.5 * ((A11 + A22) - std::sqrt((A11 - A22) * (A11 - A22) + 4.0 * A12 * A12)) * inv_area;
            if (min_eig < static_cast<double>(min_eig_thresh) || D < 1e-7) {
                if (lvl == 0) ok = 0;
                continue;
            }
            const double invD = 1.0 / D;

            float prev_dx = 0.0f, prev_dy = 0.0f;
            for (int32_t it = 0; it < max_iter; ++it) {
                const float jx0 = nx - static_cast<float>(HALF);
                const float jy0 = ny - static_cast<float>(HALF);
                const int32_t ijx = static_cast<int32_t>(std::floor(jx0));
                const int32_t ijy = static_cast<int32_t>(std::floor(jy0));
                if (ijx < -HALF || ijy < -HALF || ijx + WIN > lw + HALF || ijy + WIN > lh + HALF) {
                    if (lvl == 0) ok = 0;
                    break;
                }
                const float bx = jx0 - static_cast<float>(ijx);
                const float by = jy0 - static_cast<float>(ijy);
                const float sjw00 = (1.0f - bx) * (1.0f - by);
                const float sjw01 = bx * (1.0f - by);
                const float sjw10 = (1.0f - bx) * by;
                const float sjw11 = bx * by;

#if defined(__AVX512F__)
                const __m512 jw00 = _mm512_set1_ps(sjw00);
                const __m512 jw01 = _mm512_set1_ps(sjw01);
                const __m512 jw10 = _mm512_set1_ps(sjw10);
                const __m512 jw11 = _mm512_set1_ps(sjw11);
                __m512 b1 = _mm512_setzero_ps();
                __m512 b2 = _mm512_setzero_ps();
#elif defined(__AVX2__)
                const __m256 jw00 = _mm256_set1_ps(sjw00);
                const __m256 jw01 = _mm256_set1_ps(sjw01);
                const __m256 jw10 = _mm256_set1_ps(sjw10);
                const __m256 jw11 = _mm256_set1_ps(sjw11);
                __m256 b1_v = _mm256_setzero_ps();
                __m256 b2_v = _mm256_setzero_ps();
                float b1_s = 0.0f, b2_s = 0.0f;
#else
                float b1_s = 0.0f, b2_s = 0.0f;
#endif

                for (int32_t r = 0; r < WIN; ++r) {
                    const float* jrow0 = img1 + static_cast<size_t>(ijy + r) * st1 + ijx;
                    const float* jrow1 = jrow0 + st1;
                    const float* ip = I_patch + r * ROW_PAD;
                    const float* dxp = dIx_patch + r * ROW_PAD;
                    const float* dyp = dIy_patch + r * ROW_PAD;

#if defined(__AVX512F__)
                    {
                        const __m512 jval = _mm512_fmadd_ps(jw00, _mm512_loadu_ps(jrow0),
                                            _mm512_fmadd_ps(jw01, _mm512_loadu_ps(jrow0 + 1),
                                            _mm512_fmadd_ps(jw10, _mm512_loadu_ps(jrow1),
                                            _mm512_mul_ps(jw11, _mm512_loadu_ps(jrow1 + 1)))));
                        const __m512 diff = _mm512_sub_ps(_mm512_load_ps(ip), jval);
                        b1 = _mm512_fmadd_ps(diff, _mm512_load_ps(dxp), b1);
                        b2 = _mm512_fmadd_ps(diff, _mm512_load_ps(dyp), b2);
                    }
                    {
                        const __m512 jval = _mm512_fmadd_ps(jw00, _mm512_maskz_loadu_ps(mask5, jrow0 + 16),
                                            _mm512_fmadd_ps(jw01, _mm512_maskz_loadu_ps(mask5, jrow0 + 17),
                                            _mm512_fmadd_ps(jw10, _mm512_maskz_loadu_ps(mask5, jrow1 + 16),
                                            _mm512_mul_ps(jw11, _mm512_maskz_loadu_ps(mask5, jrow1 + 17)))));
                        const __m512 diff = _mm512_maskz_sub_ps(mask5, _mm512_maskz_loadu_ps(mask5, ip + 16), jval);
                        b1 = _mm512_fmadd_ps(diff, _mm512_maskz_loadu_ps(mask5, dxp + 16), b1);
                        b2 = _mm512_fmadd_ps(diff, _mm512_maskz_loadu_ps(mask5, dyp + 16), b2);
                    }
#elif defined(__AVX2__)
                    for (int32_t c = 0; c < 16; c += 8) {
                        const __m256 jval = _mm256_fmadd_ps(jw00, _mm256_loadu_ps(jrow0 + c),
                                            _mm256_fmadd_ps(jw01, _mm256_loadu_ps(jrow0 + c + 1),
                                            _mm256_fmadd_ps(jw10, _mm256_loadu_ps(jrow1 + c),
                                            _mm256_mul_ps(jw11, _mm256_loadu_ps(jrow1 + c + 1)))));
                        const __m256 diff = _mm256_sub_ps(_mm256_load_ps(ip + c), jval);
                        b1_v = _mm256_fmadd_ps(diff, _mm256_load_ps(dxp + c), b1_v);
                        b2_v = _mm256_fmadd_ps(diff, _mm256_load_ps(dyp + c), b2_v);
                    }
                    for (int32_t c = 16; c < WIN; ++c) {
                        const float jval = sjw00 * jrow0[c] + sjw01 * jrow0[c + 1]
                                         + sjw10 * jrow1[c] + sjw11 * jrow1[c + 1];
                        const float diff = ip[c] - jval;
                        b1_s += diff * dxp[c];
                        b2_s += diff * dyp[c];
                    }
#else
                    for (int32_t c = 0; c < WIN; ++c) {
                        const float jval = sjw00 * jrow0[c] + sjw01 * jrow0[c + 1]
                                         + sjw10 * jrow1[c] + sjw11 * jrow1[c + 1];
                        const float diff = ip[c] - jval;
                        b1_s += diff * dxp[c];
                        b2_s += diff * dyp[c];
                    }
#endif
                }

#if defined(__AVX512F__)
                const double B1 = static_cast<double>(_mm512_reduce_add_ps(b1));
                const double B2 = static_cast<double>(_mm512_reduce_add_ps(b2));
#elif defined(__AVX2__)
                const double B1 = static_cast<double>(hsum256_ps(b1_v) + b1_s);
                const double B2 = static_cast<double>(hsum256_ps(b2_v) + b2_s);
#else
                const double B1 = static_cast<double>(b1_s);
                const double B2 = static_cast<double>(b2_s);
#endif
                const float dx = static_cast<float>((A22 * B1 - A12 * B2) * invD);
                const float dy = static_cast<float>((A11 * B2 - A12 * B1) * invD);
                nx += dx;
                ny += dy;
                if (dx * dx + dy * dy <= eps_sq) break;
                if (it > 0 && std::abs(dx + prev_dx) < 0.01f && std::abs(dy + prev_dy) < 0.01f) {
                    nx -= 0.5f * dx;
                    ny -= 0.5f * dy;
                    break;
                }
                prev_dx = dx;
                prev_dy = dy;
            }
        }

        next_pts[2 * i + 0] = nx;
        next_pts[2 * i + 1] = ny;
        status[i] = (ok && nx >= 0.0f && ny >= 0.0f && nx < static_cast<float>(pyr1->w[0]) && ny < static_cast<float>(pyr1->h[0])) ? 1 : 0;
    }
    return 1;
}

}  // extern "C"
