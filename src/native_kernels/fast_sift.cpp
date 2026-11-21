#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr float PI_F = 3.14159265358979323846f;
constexpr float RAD2DEG = 180.0f / PI_F;
constexpr float DEG2RAD = PI_F / 180.0f;

// Fast accurate AVX-512 atan2 in degrees [0, 360), max error < 0.05 deg.
inline __m512 fast_atan2_deg_avx512(__m512 y, __m512 x) {
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);
    const __m512 abs_x = _mm512_andnot_ps(sign_mask, x);
    const __m512 abs_y = _mm512_andnot_ps(sign_mask, y);
    const __m512 max_v = _mm512_max_ps(abs_x, abs_y);
    const __m512 min_v = _mm512_min_ps(abs_x, abs_y);
    const __m512 eps = _mm512_set1_ps(1e-12f);
    const __m512 a = _mm512_div_ps(min_v, _mm512_add_ps(max_v, eps));
    const __m512 s = _mm512_mul_ps(a, a);

    __m512 r = _mm512_set1_ps(-0.0464964749f);
    r = _mm512_fmadd_ps(r, s, _mm512_set1_ps(0.15931422f));
    r = _mm512_fmadd_ps(r, s, _mm512_set1_ps(-0.327622764f));
    r = _mm512_mul_ps(r, s);
    r = _mm512_fmadd_ps(r, a, a);

    const __mmask16 swap_m = _mm512_cmp_ps_mask(abs_y, abs_x, _CMP_GT_OQ);
    r = _mm512_mask_sub_ps(r, swap_m, _mm512_set1_ps(0.5f * PI_F), r);
    const __mmask16 xneg = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ);
    r = _mm512_mask_sub_ps(r, xneg, _mm512_set1_ps(PI_F), r);
    __m512 deg = _mm512_mul_ps(r, _mm512_set1_ps(RAD2DEG));
    const __mmask16 yneg = _mm512_cmp_ps_mask(y, _mm512_setzero_ps(), _CMP_LT_OQ);
    deg = _mm512_mask_sub_ps(deg, yneg, _mm512_set1_ps(360.0f), deg);
    const __mmask16 ge360 = _mm512_cmp_ps_mask(deg, _mm512_set1_ps(360.0f), _CMP_GE_OQ);
    deg = _mm512_mask_sub_ps(deg, ge360, deg, _mm512_set1_ps(360.0f));
    return deg;
}

void make_gaussian_kernel(float sigma, float half_w[25], int32_t& rad) {
    // 2.6 * sigma captures >99.1% of Gaussian mass while trimming outer near-zero taps
    rad = static_cast<int32_t>(std::ceil(2.6f * sigma));
    if (rad < 1) rad = 1;
    if (rad > 20) rad = 20;
    const float inv2s2 = -0.5f / (sigma * sigma);
    double sum = 1.0;
    half_w[0] = 1.0f;
    for (int32_t k = 1; k <= rad; ++k) {
        float val = std::exp(static_cast<float>(k * k) * inv2s2);
        half_w[k] = val;
        sum += 2.0 * static_cast<double>(val);
    }
    const float inv_sum = static_cast<float>(1.0 / sum);
    for (int32_t k = 0; k <= rad; ++k) half_w[k] *= inv_sum;
}

// Compile-time unrolled horizontal AVX-512 Gaussian blur with all tap weights
// kept live in ZMM registers (zero per-iteration broadcasts or loop counters).
template <int32_t RAD>
inline void blur_row_h_fixed_avx512(const float* __restrict srow, float* __restrict trow,
                                    int32_t w, const float* __restrict hw) {
    const int32_t x_left_end = std::min(RAD, w);
    for (int32_t x = 0; x < x_left_end; ++x) {
        float acc = hw[0] * srow[x];
        for (int32_t k = 1; k <= RAD; ++k) {
            int32_t xl = (x - k < 0) ? 0 : (x - k);
            int32_t xr = (x + k >= w) ? (w - 1) : (x + k);
            acc += hw[k] * (srow[xl] + srow[xr]);
        }
        trow[x] = acc;
    }

    __m512 vw[RAD + 1];
    for (int32_t k = 0; k <= RAD; ++k) vw[k] = _mm512_set1_ps(hw[k]);

    int32_t x = RAD;
    const int32_t x_mid_end = w - RAD;
    for (; x + 31 < x_mid_end; x += 32) {
        __m512 a0 = _mm512_mul_ps(vw[0], _mm512_loadu_ps(srow + x));
        __m512 a1 = _mm512_mul_ps(vw[0], _mm512_loadu_ps(srow + x + 16));
        #pragma GCC unroll 12
        for (int32_t k = 1; k <= RAD; ++k) {
            a0 = _mm512_fmadd_ps(vw[k], _mm512_add_ps(_mm512_loadu_ps(srow + x - k),
                                                      _mm512_loadu_ps(srow + x + k)), a0);
            a1 = _mm512_fmadd_ps(vw[k], _mm512_add_ps(_mm512_loadu_ps(srow + x + 16 - k),
                                                      _mm512_loadu_ps(srow + x + 16 + k)), a1);
        }
        _mm512_storeu_ps(trow + x, a0);
        _mm512_storeu_ps(trow + x + 16, a1);
    }
    for (; x + 15 < x_mid_end; x += 16) {
        __m512 a0 = _mm512_mul_ps(vw[0], _mm512_loadu_ps(srow + x));
        #pragma GCC unroll 12
        for (int32_t k = 1; k <= RAD; ++k) {
            a0 = _mm512_fmadd_ps(vw[k], _mm512_add_ps(_mm512_loadu_ps(srow + x - k),
                                                      _mm512_loadu_ps(srow + x + k)), a0);
        }
        _mm512_storeu_ps(trow + x, a0);
    }
    for (; x < x_mid_end; ++x) {
        float acc = hw[0] * srow[x];
        for (int32_t k = 1; k <= RAD; ++k) acc += hw[k] * (srow[x - k] + srow[x + k]);
        trow[x] = acc;
    }
    for (; x < w; ++x) {
        float acc = hw[0] * srow[x];
        for (int32_t k = 1; k <= RAD; ++k) {
            int32_t xl = (x - k < 0) ? 0 : (x - k);
            int32_t xr = (x + k >= w) ? (w - 1) : (x + k);
            acc += hw[k] * (srow[xl] + srow[xr]);
        }
        trow[x] = acc;
    }
}

inline void blur_row_h_generic_avx512(const float* __restrict srow, float* __restrict trow,
                                      int32_t w, const float* __restrict hw, int32_t rad) {
    const int32_t x_left_end = std::min(rad, w);
    for (int32_t x = 0; x < x_left_end; ++x) {
        float acc = hw[0] * srow[x];
        for (int32_t k = 1; k <= rad; ++k) {
            int32_t xl = (x - k < 0) ? 0 : (x - k);
            int32_t xr = (x + k >= w) ? (w - 1) : (x + k);
            acc += hw[k] * (srow[xl] + srow[xr]);
        }
        trow[x] = acc;
    }
    int32_t x = rad;
    const int32_t x_mid_end = w - rad;
    const __m512 w0 = _mm512_set1_ps(hw[0]);
    for (; x + 31 < x_mid_end; x += 32) {
        __m512 a0 = _mm512_mul_ps(w0, _mm512_loadu_ps(srow + x));
        __m512 a1 = _mm512_mul_ps(w0, _mm512_loadu_ps(srow + x + 16));
        for (int32_t k = 1; k <= rad; ++k) {
            __m512 wk = _mm512_set1_ps(hw[k]);
            a0 = _mm512_fmadd_ps(wk, _mm512_add_ps(_mm512_loadu_ps(srow + x - k), _mm512_loadu_ps(srow + x + k)), a0);
            a1 = _mm512_fmadd_ps(wk, _mm512_add_ps(_mm512_loadu_ps(srow + x + 16 - k), _mm512_loadu_ps(srow + x + 16 + k)), a1);
        }
        _mm512_storeu_ps(trow + x, a0);
        _mm512_storeu_ps(trow + x + 16, a1);
    }
    for (; x + 15 < x_mid_end; x += 16) {
        __m512 a0 = _mm512_mul_ps(w0, _mm512_loadu_ps(srow + x));
        for (int32_t k = 1; k <= rad; ++k) {
            a0 = _mm512_fmadd_ps(_mm512_set1_ps(hw[k]),
                                 _mm512_add_ps(_mm512_loadu_ps(srow + x - k), _mm512_loadu_ps(srow + x + k)), a0);
        }
        _mm512_storeu_ps(trow + x, a0);
    }
    for (; x < x_mid_end; ++x) {
        float acc = hw[0] * srow[x];
        for (int32_t k = 1; k <= rad; ++k) acc += hw[k] * (srow[x - k] + srow[x + k]);
        trow[x] = acc;
    }
    for (; x < w; ++x) {
        float acc = hw[0] * srow[x];
        for (int32_t k = 1; k <= rad; ++k) {
            int32_t xl = (x - k < 0) ? 0 : (x - k);
            int32_t xr = (x + k >= w) ? (w - 1) : (x + k);
            acc += hw[k] * (srow[xl] + srow[xr]);
        }
        trow[x] = acc;
    }
}

// L2-fused ring-buffer separable 2D Gaussian blur with fused DoG subtraction!
// - If `dst != nullptr`, writes the blurred Gaussian image to `dst`.
// - If `dog_dst != nullptr`, simultaneously writes `blurred - src` to `dog_dst`
//   while `src` is still warm in L2 cache, eliminating the separate DoG DRAM pass.
template <int32_t RAD, bool WRITE_G, bool WRITE_DOG>
void gaussian_blur_ring_fixed_avx512(
    const float* __restrict src,
    float* __restrict dst,
    float* __restrict dog_dst,
    float* __restrict ring_buf,
    int32_t h,
    int32_t w,
    const float* __restrict hw
) {
    constexpr int32_t RING_SZ = 2 * RAD + 2;
    auto ring_ptr = [&](int32_t ry) -> float* {
        int32_t clamped = (ry < 0) ? 0 : (ry >= h ? h - 1 : ry);
        return ring_buf + static_cast<size_t>(clamped % RING_SZ) * w;
    };

    int32_t h_done = -1;
    const int32_t init_end = std::min(RAD, h - 1);
    for (int32_t ry = 0; ry <= init_end; ++ry) {
        blur_row_h_fixed_avx512<RAD>(src + static_cast<size_t>(ry) * w, ring_ptr(ry), w, hw);
        h_done = ry;
    }

    __m512 vw[RAD + 1];
    for (int32_t k = 0; k <= RAD; ++k) vw[k] = _mm512_set1_ps(hw[k]);

    for (int32_t y = 0; y < h; ++y) {
        const int32_t need_ry = std::min(y + RAD, h - 1);
        while (h_done < need_ry) {
            ++h_done;
            blur_row_h_fixed_avx512<RAD>(src + static_cast<size_t>(h_done) * w, ring_ptr(h_done), w, hw);
        }

        const float* r0 = ring_ptr(y);
        const float* r_up[RAD + 1];
        const float* r_dn[RAD + 1];
        #pragma GCC unroll 12
        for (int32_t k = 1; k <= RAD; ++k) {
            r_up[k] = ring_ptr(y - k);
            r_dn[k] = ring_ptr(y + k);
        }
        const size_t row_off = static_cast<size_t>(y) * w;
        const float* srow = WRITE_DOG ? (src + row_off) : nullptr;
        float* drow = WRITE_G ? (dst + row_off) : nullptr;
        float* dogrow = WRITE_DOG ? (dog_dst + row_off) : nullptr;

        int32_t x = 0;
        for (; x + 31 < w; x += 32) {
            __m512 a0 = _mm512_mul_ps(vw[0], _mm512_loadu_ps(r0 + x));
            __m512 a1 = _mm512_mul_ps(vw[0], _mm512_loadu_ps(r0 + x + 16));
            #pragma GCC unroll 12
            for (int32_t k = 1; k <= RAD; ++k) {
                a0 = _mm512_fmadd_ps(vw[k], _mm512_add_ps(_mm512_loadu_ps(r_up[k] + x),
                                                          _mm512_loadu_ps(r_dn[k] + x)), a0);
                a1 = _mm512_fmadd_ps(vw[k], _mm512_add_ps(_mm512_loadu_ps(r_up[k] + x + 16),
                                                          _mm512_loadu_ps(r_dn[k] + x + 16)), a1);
            }
            if constexpr (WRITE_G) {
                _mm512_storeu_ps(drow + x, a0);
                _mm512_storeu_ps(drow + x + 16, a1);
            }
            if constexpr (WRITE_DOG) {
                _mm512_storeu_ps(dogrow + x, _mm512_sub_ps(a0, _mm512_loadu_ps(srow + x)));
                _mm512_storeu_ps(dogrow + x + 16, _mm512_sub_ps(a1, _mm512_loadu_ps(srow + x + 16)));
            }
        }
        for (; x + 15 < w; x += 16) {
            __m512 a0 = _mm512_mul_ps(vw[0], _mm512_loadu_ps(r0 + x));
            #pragma GCC unroll 12
            for (int32_t k = 1; k <= RAD; ++k) {
                a0 = _mm512_fmadd_ps(vw[k], _mm512_add_ps(_mm512_loadu_ps(r_up[k] + x),
                                                          _mm512_loadu_ps(r_dn[k] + x)), a0);
            }
            if constexpr (WRITE_G) _mm512_storeu_ps(drow + x, a0);
            if constexpr (WRITE_DOG) _mm512_storeu_ps(dogrow + x, _mm512_sub_ps(a0, _mm512_loadu_ps(srow + x)));
        }
        for (; x < w; ++x) {
            float acc = hw[0] * r0[x];
            for (int32_t k = 1; k <= RAD; ++k) acc += hw[k] * (r_up[k][x] + r_dn[k][x]);
            if constexpr (WRITE_G) drow[x] = acc;
            if constexpr (WRITE_DOG) dogrow[x] = acc - srow[x];
        }
    }
}

void gaussian_blur_ring_avx512(
    const float* __restrict src,
    float* __restrict dst,
    float* __restrict dog_dst,
    float* __restrict ring_buf,
    int32_t h,
    int32_t w,
    float sigma
) {
    float hw[25];
    int32_t rad = 0;
    make_gaussian_kernel(sigma, hw, rad);

    auto dispatch = [&](auto write_g_tag, auto write_dog_tag) {
        constexpr bool WG = decltype(write_g_tag)::value;
        constexpr bool WD = decltype(write_dog_tag)::value;
        switch (rad) {
            case 3: gaussian_blur_ring_fixed_avx512<3, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 4: gaussian_blur_ring_fixed_avx512<4, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 5: gaussian_blur_ring_fixed_avx512<5, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 6: gaussian_blur_ring_fixed_avx512<6, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 7: gaussian_blur_ring_fixed_avx512<7, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 8: gaussian_blur_ring_fixed_avx512<8, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 9: gaussian_blur_ring_fixed_avx512<9, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 10: gaussian_blur_ring_fixed_avx512<10, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            default: break;
        }
        const int32_t ring_sz = 2 * rad + 2;
        auto ring_ptr = [&](int32_t ry) -> float* {
            int32_t clamped = (ry < 0) ? 0 : (ry >= h ? h - 1 : ry);
            return ring_buf + static_cast<size_t>(clamped % ring_sz) * w;
        };
        int32_t h_done = -1;
        const int32_t init_end = std::min(rad, h - 1);
        for (int32_t ry = 0; ry <= init_end; ++ry) {
            blur_row_h_generic_avx512(src + static_cast<size_t>(ry) * w, ring_ptr(ry), w, hw, rad);
            h_done = ry;
        }
        const __m512 w0 = _mm512_set1_ps(hw[0]);
        for (int32_t y = 0; y < h; ++y) {
            const int32_t need_ry = std::min(y + rad, h - 1);
            while (h_done < need_ry) {
                ++h_done;
                blur_row_h_generic_avx512(src + static_cast<size_t>(h_done) * w, ring_ptr(h_done), w, hw, rad);
            }
            const float* r0 = ring_ptr(y);
            const float* r_up[25];
            const float* r_dn[25];
            for (int32_t k = 1; k <= rad; ++k) {
                r_up[k] = ring_ptr(y - k);
                r_dn[k] = ring_ptr(y + k);
            }
            const size_t row_off = static_cast<size_t>(y) * w;
            const float* srow = WD ? (src + row_off) : nullptr;
            float* drow = WG ? (dst + row_off) : nullptr;
            float* dogrow = WD ? (dog_dst + row_off) : nullptr;
            int32_t x = 0;
            for (; x + 15 < w; x += 16) {
                __m512 a0 = _mm512_mul_ps(w0, _mm512_loadu_ps(r0 + x));
                for (int32_t k = 1; k <= rad; ++k) {
                    a0 = _mm512_fmadd_ps(_mm512_set1_ps(hw[k]),
                                         _mm512_add_ps(_mm512_loadu_ps(r_up[k] + x), _mm512_loadu_ps(r_dn[k] + x)), a0);
                }
                if constexpr (WG) _mm512_storeu_ps(drow + x, a0);
                if constexpr (WD) _mm512_storeu_ps(dogrow + x, _mm512_sub_ps(a0, _mm512_loadu_ps(srow + x)));
            }
            for (; x < w; ++x) {
                float acc = hw[0] * r0[x];
                for (int32_t k = 1; k <= rad; ++k) acc += hw[k] * (r_up[k][x] + r_dn[k][x]);
                if constexpr (WG) drow[x] = acc;
                if constexpr (WD) dogrow[x] = acc - srow[x];
            }
        }
    };

    if (dst != nullptr && dog_dst != nullptr) {
        dispatch(std::true_type{}, std::true_type{});
    } else if (dst != nullptr) {
        dispatch(std::true_type{}, std::false_type{});
    } else if (dog_dst != nullptr) {
        dispatch(std::false_type{}, std::true_type{});
    }
}

void upsample_2x_avx512(const uint8_t* __restrict src, int32_t h, int32_t w,
                        float* __restrict dst, float* __restrict row_tmp) {
    const int32_t dh = h * 2;
    const int32_t dw = w * 2;
    const __m512 v075 = _mm512_set1_ps(0.75f);
    const __m512 v025 = _mm512_set1_ps(0.25f);
    const __m512i idx_lo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    const __m512i idx_hi = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* sr = src + static_cast<size_t>(y) * w;
        float* tr = row_tmp + static_cast<size_t>(y) * dw;
        tr[0] = static_cast<float>(sr[0]);
        tr[1] = 0.75f * sr[0] + 0.25f * sr[std::min(1, w - 1)];
        int32_t x = 1;
        for (; x + 16 < w; x += 16) {
            const __m512 vc = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + x))));
            const __m512 vl = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + x - 1))));
            const __m512 vr = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sr + x + 1))));
            const __m512 c075 = _mm512_mul_ps(v075, vc);
            const __m512 even = _mm512_fmadd_ps(v025, vl, c075);
            const __m512 odd  = _mm512_fmadd_ps(v025, vr, c075);
            _mm512_storeu_ps(tr + 2 * x,      _mm512_permutex2var_ps(even, idx_lo, odd));
            _mm512_storeu_ps(tr + 2 * x + 16, _mm512_permutex2var_ps(even, idx_hi, odd));
        }
        for (; x < w - 1; ++x) {
            const float c = static_cast<float>(sr[x]);
            tr[2 * x]     = 0.75f * c + 0.25f * static_cast<float>(sr[x - 1]);
            tr[2 * x + 1] = 0.75f * c + 0.25f * static_cast<float>(sr[x + 1]);
        }
        if (w > 1) {
            tr[2 * (w - 1)]     = 0.75f * sr[w - 1] + 0.25f * sr[w - 2];
            tr[2 * (w - 1) + 1] = static_cast<float>(sr[w - 1]);
        }
    }
    for (int32_t dy = 0; dy < dh; ++dy) {
        const int32_t sy = dy >> 1;
        const int32_t sy_other = (dy & 1) ? std::min(sy + 1, h - 1) : std::max(sy - 1, 0);
        const float* r_main = row_tmp + static_cast<size_t>(sy) * dw;
        const float* r_oth  = row_tmp + static_cast<size_t>(sy_other) * dw;
        float* dr = dst + static_cast<size_t>(dy) * dw;
        int32_t x = 0;
        for (; x + 31 < dw; x += 32) {
            __m512 v0 = _mm512_fmadd_ps(v075, _mm512_loadu_ps(r_main + x),
                                        _mm512_mul_ps(v025, _mm512_loadu_ps(r_oth + x)));
            __m512 v1 = _mm512_fmadd_ps(v075, _mm512_loadu_ps(r_main + x + 16),
                                        _mm512_mul_ps(v025, _mm512_loadu_ps(r_oth + x + 16)));
            _mm512_storeu_ps(dr + x, v0);
            _mm512_storeu_ps(dr + x + 16, v1);
        }
        for (; x + 15 < dw; x += 16) {
            __m512 v = _mm512_fmadd_ps(v075, _mm512_loadu_ps(r_main + x),
                                       _mm512_mul_ps(v025, _mm512_loadu_ps(r_oth + x)));
            _mm512_storeu_ps(dr + x, v);
        }
        for (; x < dw; ++x) dr[x] = 0.75f * r_main[x] + 0.25f * r_oth[x];
    }
}

void downsample_2x(const float* __restrict src, int32_t sh, int32_t sw,
                   float* __restrict dst, int32_t dh, int32_t dw) {
    (void)sh;
    const __m512i even_idx = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    for (int32_t y = 0; y < dh; ++y) {
        const float* sr = src + static_cast<size_t>(y * 2) * sw;
        float* dr = dst + static_cast<size_t>(y) * dw;
        int32_t x = 0;
        for (; x + 15 < dw; x += 16) {
            __m512 a = _mm512_loadu_ps(sr + 2 * x);
            __m512 b = _mm512_loadu_ps(sr + 2 * x + 16);
            _mm512_storeu_ps(dr + x, _mm512_permutex2var_ps(a, even_idx, b));
        }
        for (; x < dw; ++x) dr[x] = sr[x * 2];
    }
}

struct RawExtremum {
    float x, y, response, scl_octv;
    int32_t ri, ci, octave_idx, layer, pyr_o;
};

struct SiftKeypoint {
    float x, y, angle, response, scl_octv;
    int32_t octave_idx, layer, pyr_o;
};

inline bool solve3x3(const float A[3][3], const float b[3], float sol[3]) {
    const double a00 = A[0][0], a01 = A[0][1], a02 = A[0][2];
    const double a10 = A[1][0], a11 = A[1][1], a12 = A[1][2];
    const double a20 = A[2][0], a21 = A[2][1], a22 = A[2][2];
    const double c00 = a11 * a22 - a12 * a21;
    const double c01 = a12 * a20 - a10 * a22;
    const double c02 = a10 * a21 - a11 * a20;
    const double det = a00 * c00 + a01 * c01 + a02 * c02;
    if (std::abs(det) < 1e-12) return false;
    const double inv_det = 1.0 / det;
    const double c10 = a02 * a21 - a01 * a22;
    const double c11 = a00 * a22 - a02 * a20;
    const double c12 = a01 * a20 - a00 * a21;
    const double c20 = a01 * a12 - a02 * a11;
    const double c21 = a02 * a10 - a00 * a12;
    const double c22 = a00 * a11 - a01 * a10;
    sol[0] = static_cast<float>((c00 * b[0] + c10 * b[1] + c20 * b[2]) * inv_det);
    sol[1] = static_cast<float>((c01 * b[0] + c11 * b[1] + c21 * b[2]) * inv_det);
    sol[2] = static_cast<float>((c02 * b[0] + c12 * b[1] + c22 * b[2]) * inv_det);
    return std::isfinite(sol[0]) && std::isfinite(sol[1]) && std::isfinite(sol[2]);
}

struct DescGridLUT {
    static constexpr int32_t N = 256;
    alignas(64) float u[N];
    alignas(64) float v[N];
    alignas(64) float gauss[N];
    int16_t idx00[N], idx01[N], idx10[N], idx11[N];
    float w00[N], w01[N], w10[N], w11[N];

    DescGridLUT() {
        constexpr int32_t d = 4;
        constexpr int32_t n = 8;
        int32_t k = 0;
        for (int32_t ri = 0; ri < 16; ++ri) {
            const float r_rot = -1.875f + 0.25f * static_cast<float>(ri);
            const float rb = r_rot + 1.5f;
            const int32_t r0 = static_cast<int32_t>(std::floor(rb));
            const float dr = rb - static_cast<float>(r0);
            for (int32_t ci = 0; ci < 16; ++ci) {
                const float c_rot = -1.875f + 0.25f * static_cast<float>(ci);
                const float cb = c_rot + 1.5f;
                const int32_t c0 = static_cast<int32_t>(std::floor(cb));
                const float dc = cb - static_cast<float>(c0);
                const float g = std::exp(-(r_rot * r_rot + c_rot * c_rot) * 0.125f);
                u[k] = c_rot;
                v[k] = r_rot;
                gauss[k] = g;
                w00[k] = g * (1.0f - dr) * (1.0f - dc);
                w01[k] = g * (1.0f - dr) * dc;
                w10[k] = g * dr * (1.0f - dc);
                w11[k] = g * dr * dc;
                const int32_t stride_r = (d + 2) * (n + 2);
                const int32_t stride_c = (n + 2);
                idx00[k] = static_cast<int16_t>((r0 + 1) * stride_r + (c0 + 1) * stride_c);
                idx01[k] = static_cast<int16_t>(idx00[k] + stride_c);
                idx10[k] = static_cast<int16_t>(idx00[k] + stride_r);
                idx11[k] = static_cast<int16_t>(idx00[k] + stride_r + stride_c);
                ++k;
            }
        }
    }
};

struct OriGridLUT {
    static constexpr int32_t N = 128;
    static constexpr int32_t N_VALID = 121;
    alignas(64) float du[N];
    alignas(64) float dv[N];
    alignas(64) float gauss[N];

    OriGridLUT() {
        int32_t k = 0;
        for (int32_t ri = -5; ri <= 5; ++ri) {
            const float fy = 0.6f * static_cast<float>(ri);
            for (int32_t ci = -5; ci <= 5; ++ci) {
                const float fx = 0.6f * static_cast<float>(ci);
                du[k] = fx;
                dv[k] = fy;
                gauss[k] = std::exp(-0.5f * (fx * fx + fy * fy));
                ++k;
            }
        }
        for (; k < N; ++k) { du[k] = 0.0f; dv[k] = 0.0f; gauss[k] = 0.0f; }
    }
};

const DescGridLUT g_desc_lut;
const OriGridLUT g_ori_lut;

float calc_orientation_hist_grid_avx512(
    const float* __restrict gimg,
    int32_t h,
    int32_t w,
    float ptx,
    float pty,
    float ori_sig,
    float hist[36]
) {
    float temphist[40];
    std::memset(temphist, 0, sizeof(temphist));

    const __m512 v_ptx = _mm512_set1_ps(ptx);
    const __m512 v_pty = _mm512_set1_ps(pty);
    const __m512 v_sig = _mm512_set1_ps(ori_sig);
    const __m512 v_bscale = _mm512_set1_ps(36.0f / 360.0f);
    const __m512i v_one = _mm512_set1_epi32(1);
    const __m512i v_zero = _mm512_setzero_si512();
    const __m512i v_36 = _mm512_set1_epi32(36);
    const __m512i v_w_m2 = _mm512_set1_epi32(w - 2);
    const __m512i v_h_m2 = _mm512_set1_epi32(h - 2);
    const __m512i v_w = _mm512_set1_epi32(w);

    alignas(64) int32_t bin_buf[16];
    alignas(64) float w_buf[16];

    for (int32_t k = 0; k < OriGridLUT::N; k += 16) {
        const __m512 sx = _mm512_fmadd_ps(_mm512_load_ps(g_ori_lut.du + k), v_sig, v_ptx);
        const __m512 sy = _mm512_fmadd_ps(_mm512_load_ps(g_ori_lut.dv + k), v_sig, v_pty);
        __m512i ix = _mm512_cvtps_epi32(sx);
        __m512i iy = _mm512_cvtps_epi32(sy);
        const __mmask16 in_m =
            _mm512_cmp_epi32_mask(ix, v_one, _MM_CMPINT_GE)
            & _mm512_cmp_epi32_mask(ix, v_w_m2, _MM_CMPINT_LE)
            & _mm512_cmp_epi32_mask(iy, v_one, _MM_CMPINT_GE)
            & _mm512_cmp_epi32_mask(iy, v_h_m2, _MM_CMPINT_LE);
        if (!in_m) continue;

        ix = _mm512_max_epi32(v_one, _mm512_min_epi32(v_w_m2, ix));
        iy = _mm512_max_epi32(v_one, _mm512_min_epi32(v_h_m2, iy));
        const __m512i off = _mm512_add_epi32(_mm512_mullo_epi32(iy, v_w), ix);

        const __m512 gx = _mm512_sub_ps(
            _mm512_i32gather_ps(_mm512_add_epi32(off, v_one), gimg, 4),
            _mm512_i32gather_ps(_mm512_sub_epi32(off, v_one), gimg, 4)
        );
        const __m512 gy = _mm512_sub_ps(
            _mm512_i32gather_ps(_mm512_sub_epi32(off, v_w), gimg, 4),
            _mm512_i32gather_ps(_mm512_add_epi32(off, v_w), gimg, 4)
        );
        const __m512 mag = _mm512_sqrt_ps(_mm512_fmadd_ps(gx, gx, _mm512_mul_ps(gy, gy)));
        const __m512 wt = _mm512_maskz_mul_ps(in_m, mag, _mm512_load_ps(g_ori_lut.gauss + k));
        const __m512 ori = fast_atan2_deg_avx512(gy, gx);
        __m512i bins = _mm512_cvtps_epi32(_mm512_mul_ps(ori, v_bscale));
        bins = _mm512_mask_sub_epi32(bins, _mm512_cmp_epi32_mask(bins, v_36, _MM_CMPINT_GE), bins, v_36);
        bins = _mm512_mask_add_epi32(bins, _mm512_cmp_epi32_mask(bins, v_zero, _MM_CMPINT_LT), bins, v_36);

        _mm512_store_si512(bin_buf, bins);
        _mm512_store_ps(w_buf, wt);
        for (int32_t lane = 0; lane < 16; ++lane) {
            temphist[bin_buf[lane] + 2] += w_buf[lane];
        }
    }

    temphist[0] = temphist[36];
    temphist[1] = temphist[37];
    temphist[38] = temphist[2];
    temphist[39] = temphist[3];

    float max_val = 0.0f;
    for (int32_t i = 0; i < 36; ++i) {
        float v = (temphist[i] + temphist[i + 4]) * (1.0f / 16.0f)
                + (temphist[i + 1] + temphist[i + 3]) * (4.0f / 16.0f)
                + temphist[i + 2] * (6.0f / 16.0f);
        hist[i] = v;
        if (v > max_val) max_val = v;
    }
    return max_val;
}

void calc_sift_descriptor_grid_avx512(
    const float* __restrict gimg,
    int32_t h,
    int32_t w,
    float ptx,
    float pty,
    float ori_deg,
    float scl,
    float* __restrict dst_desc
) {
    constexpr int32_t d = 4;
    constexpr int32_t n = 8;
    const float hist_width = 3.0f * scl;
    const float cos_s = std::cos(ori_deg * DEG2RAD) * hist_width;
    const float sin_s = std::sin(ori_deg * DEG2RAD) * hist_width;

    const __m512 v_ptx = _mm512_set1_ps(ptx);
    const __m512 v_pty = _mm512_set1_ps(pty);
    const __m512 v_cos = _mm512_set1_ps(cos_s);
    const __m512 v_sin = _mm512_set1_ps(sin_s);
    const __m512 v_ori = _mm512_set1_ps(ori_deg);
    const __m512 v_bpd = _mm512_set1_ps(static_cast<float>(n) / 360.0f);
    const __m512i v_one = _mm512_set1_epi32(1);
    const __m512i v_seven = _mm512_set1_epi32(7);
    const __m512i v_w_m2 = _mm512_set1_epi32(w - 2);
    const __m512i v_h_m2 = _mm512_set1_epi32(h - 2);
    const __m512i v_w = _mm512_set1_epi32(w);

    float hist[(d + 2) * (d + 2) * (n + 2)];
    std::memset(hist, 0, sizeof(hist));

    alignas(64) float buf_mag[16];
    alignas(64) float buf_dob[16];
    alignas(64) int32_t buf_o0[16];

    for (int32_t k = 0; k < DescGridLUT::N; k += 16) {
        const __m512 vu = _mm512_load_ps(g_desc_lut.u + k);
        const __m512 vv = _mm512_load_ps(g_desc_lut.v + k);
        const __m512 sx = _mm512_add_ps(v_ptx, _mm512_fmsub_ps(vu, v_cos, _mm512_mul_ps(vv, v_sin)));
        const __m512 sy = _mm512_add_ps(v_pty, _mm512_fmadd_ps(vu, v_sin, _mm512_mul_ps(vv, v_cos)));

        __m512i ix = _mm512_cvtps_epi32(sx);
        __m512i iy = _mm512_cvtps_epi32(sy);
        const __mmask16 in_m =
            _mm512_cmp_epi32_mask(ix, v_one, _MM_CMPINT_GE)
            & _mm512_cmp_epi32_mask(ix, v_w_m2, _MM_CMPINT_LE)
            & _mm512_cmp_epi32_mask(iy, v_one, _MM_CMPINT_GE)
            & _mm512_cmp_epi32_mask(iy, v_h_m2, _MM_CMPINT_LE);
        if (!in_m) continue;

        ix = _mm512_max_epi32(v_one, _mm512_min_epi32(v_w_m2, ix));
        iy = _mm512_max_epi32(v_one, _mm512_min_epi32(v_h_m2, iy));
        const __m512i off = _mm512_add_epi32(_mm512_mullo_epi32(iy, v_w), ix);

        const __m512 gx = _mm512_sub_ps(
            _mm512_i32gather_ps(_mm512_add_epi32(off, v_one), gimg, 4),
            _mm512_i32gather_ps(_mm512_sub_epi32(off, v_one), gimg, 4)
        );
        const __m512 gy = _mm512_sub_ps(
            _mm512_i32gather_ps(_mm512_sub_epi32(off, v_w), gimg, 4),
            _mm512_i32gather_ps(_mm512_add_epi32(off, v_w), gimg, 4)
        );
        const __m512 mag = _mm512_maskz_sqrt_ps(in_m, _mm512_fmadd_ps(gx, gx, _mm512_mul_ps(gy, gy)));
        const __m512 ori = fast_atan2_deg_avx512(gy, gx);
        const __m512 ob = _mm512_mul_ps(_mm512_sub_ps(ori, v_ori), v_bpd);
        const __m512 ob_fl = _mm512_roundscale_ps(ob, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        const __m512 dob = _mm512_sub_ps(ob, ob_fl);
        const __m512i o0 = _mm512_and_si512(_mm512_cvttps_epi32(ob_fl), v_seven);

        _mm512_store_ps(buf_mag, mag);
        _mm512_store_ps(buf_dob, dob);
        _mm512_store_si512(buf_o0, o0);

        for (int32_t lane = 0; lane < 16; ++lane) {
            const float m = buf_mag[lane];
            if (m == 0.0f) continue;
            const int32_t sk = k + lane;
            const int32_t bin0 = buf_o0[lane];
            const float o1w = buf_dob[lane];
            const float o0w = 1.0f - o1w;

            const float m00 = m * g_desc_lut.w00[sk];
            const float m01 = m * g_desc_lut.w01[sk];
            const float m10 = m * g_desc_lut.w10[sk];
            const float m11 = m * g_desc_lut.w11[sk];

            const int32_t p00 = g_desc_lut.idx00[sk] + bin0;
            const int32_t p01 = g_desc_lut.idx01[sk] + bin0;
            const int32_t p10 = g_desc_lut.idx10[sk] + bin0;
            const int32_t p11 = g_desc_lut.idx11[sk] + bin0;

            hist[p00] += m00 * o0w; hist[p00 + 1] += m00 * o1w;
            hist[p01] += m01 * o0w; hist[p01 + 1] += m01 * o1w;
            hist[p10] += m10 * o0w; hist[p10 + 1] += m10 * o1w;
            hist[p11] += m11 * o0w; hist[p11 + 1] += m11 * o1w;
        }
    }

    alignas(64) float raw[128];
    for (int32_t i = 0; i < d; ++i) {
        for (int32_t j = 0; j < d; ++j) {
            const int32_t idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
            hist[idx] += hist[idx + n];
            hist[idx + 1] += hist[idx + n + 1];
            float* out_cell = raw + (i * d + j) * n;
            for (int32_t k = 0; k < n; ++k) out_cell[k] = hist[idx + k];
        }
    }

    // Vectorized 128-float L2 norm -> 0.2 clip -> L2 norm -> [0, 255] round
    __m512 acc0 = _mm512_setzero_ps();
    for (int32_t k = 0; k < 128; k += 16) {
        __m512 v = _mm512_load_ps(raw + k);
        acc0 = _mm512_fmadd_ps(v, v, acc0);
    }
    const __m512 v_thr = _mm512_set1_ps(std::sqrt(_mm512_reduce_add_ps(acc0)) * 0.2f);
    __m512 acc1 = _mm512_setzero_ps();
    for (int32_t k = 0; k < 128; k += 16) {
        __m512 v = _mm512_min_ps(_mm512_load_ps(raw + k), v_thr);
        _mm512_store_ps(raw + k, v);
        acc1 = _mm512_fmadd_ps(v, v, acc1);
    }
    const __m512 v_scale = _mm512_set1_ps(512.0f / std::max(std::sqrt(_mm512_reduce_add_ps(acc1)), 1e-6f));
    const __m512 v_zero = _mm512_setzero_ps();
    const __m512 v_255 = _mm512_set1_ps(255.0f);
    for (int32_t k = 0; k < 128; k += 16) {
        __m512 v = _mm512_roundscale_ps(_mm512_mul_ps(_mm512_load_ps(raw + k), v_scale),
                                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v = _mm512_min_ps(v_255, _mm512_max_ps(v_zero, v));
        _mm512_storeu_ps(dst_desc + k, v);
    }
}

struct SiftWorkspace {
    int32_t h = 0, w = 0, first_octave = -1, n_octaves = 0;
    std::vector<int32_t> oct_h, oct_w;
    std::vector<std::vector<std::vector<float>>> gpyr;
    std::vector<std::vector<std::vector<float>>> dogpyr;
    std::vector<float> ring_buf;
    std::vector<float> base_raw;
    std::vector<float> row_tmp;

    void ensure(int32_t ih, int32_t iw, int32_t fo) {
        if (h == ih && w == iw && first_octave == fo) return;
        h = ih; w = iw; first_octave = fo;
        const int32_t base_h = (fo < 0) ? (h * 2) : h;
        const int32_t base_w = (fo < 0) ? (w * 2) : w;
        n_octaves = static_cast<int32_t>(std::round(std::log2(static_cast<double>(std::min(base_h, base_w))))) - 2;
        n_octaves = std::max(1, std::min(n_octaves, 6));
        oct_h.resize(n_octaves);
        oct_w.resize(n_octaves);
        gpyr.resize(n_octaves);
        dogpyr.resize(n_octaves);
        for (int32_t o = 0; o < n_octaves; ++o) {
            oct_h[o] = (o == 0) ? base_h : (oct_h[o - 1] / 2);
            oct_w[o] = (o == 0) ? base_w : (oct_w[o - 1] / 2);
            const size_t npix = static_cast<size_t>(oct_h[o]) * oct_w[o];
            gpyr[o].resize(5);  // layer 5 only contributes to dogpyr[o][4], never stored!
            dogpyr[o].resize(5);
            for (int32_t i = 0; i < 5; ++i) gpyr[o][i].resize(npix);
            for (int32_t i = 0; i < 5; ++i) dogpyr[o][i].resize(npix);
        }
        ring_buf.resize(static_cast<size_t>(48) * base_w);
        base_raw.resize(static_cast<size_t>(base_h) * base_w);
        row_tmp.resize(static_cast<size_t>(h) * base_w);
    }
};

thread_local SiftWorkspace g_ws;

}  // namespace

extern "C" {

int32_t extract_sift_grid_avx512(
    const uint8_t* gray,
    int32_t h,
    int32_t w,
    int32_t target_features,
    int32_t grid_cols,
    int32_t grid_rows,
    float contrast_threshold,
    int32_t first_octave,
    float* out_pixels,
    float* out_descriptors,
    int16_t* out_octaves,
    float* out_angles,
    float* out_responses,
    int32_t* out_count
) {
    if (h < 16 || w < 16 || target_features <= 0) {
        *out_count = 0;
        return 1;
    }

    constexpr int32_t nOctaveLayers = 3;
    constexpr int32_t nG = nOctaveLayers + 3;
    constexpr float sigma = 1.6f;
    constexpr float edgeThreshold = 10.0f;
    constexpr int32_t img_border = 5;

    g_ws.ensure(h, w, first_octave);
    const int32_t n_octaves = g_ws.n_octaves;

    float sig[nG];
    sig[0] = sigma;
    const double k_step = std::pow(2.0, 1.0 / nOctaveLayers);
    for (int32_t i = 1; i < nG; ++i) {
        const double sig_prev = std::pow(k_step, i - 1) * sigma;
        const double sig_total = sig_prev * k_step;
        sig[i] = static_cast<float>(std::sqrt(sig_total * sig_total - sig_prev * sig_prev));
    }

    const float pre_thresh = std::floor(0.5f * contrast_threshold / nOctaveLayers * 255.0f);
    const float final_thresh = (contrast_threshold / nOctaveLayers) * 255.0f;
    const __m512 v_pre_thresh = _mm512_set1_ps(pre_thresh);
    const __m512 v_abs_mask = _mm512_set1_ps(-0.0f);
    const __m512 v_zero = _mm512_setzero_ps();

    std::vector<RawExtremum> raw_ext;
    raw_ext.reserve(8192);

    // 1+2. Per-octave fused Gaussian + DoG pyramid followed IMMEDIATELY by L2/L3-warm extrema scan!
    for (int32_t o = 0; o < n_octaves; ++o) {
        const int32_t oh = g_ws.oct_h[o];
        const int32_t ow = g_ws.oct_w[o];
        const size_t npix = static_cast<size_t>(oh) * ow;
        if (o == 0) {
            float sig_diff = 0.0f;
            if (first_octave < 0) {
                upsample_2x_avx512(gray, h, w, g_ws.base_raw.data(), g_ws.row_tmp.data());
                sig_diff = std::sqrt(std::max(sigma * sigma - 4.0f * 0.25f, 0.01f));
            } else {
                size_t idx = 0;
                for (; idx + 15 < npix; idx += 16) {
                    _mm512_storeu_ps(g_ws.base_raw.data() + idx,
                                     _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(
                                         _mm_loadu_si128(reinterpret_cast<const __m128i*>(gray + idx)))));
                }
                for (; idx < npix; ++idx) g_ws.base_raw[idx] = static_cast<float>(gray[idx]);
                sig_diff = std::sqrt(std::max(sigma * sigma - 0.25f, 0.01f));
            }
            gaussian_blur_ring_avx512(g_ws.base_raw.data(), g_ws.gpyr[0][0].data(), nullptr,
                                      g_ws.ring_buf.data(), oh, ow, sig_diff);
        } else {
            downsample_2x(g_ws.gpyr[o - 1][nOctaveLayers].data(), g_ws.oct_h[o - 1], g_ws.oct_w[o - 1],
                          g_ws.gpyr[o][0].data(), oh, ow);
        }

        // Layers 1..4 write both gpyr[o][i] and fused dogpyr[o][i-1] = gpyr[o][i] - gpyr[o][i-1]
        for (int32_t i = 1; i < nG - 1; ++i) {
            gaussian_blur_ring_avx512(g_ws.gpyr[o][i - 1].data(), g_ws.gpyr[o][i].data(),
                                      g_ws.dogpyr[o][i - 1].data(), g_ws.ring_buf.data(),
                                      oh, ow, sig[i]);
        }
        // Layer 5 only writes dogpyr[o][4] = blur(gpyr[o][4]) - gpyr[o][4] (skips unused gpyr[o][5] store!)
        gaussian_blur_ring_avx512(g_ws.gpyr[o][nG - 2].data(), nullptr,
                                  g_ws.dogpyr[o][nG - 2].data(), g_ws.ring_buf.data(),
                                  oh, ow, sig[nG - 1]);

        if (oh <= 2 * img_border || ow <= 2 * img_border) continue;
        const int32_t actual_oct = o + first_octave;
        const float oct_scale = (actual_oct >= 0)
            ? static_cast<float>(1 << actual_oct)
            : (1.0f / static_cast<float>(1 << (-actual_oct)));

        for (int32_t i = 1; i <= nOctaveLayers; ++i) {
            const float* prev_img = g_ws.dogpyr[o][i - 1].data();
            const float* curr_img = g_ws.dogpyr[o][i].data();
            const float* next_img = g_ws.dogpyr[o][i + 1].data();

            for (int32_t r = img_border; r < oh - img_border; ++r) {
                const size_t row_off = static_cast<size_t>(r) * ow;
                const float* rc = curr_img + row_off;
                const float* rp = curr_img + row_off - ow;
                const float* rn = curr_img + row_off + ow;
                const float* p_row = prev_img + row_off;
                const float* n_row = next_img + row_off;

                int32_t c = img_border;
                for (; c + 15 < ow - img_border; c += 16) {
                    const __m512 vc = _mm512_loadu_ps(rc + c);
                    const __m512 av = _mm512_andnot_ps(v_abs_mask, vc);
                    __mmask16 cand = _mm512_cmp_ps_mask(av, v_pre_thresh, _CMP_GT_OQ);
                    if (!cand) continue;

                    const __m512 vl = _mm512_loadu_ps(rc + c - 1);
                    const __m512 vr = _mm512_loadu_ps(rc + c + 1);
                    const __m512 vu = _mm512_loadu_ps(rp + c);
                    const __m512 vd = _mm512_loadu_ps(rn + c);
                    const __m512 vs_p = _mm512_loadu_ps(p_row + c);
                    const __m512 vs_n = _mm512_loadu_ps(n_row + c);

                    const __m512 nmax = _mm512_max_ps(
                        _mm512_max_ps(_mm512_max_ps(vl, vr), _mm512_max_ps(vu, vd)),
                        _mm512_max_ps(vs_p, vs_n)
                    );
                    const __m512 nmin = _mm512_min_ps(
                        _mm512_min_ps(_mm512_min_ps(vl, vr), _mm512_min_ps(vu, vd)),
                        _mm512_min_ps(vs_p, vs_n)
                    );
                    const __mmask16 is_pos = _mm512_cmp_ps_mask(vc, v_zero, _CMP_GT_OQ);
                    const __mmask16 ok_max = is_pos & _mm512_cmp_ps_mask(vc, nmax, _CMP_GE_OQ);
                    const __mmask16 ok_min = (~is_pos) & _mm512_cmp_ps_mask(vc, nmin, _CMP_LE_OQ);
                    cand &= (ok_max | ok_min);
                    if (!cand) continue;

                    uint32_t bits = static_cast<uint32_t>(cand);
                    while (bits) {
                        const int32_t lane = __builtin_ctz(bits);
                        bits &= bits - 1;
                        const int32_t cx = c + lane;
                        const float val = rc[cx];

                        bool ext = true;
                        if (val > 0.0f) {
                            for (int32_t dl = -1; dl <= 1 && ext; ++dl) {
                                const float* img_l = (dl == -1) ? prev_img : (dl == 0 ? curr_img : next_img);
                                for (int32_t dr = -1; dr <= 1; ++dr) {
                                    const float* rr = img_l + static_cast<size_t>(r + dr) * ow + cx;
                                    if (val < rr[-1] || val < rr[0] || val < rr[1]) { ext = false; break; }
                                }
                            }
                        } else {
                            for (int32_t dl = -1; dl <= 1 && ext; ++dl) {
                                const float* img_l = (dl == -1) ? prev_img : (dl == 0 ? curr_img : next_img);
                                for (int32_t dr = -1; dr <= 1; ++dr) {
                                    const float* rr = img_l + static_cast<size_t>(r + dr) * ow + cx;
                                    if (val > rr[-1] || val > rr[0] || val > rr[1]) { ext = false; break; }
                                }
                            }
                        }
                        if (!ext) continue;

                        int32_t ri = r, ci = cx, li = i;
                        float xc = 0.0f, xr = 0.0f, xi = 0.0f;
                        bool converged = false;
                        for (int32_t step = 0; step < 5; ++step) {
                            const float* dp = g_ws.dogpyr[o][li - 1].data();
                            const float* dc = g_ws.dogpyr[o][li].data();
                            const float* dn = g_ws.dogpyr[o][li + 1].data();
                            const size_t idx = static_cast<size_t>(ri) * ow + ci;
                            const float v0 = dc[idx];
                            const float dx = 0.5f * (dc[idx + 1] - dc[idx - 1]);
                            const float dy = 0.5f * (dc[idx + ow] - dc[idx - ow]);
                            const float ds = 0.5f * (dn[idx] - dp[idx]);
                            const float dxx = dc[idx + 1] + dc[idx - 1] - 2.0f * v0;
                            const float dyy = dc[idx + ow] + dc[idx - ow] - 2.0f * v0;
                            const float dss = dn[idx] + dp[idx] - 2.0f * v0;
                            const float dxy = 0.25f * (dc[idx + ow + 1] - dc[idx + ow - 1] - dc[idx - ow + 1] + dc[idx - ow - 1]);
                            const float dxs = 0.25f * (dn[idx + 1] - dn[idx - 1] - dp[idx + 1] + dp[idx - 1]);
                            const float dys = 0.25f * (dn[idx + ow] - dn[idx - ow] - dp[idx + ow] + dp[idx - ow]);
                            const float H[3][3] = {{dxx, dxy, dxs}, {dxy, dyy, dys}, {dxs, dys, dss}};
                            const float b[3] = {-dx, -dy, -ds};
                            float sol[3];
                            if (!solve3x3(H, b, sol)) break;
                            xc = sol[0]; xr = sol[1]; xi = sol[2];
                            if (std::abs(xc) < 0.5f && std::abs(xr) < 0.5f && std::abs(xi) < 0.5f) {
                                converged = true;
                                break;
                            }
                            if (std::abs(xc) > 1e5f || std::abs(xr) > 1e5f || std::abs(xi) > 1e5f) break;
                            ci += static_cast<int32_t>(std::round(xc));
                            ri += static_cast<int32_t>(std::round(xr));
                            li += static_cast<int32_t>(std::round(xi));
                            if (li < 1 || li > nOctaveLayers || ci < img_border || ci >= ow - img_border ||
                                ri < img_border || ri >= oh - img_border) {
                                break;
                            }
                        }
                        if (!converged) continue;

                        const float* dp = g_ws.dogpyr[o][li - 1].data();
                        const float* dc = g_ws.dogpyr[o][li].data();
                        const float* dn = g_ws.dogpyr[o][li + 1].data();
                        const size_t idx = static_cast<size_t>(ri) * ow + ci;
                        const float v0 = dc[idx];
                        const float dx = 0.5f * (dc[idx + 1] - dc[idx - 1]);
                        const float dy = 0.5f * (dc[idx + ow] - dc[idx - ow]);
                        const float ds = 0.5f * (dn[idx] - dp[idx]);
                        const float contr = v0 + 0.5f * (dx * xc + dy * xr + ds * xi);
                        if (std::abs(contr) < final_thresh) continue;

                        const float dxx = dc[idx + 1] + dc[idx - 1] - 2.0f * v0;
                        const float dyy = dc[idx + ow] + dc[idx - ow] - 2.0f * v0;
                        const float dxy = 0.25f * (dc[idx + ow + 1] - dc[idx + ow - 1] - dc[idx - ow + 1] + dc[idx - ow - 1]);
                        const float tr = dxx + dyy;
                        const float det = dxx * dyy - dxy * dxy;
                        if (det <= 0.0f || tr * tr * edgeThreshold >= (edgeThreshold + 1.0f) * (edgeThreshold + 1.0f) * det) {
                            continue;
                        }

                        const float ptx_img = (static_cast<float>(ci) + xc) * oct_scale;
                        const float pty_img = (static_cast<float>(ri) + xr) * oct_scale;
                        if (ptx_img < 0.0f || ptx_img >= static_cast<float>(w) ||
                            pty_img < 0.0f || pty_img >= static_cast<float>(h)) {
                            continue;
                        }

                        RawExtremum ex;
                        ex.x = ptx_img;
                        ex.y = pty_img;
                        ex.response = std::abs(contr) / 255.0f;
                        ex.scl_octv = sigma * std::pow(2.0f, (static_cast<float>(li) + xi) / nOctaveLayers);
                        ex.ri = ri;
                        ex.ci = ci;
                        ex.octave_idx = actual_oct;
                        ex.layer = li;
                        ex.pyr_o = o;
                        raw_ext.push_back(ex);
                    }
                }
            }
        }
    }

    const int32_t total_ext = static_cast<int32_t>(raw_ext.size());
    if (total_ext == 0) {
        *out_count = 0;
        return 1;
    }

    // 3. Response-cap raw extrema before orientation calculation
    const int32_t raw_cap = std::max(static_cast<int32_t>(target_features * 1.4), 2100);
    std::vector<int32_t> ext_order(total_ext);
    for (int32_t i = 0; i < total_ext; ++i) ext_order[i] = i;
    if (total_ext > raw_cap) {
        std::nth_element(ext_order.begin(), ext_order.begin() + raw_cap, ext_order.end(),
                         [&](int32_t a, int32_t b) { return raw_ext[a].response > raw_ext[b].response; });
        ext_order.resize(raw_cap);
    }
    std::stable_sort(ext_order.begin(), ext_order.end(), [&](int32_t a, int32_t b) {
        return raw_ext[a].response > raw_ext[b].response;
    });

    std::vector<SiftKeypoint> kps;
    kps.reserve(raw_cap + 512);
    for (int32_t ei : ext_order) {
        const RawExtremum& ex = raw_ext[ei];
        float ohist[36];
        const float inv_oct_scale = (ex.octave_idx >= 0)
            ? (1.0f / static_cast<float>(1 << ex.octave_idx))
            : static_cast<float>(1 << (-ex.octave_idx));
        const float omax = calc_orientation_hist_grid_avx512(
            g_ws.gpyr[ex.pyr_o][ex.layer].data(),
            g_ws.oct_h[ex.pyr_o], g_ws.oct_w[ex.pyr_o],
            ex.x * inv_oct_scale, ex.y * inv_oct_scale,
            1.5f * ex.scl_octv, ohist
        );
        const float mag_thr = 0.8f * omax;
        for (int32_t b = 0; b < 36; ++b) {
            const int32_t bl = (b > 0) ? (b - 1) : 35;
            const int32_t br = (b < 35) ? (b + 1) : 0;
            if (ohist[b] > ohist[bl] && ohist[b] > ohist[br] && ohist[b] >= mag_thr) {
                const float denom = ohist[bl] - 2.0f * ohist[b] + ohist[br];
                float bin = static_cast<float>(b) + (std::abs(denom) > 1e-12f ? 0.5f * (ohist[bl] - ohist[br]) / denom : 0.0f);
                if (bin < 0.0f) bin += 36.0f;
                if (bin >= 36.0f) bin -= 36.0f;
                float angle = 360.0f - bin * (360.0f / 36.0f);
                if (std::abs(angle - 360.0f) < 1e-4f) angle = 0.0f;
                SiftKeypoint kp;
                kp.x = ex.x; kp.y = ex.y; kp.angle = angle;
                kp.response = ex.response; kp.scl_octv = ex.scl_octv;
                kp.octave_idx = ex.octave_idx; kp.layer = ex.layer; kp.pyr_o = ex.pyr_o;
                kps.push_back(kp);
            }
        }
        if (static_cast<int32_t>(kps.size()) >= raw_cap) break;
    }

    const int32_t pool_n = std::min(static_cast<int32_t>(kps.size()), raw_cap);
    std::vector<int32_t> kept;
    kept.reserve(target_features);
    if (pool_n <= target_features) {
        for (int32_t pos = 0; pos < pool_n; ++pos) kept.push_back(pos);
    } else {
        const int32_t cols = std::max(1, grid_cols);
        const int32_t rows = std::max(1, grid_rows);
        const int32_t n_cells = cols * rows;
        const int32_t cell_quota = std::max(1, static_cast<int32_t>(std::ceil(static_cast<double>(target_features) / n_cells)));
        std::vector<int32_t> cell_counts(n_cells, 0);
        std::vector<uint8_t> selected(pool_n, 0);
        const float sx = static_cast<float>(cols) / static_cast<float>(std::max(w, 1));
        const float sy = static_cast<float>(rows) / static_cast<float>(std::max(h, 1));

        for (int32_t pos = 0; pos < pool_n && static_cast<int32_t>(kept.size()) < target_features; ++pos) {
            const SiftKeypoint& kp = kps[pos];
            int32_t gx = std::max(0, std::min(cols - 1, static_cast<int32_t>(kp.x * sx)));
            int32_t gy = std::max(0, std::min(rows - 1, static_cast<int32_t>(kp.y * sy)));
            int32_t cid = gy * cols + gx;
            if (cell_counts[cid] < cell_quota) {
                cell_counts[cid]++;
                selected[pos] = 1;
                kept.push_back(pos);
            }
        }
        for (int32_t pos = 0; pos < pool_n && static_cast<int32_t>(kept.size()) < target_features; ++pos) {
            if (!selected[pos]) kept.push_back(pos);
        }
    }

    // 4. Precomputed 16x16 grid AVX-512 SIFT descriptors for kept keypoints only
    const int32_t n_kept = static_cast<int32_t>(kept.size());
    for (int32_t idx = 0; idx < n_kept; ++idx) {
        const SiftKeypoint& kp = kps[kept[idx]];
        const int32_t o = kp.pyr_o;
        const float inv_oct_scale = (kp.octave_idx >= 0)
            ? (1.0f / static_cast<float>(1 << kp.octave_idx))
            : static_cast<float>(1 << (-kp.octave_idx));
        const float ori = 360.0f - kp.angle;
        calc_sift_descriptor_grid_avx512(
            g_ws.gpyr[o][kp.layer].data(),
            g_ws.oct_h[o],
            g_ws.oct_w[o],
            kp.x * inv_oct_scale,
            kp.y * inv_oct_scale,
            (ori >= 360.0f ? 0.0f : ori),
            kp.scl_octv,
            out_descriptors + static_cast<size_t>(idx) * 128
        );
        out_pixels[2 * idx + 0] = kp.x;
        out_pixels[2 * idx + 1] = kp.y;
        out_octaves[idx] = static_cast<int16_t>(kp.octave_idx);
        out_angles[idx] = kp.angle;
        out_responses[idx] = kp.response;
    }

    *out_count = n_kept;
    return 1;
}

}  // extern "C"
