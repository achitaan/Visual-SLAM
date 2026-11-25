#include "portable_export.h"
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace {

constexpr float PI_F = 3.14159265358979323846f;
constexpr float RAD2DEG = 180.0f / PI_F;
constexpr float DEG2RAD = PI_F / 180.0f;
constexpr int32_t Q_SHIFT = 7;            // Q7 fixed point for int16 gpyr (0..255 -> 0..32640)
constexpr int32_t DOG_SHIFT = 9;          // Q9 fixed point for int16 dogpyr (1/512 gray level, range [-64,+64])
constexpr float INV_Q = 1.0f / 128.0f;
constexpr float INV_DOG_Q = 1.0f / 512.0f;

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#define VSLAM_SIFT_AVX512 1
#else
#define VSLAM_SIFT_AVX512 0
#endif

#if VSLAM_SIFT_AVX512
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
#else
inline float fast_atan2_deg_scalar(float y, float x) {
    float deg = std::atan2(y, x) * RAD2DEG;
    if (deg < 0.0f) deg += 360.0f;
    if (deg >= 360.0f) deg -= 360.0f;
    return deg;
}
#endif

void make_gaussian_kernel(float sigma, float half_w[25], int32_t& rad) {
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

void make_gaussian_kernel_q15(float sigma, int16_t hw_q15[25], int32_t& rad) {
    float hw[25];
    make_gaussian_kernel(sigma, hw, rad);
    int32_t sum = 0;
    for (int32_t k = 1; k <= rad; ++k) {
        int32_t v = static_cast<int32_t>(std::lround(hw[k] * 32768.0f));
        hw_q15[k] = static_cast<int16_t>(v);
        sum += 2 * v;
    }
    hw_q15[0] = static_cast<int16_t>(32768 - sum);
}

// =========================================================================
// Q7 int16_t gpyr + Q9 int16_t dogpyr 32-wide AVX-512BW Gaussian + DoG
// Exact int32 vpmaddwd Q22 accumulation -> Q9 DoG directly before rounding!
// =========================================================================

void upsample_2x_i16_avx512(const uint8_t* VSLAM_RESTRICT src, int32_t h, int32_t w,
                            int16_t* VSLAM_RESTRICT dst, int16_t* VSLAM_RESTRICT row_tmp) {
    const int32_t dh = h * 2;
    const int32_t dw = w * 2;
#if VSLAM_SIFT_AVX512
    const __m512i v96 = _mm512_set1_epi16(96);  // 0.75 * 128
    const __m512i v32 = _mm512_set1_epi16(32);  // 0.25 * 128
    alignas(64) static constexpr int16_t IL_LO[32] = {
        0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
        8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47};
    alignas(64) static constexpr int16_t IL_HI[32] = {
        16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55,
        24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63};
    const __m512i il_lo = _mm512_load_si512(IL_LO);
    const __m512i il_hi = _mm512_load_si512(IL_HI);
#endif

    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* sr = src + static_cast<size_t>(y) * w;
        int16_t* tr = row_tmp + static_cast<size_t>(y) * dw;
        tr[0] = static_cast<int16_t>(sr[0] << Q_SHIFT);
        tr[1] = static_cast<int16_t>(96 * sr[0] + 32 * sr[std::min(1, w - 1)]);
        int32_t x = 1;
#if VSLAM_SIFT_AVX512
        for (; x + 32 < w; x += 32) {
            const __m512i vc = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sr + x)));
            const __m512i vl = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sr + x - 1)));
            const __m512i vr = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sr + x + 1)));
            const __m512i c96 = _mm512_mullo_epi16(vc, v96);
            const __m512i ev = _mm512_add_epi16(c96, _mm512_mullo_epi16(vl, v32));
            const __m512i od = _mm512_add_epi16(c96, _mm512_mullo_epi16(vr, v32));
            _mm512_storeu_si512(tr + 2 * x,      _mm512_permutex2var_epi16(ev, il_lo, od));
            _mm512_storeu_si512(tr + 2 * x + 32, _mm512_permutex2var_epi16(ev, il_hi, od));
        }
#endif
        for (; x < w - 1; ++x) {
            const int32_t c = sr[x];
            tr[2 * x]     = static_cast<int16_t>(96 * c + 32 * sr[x - 1]);
            tr[2 * x + 1] = static_cast<int16_t>(96 * c + 32 * sr[x + 1]);
        }
        if (w > 1) {
            tr[2 * (w - 1)]     = static_cast<int16_t>(96 * sr[w - 1] + 32 * sr[w - 2]);
            tr[2 * (w - 1) + 1] = static_cast<int16_t>(sr[w - 1] << Q_SHIFT);
        }
    }

#if VSLAM_SIFT_AVX512
    const __m512i w31 = _mm512_set1_epi32((1 << 16) | 3);
    const __m512i v_rnd2 = _mm512_set1_epi32(2);
#elif defined(__AVX2__)
    const __m256i w31 = _mm256_set1_epi32((1 << 16) | 3);
    const __m256i v_rnd2 = _mm256_set1_epi32(2);
#endif
    for (int32_t dy = 0; dy < dh; ++dy) {
        const int32_t sy = dy >> 1;
        const int32_t sy_other = (dy & 1) ? std::min(sy + 1, h - 1) : std::max(sy - 1, 0);
        const int16_t* r_main = row_tmp + static_cast<size_t>(sy) * dw;
        const int16_t* r_oth  = row_tmp + static_cast<size_t>(sy_other) * dw;
        int16_t* dr = dst + static_cast<size_t>(dy) * dw;
        int32_t x = 0;
#if VSLAM_SIFT_AVX512
        for (; x + 31 < dw; x += 32) {
            __m512i m = _mm512_loadu_si512(r_main + x);
            __m512i o = _mm512_loadu_si512(r_oth + x);
            __m512i lo = _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(_mm512_unpacklo_epi16(m, o), w31), v_rnd2), 2);
            __m512i hi = _mm512_srai_epi32(_mm512_add_epi32(_mm512_madd_epi16(_mm512_unpackhi_epi16(m, o), w31), v_rnd2), 2);
            _mm512_storeu_si512(dr + x, _mm512_packs_epi32(lo, hi));
        }
#elif defined(__AVX2__)
        for (; x + 15 < dw; x += 16) {
            __m256i m = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_main + x));
            __m256i o = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_oth + x));
            __m256i lo = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(_mm256_unpacklo_epi16(m, o), w31), v_rnd2), 2);
            __m256i hi = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(_mm256_unpackhi_epi16(m, o), w31), v_rnd2), 2);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dr + x), _mm256_packs_epi32(lo, hi));
        }
#endif
        for (; x < dw; ++x) dr[x] = static_cast<int16_t>((3 * r_main[x] + r_oth[x] + 2) >> 2);
    }
}

template <int32_t RAD>
inline void blur_row_h_i16_fixed_avx512(const int16_t* VSLAM_RESTRICT srow, int16_t* VSLAM_RESTRICT trow,
                                        int32_t w, const int16_t* VSLAM_RESTRICT hw) {
    const int32_t x_left_end = std::min(RAD, w);
    for (int32_t x = 0; x < x_left_end; ++x) {
        int32_t acc = static_cast<int32_t>(hw[0]) * srow[x];
        for (int32_t k = 1; k <= RAD; ++k) {
            int32_t xl = (x - k < 0) ? 0 : (x - k);
            int32_t xr = (x + k >= w) ? (w - 1) : (x + k);
            acc += static_cast<int32_t>(hw[k]) * (static_cast<int32_t>(srow[xl]) + srow[xr]);
        }
        trow[x] = static_cast<int16_t>((acc + 16384) >> 15);
    }

    int32_t x = RAD;
    const int32_t x_mid_end = w - RAD;
#if VSLAM_SIFT_AVX512
    __m512i vw[RAD + 1];
    for (int32_t k = 0; k <= RAD; ++k) vw[k] = _mm512_set1_epi16(hw[k]);
    const __m512i v_zero = _mm512_setzero_si512();
    const __m512i v_half = _mm512_set1_epi32(16384);

    for (; x + 31 < x_mid_end; x += 32) {
        const __m512i c0 = _mm512_loadu_si512(srow + x);
        __m512i a_lo = _mm512_add_epi32(v_half, _mm512_madd_epi16(_mm512_unpacklo_epi16(c0, v_zero), vw[0]));
        __m512i a_hi = _mm512_add_epi32(v_half, _mm512_madd_epi16(_mm512_unpackhi_epi16(c0, v_zero), vw[0]));
        for (int32_t k = 1; k <= RAD; ++k) {
            const __m512i vl = _mm512_loadu_si512(srow + x - k);
            const __m512i vr = _mm512_loadu_si512(srow + x + k);
            a_lo = _mm512_add_epi32(a_lo, _mm512_madd_epi16(_mm512_unpacklo_epi16(vl, vr), vw[k]));
            a_hi = _mm512_add_epi32(a_hi, _mm512_madd_epi16(_mm512_unpackhi_epi16(vl, vr), vw[k]));
        }
        _mm512_storeu_si512(trow + x, _mm512_packs_epi32(_mm512_srai_epi32(a_lo, 15), _mm512_srai_epi32(a_hi, 15)));
    }
#elif defined(__AVX2__)
    __m256i vw[RAD + 1];
    for (int32_t k = 0; k <= RAD; ++k) vw[k] = _mm256_set1_epi16(hw[k]);
    const __m256i v_zero = _mm256_setzero_si256();
    const __m256i v_half = _mm256_set1_epi32(16384);

    for (; x + 15 < x_mid_end; x += 16) {
        const __m256i c0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srow + x));
        __m256i a_lo = _mm256_add_epi32(v_half, _mm256_madd_epi16(_mm256_unpacklo_epi16(c0, v_zero), vw[0]));
        __m256i a_hi = _mm256_add_epi32(v_half, _mm256_madd_epi16(_mm256_unpackhi_epi16(c0, v_zero), vw[0]));
        for (int32_t k = 1; k <= RAD; ++k) {
            const __m256i vl = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srow + x - k));
            const __m256i vr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srow + x + k));
            a_lo = _mm256_add_epi32(a_lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(vl, vr), vw[k]));
            a_hi = _mm256_add_epi32(a_hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(vl, vr), vw[k]));
        }
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(trow + x),
            _mm256_packs_epi32(_mm256_srai_epi32(a_lo, 15), _mm256_srai_epi32(a_hi, 15))
        );
    }
#endif
    for (; x < x_mid_end; ++x) {
        int32_t acc = static_cast<int32_t>(hw[0]) * srow[x];
        for (int32_t k = 1; k <= RAD; ++k) acc += static_cast<int32_t>(hw[k]) * (static_cast<int32_t>(srow[x - k]) + srow[x + k]);
        trow[x] = static_cast<int16_t>((acc + 16384) >> 15);
    }
    for (; x < w; ++x) {
        int32_t acc = static_cast<int32_t>(hw[0]) * srow[x];
        for (int32_t k = 1; k <= RAD; ++k) {
            int32_t xl = (x - k < 0) ? 0 : (x - k);
            int32_t xr = (x + k >= w) ? (w - 1) : (x + k);
            acc += static_cast<int32_t>(hw[k]) * (static_cast<int32_t>(srow[xl]) + srow[xr]);
        }
        trow[x] = static_cast<int16_t>((acc + 16384) >> 15);
    }
}

template <int32_t RAD, bool WRITE_G, bool WRITE_DOG>
void gaussian_blur_ring_i16_fixed_avx512(
    const int16_t* VSLAM_RESTRICT src,
    int16_t* VSLAM_RESTRICT dst,
    int16_t* VSLAM_RESTRICT dog_dst,
    int16_t* VSLAM_RESTRICT ring_buf,
    int32_t h,
    int32_t w,
    const int16_t* VSLAM_RESTRICT hw
) {
    constexpr int32_t RING_SZ = 2 * RAD + 2;
    auto ring_ptr = [&](int32_t ry) -> int16_t* {
        int32_t clamped = (ry < 0) ? 0 : (ry >= h ? h - 1 : ry);
        return ring_buf + static_cast<size_t>(clamped % RING_SZ) * w;
    };

    int32_t h_done = -1;
    const int32_t init_end = std::min(RAD, h - 1);
    for (int32_t ry = 0; ry <= init_end; ++ry) {
        blur_row_h_i16_fixed_avx512<RAD>(src + static_cast<size_t>(ry) * w, ring_ptr(ry), w, hw);
        h_done = ry;
    }

#if VSLAM_SIFT_AVX512
    __m512i vw[RAD + 1];
    for (int32_t k = 0; k <= RAD; ++k) vw[k] = _mm512_set1_epi16(hw[k]);
    const __m512i v_zero = _mm512_setzero_si512();
    const __m512i v_half = _mm512_set1_epi32(16384);
    const __m512i v_dhalf = _mm512_set1_epi32(4096);
#elif defined(__AVX2__)
    __m256i vw[RAD + 1];
    for (int32_t k = 0; k <= RAD; ++k) vw[k] = _mm256_set1_epi16(hw[k]);
    const __m256i v_zero = _mm256_setzero_si256();
    const __m256i v_half = _mm256_set1_epi32(16384);
    const __m256i v_dhalf = _mm256_set1_epi32(4096);
#endif

    for (int32_t y = 0; y < h; ++y) {
        const int32_t need_ry = std::min(y + RAD, h - 1);
        while (h_done < need_ry) {
            ++h_done;
            blur_row_h_i16_fixed_avx512<RAD>(src + static_cast<size_t>(h_done) * w, ring_ptr(h_done), w, hw);
        }

        const int16_t* r0 = ring_ptr(y);
        const int16_t* r_up[RAD + 1];
        const int16_t* r_dn[RAD + 1];
        for (int32_t k = 1; k <= RAD; ++k) {
            r_up[k] = ring_ptr(y - k);
            r_dn[k] = ring_ptr(y + k);
        }
        const size_t row_off = static_cast<size_t>(y) * w;
        const int16_t* srow = WRITE_DOG ? (src + row_off) : nullptr;
        int16_t* drow = WRITE_G ? (dst + row_off) : nullptr;
        int16_t* dogrow = WRITE_DOG ? (dog_dst + row_off) : nullptr;

        int32_t x = 0;
#if VSLAM_SIFT_AVX512
        for (; x + 31 < w; x += 32) {
            const __m512i c0 = _mm512_loadu_si512(r0 + x);
            __m512i a_lo = _mm512_madd_epi16(_mm512_unpacklo_epi16(c0, v_zero), vw[0]);
            __m512i a_hi = _mm512_madd_epi16(_mm512_unpackhi_epi16(c0, v_zero), vw[0]);
            for (int32_t k = 1; k <= RAD; ++k) {
                const __m512i vu = _mm512_loadu_si512(r_up[k] + x);
                const __m512i vd = _mm512_loadu_si512(r_dn[k] + x);
                a_lo = _mm512_add_epi32(a_lo, _mm512_madd_epi16(_mm512_unpacklo_epi16(vu, vd), vw[k]));
                a_hi = _mm512_add_epi32(a_hi, _mm512_madd_epi16(_mm512_unpackhi_epi16(vu, vd), vw[k]));
            }
            if constexpr (WRITE_G) {
                const __m512i res = _mm512_packs_epi32(
                    _mm512_srai_epi32(_mm512_add_epi32(a_lo, v_half), 15),
                    _mm512_srai_epi32(_mm512_add_epi32(a_hi, v_half), 15));
                _mm512_storeu_si512(drow + x, res);
            }
            if constexpr (WRITE_DOG) {
                const __m512i sv = _mm512_loadu_si512(srow + x);
                const __m512i s_lo = _mm512_slli_epi32(_mm512_unpacklo_epi16(sv, v_zero), 15);
                const __m512i s_hi = _mm512_slli_epi32(_mm512_unpackhi_epi16(sv, v_zero), 15);
                const __m512i d_lo = _mm512_srai_epi32(_mm512_add_epi32(_mm512_sub_epi32(a_lo, s_lo), v_dhalf), 13);
                const __m512i d_hi = _mm512_srai_epi32(_mm512_add_epi32(_mm512_sub_epi32(a_hi, s_hi), v_dhalf), 13);
                _mm512_storeu_si512(dogrow + x, _mm512_packs_epi32(d_lo, d_hi));
            }
        }
#elif defined(__AVX2__)
        for (; x + 15 < w; x += 16) {
            const __m256i c0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + x));
            __m256i a_lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(c0, v_zero), vw[0]);
            __m256i a_hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(c0, v_zero), vw[0]);
            for (int32_t k = 1; k <= RAD; ++k) {
                const __m256i vu = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_up[k] + x));
                const __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r_dn[k] + x));
                a_lo = _mm256_add_epi32(a_lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(vu, vd), vw[k]));
                a_hi = _mm256_add_epi32(a_hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(vu, vd), vw[k]));
            }
            if constexpr (WRITE_G) {
                const __m256i res = _mm256_packs_epi32(
                    _mm256_srai_epi32(_mm256_add_epi32(a_lo, v_half), 15),
                    _mm256_srai_epi32(_mm256_add_epi32(a_hi, v_half), 15));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(drow + x), res);
            }
            if constexpr (WRITE_DOG) {
                const __m256i sv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(srow + x));
                const __m256i s_lo = _mm256_slli_epi32(_mm256_unpacklo_epi16(sv, v_zero), 15);
                const __m256i s_hi = _mm256_slli_epi32(_mm256_unpackhi_epi16(sv, v_zero), 15);
                const __m256i d_lo = _mm256_srai_epi32(_mm256_add_epi32(_mm256_sub_epi32(a_lo, s_lo), v_dhalf), 13);
                const __m256i d_hi = _mm256_srai_epi32(_mm256_add_epi32(_mm256_sub_epi32(a_hi, s_hi), v_dhalf), 13);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dogrow + x), _mm256_packs_epi32(d_lo, d_hi));
            }
        }
#endif
        for (; x < w; ++x) {
            int32_t acc = static_cast<int32_t>(hw[0]) * r0[x];
            for (int32_t k = 1; k <= RAD; ++k) acc += static_cast<int32_t>(hw[k]) * (static_cast<int32_t>(r_up[k][x]) + r_dn[k][x]);
            if constexpr (WRITE_G) drow[x] = static_cast<int16_t>((acc + 16384) >> 15);
            if constexpr (WRITE_DOG) {
                int32_t diff = (acc - (static_cast<int32_t>(srow[x]) << 15) + 4096) >> 13;
                dogrow[x] = static_cast<int16_t>(std::max(-32768, std::min(32767, diff)));
            }
        }
    }
}

void gaussian_blur_ring_i16_avx512(
    const int16_t* VSLAM_RESTRICT src,
    int16_t* VSLAM_RESTRICT dst,
    int16_t* VSLAM_RESTRICT dog_dst,
    int16_t* VSLAM_RESTRICT ring_buf,
    int32_t h,
    int32_t w,
    float sigma
) {
    int16_t hw[25];
    int32_t rad = 0;
    make_gaussian_kernel_q15(sigma, hw, rad);

    auto dispatch = [&](auto wg_tag, auto wd_tag) {
        constexpr bool WG = decltype(wg_tag)::value;
        constexpr bool WD = decltype(wd_tag)::value;
        switch (rad) {
            case 3: gaussian_blur_ring_i16_fixed_avx512<3, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 4: gaussian_blur_ring_i16_fixed_avx512<4, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 5: gaussian_blur_ring_i16_fixed_avx512<5, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 6: gaussian_blur_ring_i16_fixed_avx512<6, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 7: gaussian_blur_ring_i16_fixed_avx512<7, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 8: gaussian_blur_ring_i16_fixed_avx512<8, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 9: gaussian_blur_ring_i16_fixed_avx512<9, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            case 10: gaussian_blur_ring_i16_fixed_avx512<10, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
            default: gaussian_blur_ring_i16_fixed_avx512<9, WG, WD>(src, dst, dog_dst, ring_buf, h, w, hw); return;
        }
    };
    if (dst != nullptr && dog_dst != nullptr) dispatch(std::true_type{}, std::true_type{});
    else if (dst != nullptr) dispatch(std::true_type{}, std::false_type{});
    else if (dog_dst != nullptr) dispatch(std::false_type{}, std::true_type{});
}

void downsample_2x_i16(const int16_t* VSLAM_RESTRICT src, int32_t sh, int32_t sw,
                       int16_t* VSLAM_RESTRICT dst, int32_t dh, int32_t dw) {
    (void)sh;
#if VSLAM_SIFT_AVX512
    alignas(64) static constexpr int16_t EVEN_IDX[32] = {
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62};
    const __m512i even_idx = _mm512_load_si512(EVEN_IDX);
#endif
    for (int32_t y = 0; y < dh; ++y) {
        const int16_t* sr = src + static_cast<size_t>(y * 2) * sw;
        int16_t* dr = dst + static_cast<size_t>(y) * dw;
        int32_t x = 0;
#if VSLAM_SIFT_AVX512
        for (; x + 31 < dw; x += 32) {
            __m512i a = _mm512_loadu_si512(sr + 2 * x);
            __m512i b = _mm512_loadu_si512(sr + 2 * x + 32);
            _mm512_storeu_si512(dr + x, _mm512_permutex2var_epi16(a, even_idx, b));
        }
#endif
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

#if VSLAM_SIFT_AVX512
// Gather 16 float values from either float32 or int16 Gaussian layer (buffers padded +4 elements).
template <typename T>
inline __m512 gather16_to_f32(__m512i off, const T* base);

template <>
inline __m512 gather16_to_f32<float>(__m512i off, const float* base) {
    return _mm512_i32gather_ps(off, base, 4);
}

template <>
inline __m512 gather16_to_f32<int16_t>(__m512i off, const int16_t* base) {
    __m512i raw = _mm512_i32gather_epi32(off, base, 2);
    __m512i sext = _mm512_srai_epi32(_mm512_slli_epi32(raw, 16), 16);
    return _mm512_cvtepi32_ps(sext);
}
#endif

template <typename T>
float calc_orientation_hist_grid_avx512(
    const T* VSLAM_RESTRICT gimg,
    int32_t h,
    int32_t w,
    float ptx,
    float pty,
    float ori_sig,
    float hist[36]
) {
    float temphist[40];
    std::memset(temphist, 0, sizeof(temphist));

#if VSLAM_SIFT_AVX512
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
            gather16_to_f32(_mm512_add_epi32(off, v_one), gimg),
            gather16_to_f32(_mm512_sub_epi32(off, v_one), gimg)
        );
        const __m512 gy = _mm512_sub_ps(
            gather16_to_f32(_mm512_sub_epi32(off, v_w), gimg),
            gather16_to_f32(_mm512_add_epi32(off, v_w), gimg)
        );
        const __m512 mag = _mm512_sqrt_ps(_mm512_fmadd_ps(gx, gx, _mm512_mul_ps(gy, gy)));
        const __m512 wt = _mm512_maskz_mul_ps(in_m, mag, _mm512_load_ps(g_ori_lut.gauss + k));
        const __m512 ori = fast_atan2_deg_avx512(gy, gx);
        __m512i bins = _mm512_cvtps_epi32(_mm512_mul_ps(ori, v_bscale));
        bins = _mm512_mask_sub_epi32(bins, _mm512_cmp_epi32_mask(bins, v_36, _MM_CMPINT_GE), bins, v_36);
        bins = _mm512_mask_add_epi32(bins, _mm512_cmp_epi32_mask(bins, v_zero, _MM_CMPINT_LT), bins, v_36);

        _mm512_store_si512(bin_buf, bins);
        _mm512_store_ps(w_buf, wt);
        for (int32_t lane = 0; lane < 16; ++lane) temphist[bin_buf[lane] + 2] += w_buf[lane];
    }
#else
    for (int32_t k = 0; k < OriGridLUT::N; ++k) {
        const float gw = g_ori_lut.gauss[k];
        if (gw == 0.0f) continue;
        const int32_t ix = static_cast<int32_t>(std::nearbyint(g_ori_lut.du[k] * ori_sig + ptx));
        const int32_t iy = static_cast<int32_t>(std::nearbyint(g_ori_lut.dv[k] * ori_sig + pty));
        if (ix < 1 || ix > w - 2 || iy < 1 || iy > h - 2) continue;
        const int32_t off = iy * w + ix;
        const float gx = static_cast<float>(gimg[off + 1]) - static_cast<float>(gimg[off - 1]);
        const float gy = static_cast<float>(gimg[off - w]) - static_cast<float>(gimg[off + w]);
        const float mag = std::sqrt(gx * gx + gy * gy);
        const float ori = fast_atan2_deg_scalar(gy, gx);
        int32_t bin = static_cast<int32_t>(std::nearbyint(ori * (36.0f / 360.0f)));
        if (bin >= 36) bin -= 36;
        if (bin < 0) bin += 36;
        temphist[bin + 2] += mag * gw;
    }
#endif

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

template <typename T>
void calc_sift_descriptor_grid_avx512(
    const T* VSLAM_RESTRICT gimg,
    int32_t h,
    int32_t w,
    float ptx,
    float pty,
    float ori_deg,
    float scl,
    float* VSLAM_RESTRICT dst_desc
) {
    constexpr int32_t d = 4;
    constexpr int32_t n = 8;
    const float hist_width = 3.0f * scl;
    const float cos_s = std::cos(ori_deg * DEG2RAD) * hist_width;
    const float sin_s = std::sin(ori_deg * DEG2RAD) * hist_width;

    float hist[(d + 2) * (d + 2) * (n + 2)];
    std::memset(hist, 0, sizeof(hist));

#if VSLAM_SIFT_AVX512
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
            gather16_to_f32(_mm512_add_epi32(off, v_one), gimg),
            gather16_to_f32(_mm512_sub_epi32(off, v_one), gimg)
        );
        const __m512 gy = _mm512_sub_ps(
            gather16_to_f32(_mm512_sub_epi32(off, v_w), gimg),
            gather16_to_f32(_mm512_add_epi32(off, v_w), gimg)
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
#else
    const float bpd = static_cast<float>(n) / 360.0f;
    for (int32_t sk = 0; sk < DescGridLUT::N; ++sk) {
        const float vu = g_desc_lut.u[sk];
        const float vv = g_desc_lut.v[sk];
        const float sx = ptx + (vu * cos_s - vv * sin_s);
        const float sy = pty + (vu * sin_s + vv * cos_s);
        const int32_t ix = static_cast<int32_t>(std::nearbyint(sx));
        const int32_t iy = static_cast<int32_t>(std::nearbyint(sy));
        if (ix < 1 || ix > w - 2 || iy < 1 || iy > h - 2) continue;
        const int32_t off = iy * w + ix;
        const float gx = static_cast<float>(gimg[off + 1]) - static_cast<float>(gimg[off - 1]);
        const float gy = static_cast<float>(gimg[off - w]) - static_cast<float>(gimg[off + w]);
        const float m = std::sqrt(gx * gx + gy * gy);
        if (m == 0.0f) continue;
        const float ori = fast_atan2_deg_scalar(gy, gx);
        const float ob = (ori - ori_deg) * bpd;
        const float ob_fl = std::floor(ob);
        const float o1w = ob - ob_fl;
        const float o0w = 1.0f - o1w;
        const int32_t bin0 = static_cast<int32_t>(ob_fl) & 7;

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
#endif

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

#if VSLAM_SIFT_AVX512
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
#else
    float nrm0 = 0.0f;
    for (int32_t k = 0; k < 128; ++k) nrm0 += raw[k] * raw[k];
    const float thr = std::sqrt(nrm0) * 0.2f;
    float nrm1 = 0.0f;
    for (int32_t k = 0; k < 128; ++k) {
        if (raw[k] > thr) raw[k] = thr;
        nrm1 += raw[k] * raw[k];
    }
    const float scl_out = 512.0f / std::max(std::sqrt(nrm1), 1e-6f);
    for (int32_t k = 0; k < 128; ++k) {
        float v = std::nearbyint(raw[k] * scl_out);
        if (v < 0.0f) v = 0.0f;
        else if (v > 255.0f) v = 255.0f;
        dst_desc[k] = v;
    }
#endif
}

struct SiftWorkspace {
    int32_t h = 0, w = 0, first_octave = -1, n_octaves = 0;
    std::vector<int32_t> oct_h, oct_w;
    // Unified Q5 int16_t pyramid across all octaves (2x less DRAM & 32-wide AVX-512BW)
    std::vector<std::vector<std::vector<int16_t>>> gpyr_i16;
    std::vector<std::vector<std::vector<int16_t>>> dogpyr_i16;
    std::vector<int16_t> ring_buf_i16;
    std::vector<int16_t> base_raw_i16;
    std::vector<int16_t> row_tmp_i16;

    void ensure(int32_t ih, int32_t iw, int32_t fo) {
        if (h == ih && w == iw && first_octave == fo) return;
        h = ih; w = iw; first_octave = fo;
        const int32_t base_h = (fo < 0) ? (h * 2) : h;
        const int32_t base_w = (fo < 0) ? (w * 2) : w;
        n_octaves = static_cast<int32_t>(std::round(std::log2(static_cast<double>(std::min(base_h, base_w))))) - 2;
        n_octaves = std::max(1, std::min(n_octaves, 6));
        oct_h.resize(n_octaves);
        oct_w.resize(n_octaves);
        gpyr_i16.resize(n_octaves);
        dogpyr_i16.resize(n_octaves);
        for (int32_t o = 0; o < n_octaves; ++o) {
            oct_h[o] = (o == 0) ? base_h : (oct_h[o - 1] / 2);
            oct_w[o] = (o == 0) ? base_w : (oct_w[o - 1] / 2);
            const size_t npix = static_cast<size_t>(oct_h[o]) * oct_w[o];
            gpyr_i16[o].resize(5);
            dogpyr_i16[o].resize(5);
            for (int32_t i = 0; i < 5; ++i) gpyr_i16[o][i].resize(npix + 32, 0);
            for (int32_t i = 0; i < 5; ++i) dogpyr_i16[o][i].resize(npix + 32, 0);
        }
        ring_buf_i16.resize(static_cast<size_t>(48) * base_w);
        base_raw_i16.resize(static_cast<size_t>(base_h) * base_w + 32);
        if (fo < 0) {
            row_tmp_i16.resize(static_cast<size_t>(h) * base_w + 32);
        }
    }
};

thread_local SiftWorkspace g_ws;

}  // namespace

extern "C" {

VSLAM_EXPORT int32_t extract_sift_grid_avx512(
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

    const float final_thresh = (contrast_threshold / nOctaveLayers) * 255.0f;
    const float pre_thresh = 0.5f * final_thresh;
    const int16_t pre_thresh_i16 = static_cast<int16_t>(std::floor(pre_thresh * static_cast<float>(1 << DOG_SHIFT)));
#if VSLAM_SIFT_AVX512
    const __m512i v_pt16 = _mm512_set1_epi16(pre_thresh_i16);
    const __m512i v_z16 = _mm512_setzero_si512();
#elif defined(__AVX2__)
    const __m256i v_pt16 = _mm256_set1_epi16(pre_thresh_i16);
    const __m256i v_z16 = _mm256_setzero_si256();
#endif

    std::vector<RawExtremum> raw_ext;
    raw_ext.reserve(8192);

    // 1+2. Unified Q7 gpyr + Q9 dogpyr SIMD pyramid and extrema scan
    for (int32_t o = 0; o < n_octaves; ++o) {
        const int32_t oh = g_ws.oct_h[o];
        const int32_t ow = g_ws.oct_w[o];
        const int32_t actual_oct = o + first_octave;
        const float oct_scale = (actual_oct >= 0)
            ? static_cast<float>(1 << actual_oct)
            : (1.0f / static_cast<float>(1 << (-actual_oct)));

        if (o == 0) {
            if (first_octave < 0) {
                upsample_2x_i16_avx512(gray, h, w, g_ws.base_raw_i16.data(), g_ws.row_tmp_i16.data());
                const float sig_diff = std::sqrt(std::max(sigma * sigma - 4.0f * 0.25f, 0.01f));
                gaussian_blur_ring_i16_avx512(g_ws.base_raw_i16.data(), g_ws.gpyr_i16[0][0].data(), nullptr,
                                              g_ws.ring_buf_i16.data(), oh, ow, sig_diff);
            } else {
                const size_t npix = static_cast<size_t>(oh) * ow;
                size_t idx = 0;
#if VSLAM_SIFT_AVX512
                for (; idx + 31 < npix; idx += 32) {
                    __m512i v = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(gray + idx)));
                    _mm512_storeu_si512(g_ws.base_raw_i16.data() + idx, _mm512_slli_epi16(v, Q_SHIFT));
                }
#elif defined(__AVX2__)
                for (; idx + 15 < npix; idx += 16) {
                    __m256i v = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gray + idx)));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(g_ws.base_raw_i16.data() + idx), _mm256_slli_epi16(v, Q_SHIFT));
                }
#endif
                for (; idx < npix; ++idx) g_ws.base_raw_i16[idx] = static_cast<int16_t>(gray[idx] << Q_SHIFT);
                const float sig_diff = std::sqrt(std::max(sigma * sigma - 0.25f, 0.01f));
                gaussian_blur_ring_i16_avx512(g_ws.base_raw_i16.data(), g_ws.gpyr_i16[0][0].data(), nullptr,
                                              g_ws.ring_buf_i16.data(), oh, ow, sig_diff);
            }
        } else {
            downsample_2x_i16(g_ws.gpyr_i16[o - 1][nOctaveLayers].data(), g_ws.oct_h[o - 1], g_ws.oct_w[o - 1],
                              g_ws.gpyr_i16[o][0].data(), oh, ow);
        }

        for (int32_t i = 1; i < nG - 1; ++i) {
            gaussian_blur_ring_i16_avx512(g_ws.gpyr_i16[o][i - 1].data(), g_ws.gpyr_i16[o][i].data(),
                                          g_ws.dogpyr_i16[o][i - 1].data(), g_ws.ring_buf_i16.data(),
                                          oh, ow, sig[i]);
        }
        gaussian_blur_ring_i16_avx512(g_ws.gpyr_i16[o][nG - 2].data(), nullptr,
                                      g_ws.dogpyr_i16[o][nG - 2].data(), g_ws.ring_buf_i16.data(),
                                      oh, ow, sig[nG - 1]);

        if (oh <= 2 * img_border || ow <= 2 * img_border) continue;

        auto refine_candidate = [&](int32_t i, int32_t r, int32_t cx,
                                    const int16_t* prev_img, const int16_t* curr_img, const int16_t* next_img) {
            const int16_t val = curr_img[static_cast<size_t>(r) * ow + cx];
            bool ext = true;
            if (val > 0) {
                for (int32_t dl = -1; dl <= 1 && ext; ++dl) {
                    const int16_t* img_l = (dl == -1) ? prev_img : (dl == 0 ? curr_img : next_img);
                    for (int32_t dr = -1; dr <= 1; ++dr) {
                        const int16_t* rr = img_l + static_cast<size_t>(r + dr) * ow + cx;
                        if (val < rr[-1] || val < rr[0] || val < rr[1]) { ext = false; break; }
                    }
                }
            } else {
                for (int32_t dl = -1; dl <= 1 && ext; ++dl) {
                    const int16_t* img_l = (dl == -1) ? prev_img : (dl == 0 ? curr_img : next_img);
                    for (int32_t dr = -1; dr <= 1; ++dr) {
                        const int16_t* rr = img_l + static_cast<size_t>(r + dr) * ow + cx;
                        if (val > rr[-1] || val > rr[0] || val > rr[1]) { ext = false; break; }
                    }
                }
            }
            if (!ext) return;

            int32_t ri = r, ci = cx, li = i;
            float xc = 0.0f, xr = 0.0f, xi = 0.0f;
            bool converged = false;
            for (int32_t step = 0; step < 5; ++step) {
                const int16_t* dp = g_ws.dogpyr_i16[o][li - 1].data();
                const int16_t* dc = g_ws.dogpyr_i16[o][li].data();
                const int16_t* dn = g_ws.dogpyr_i16[o][li + 1].data();
                const size_t idx = static_cast<size_t>(ri) * ow + ci;
                const float v0 = static_cast<float>(dc[idx]);
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
            if (!converged) return;

            const int16_t* dp = g_ws.dogpyr_i16[o][li - 1].data();
            const int16_t* dc = g_ws.dogpyr_i16[o][li].data();
            const int16_t* dn = g_ws.dogpyr_i16[o][li + 1].data();
            const size_t idx = static_cast<size_t>(ri) * ow + ci;
            const float v0 = static_cast<float>(dc[idx]);
            const float dx = 0.5f * (dc[idx + 1] - dc[idx - 1]);
            const float dy = 0.5f * (dc[idx + ow] - dc[idx - ow]);
            const float ds = 0.5f * (dn[idx] - dp[idx]);
            const float contr = (v0 + 0.5f * (dx * xc + dy * xr + ds * xi)) * INV_DOG_Q;
            if (std::abs(contr) < final_thresh) return;

            const float dxx = dc[idx + 1] + dc[idx - 1] - 2.0f * v0;
            const float dyy = dc[idx + ow] + dc[idx - ow] - 2.0f * v0;
            const float dxy = 0.25f * (dc[idx + ow + 1] - dc[idx + ow - 1] - dc[idx - ow + 1] + dc[idx - ow - 1]);
            const float tr = dxx + dyy;
            const float det = dxx * dyy - dxy * dxy;
            if (det <= 0.0f || tr * tr * edgeThreshold >= (edgeThreshold + 1.0f) * (edgeThreshold + 1.0f) * det) {
                return;
            }

            const float ptx_img = (static_cast<float>(ci) + xc) * oct_scale;
            const float pty_img = (static_cast<float>(ri) + xr) * oct_scale;
            if (ptx_img < 0.0f || ptx_img >= static_cast<float>(w) ||
                pty_img < 0.0f || pty_img >= static_cast<float>(h)) {
                return;
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
        };

        for (int32_t i = 1; i <= nOctaveLayers; ++i) {
            const int16_t* prev_img = g_ws.dogpyr_i16[o][i - 1].data();
            const int16_t* curr_img = g_ws.dogpyr_i16[o][i].data();
            const int16_t* next_img = g_ws.dogpyr_i16[o][i + 1].data();

            for (int32_t r = img_border; r < oh - img_border; ++r) {
                const size_t row_off = static_cast<size_t>(r) * ow;
                const int16_t* rc = curr_img + row_off;
                const int16_t* rp = curr_img + row_off - ow;
                const int16_t* rn = curr_img + row_off + ow;
                const int16_t* p_row = prev_img + row_off;
                const int16_t* n_row = next_img + row_off;

                int32_t c = img_border;
#if VSLAM_SIFT_AVX512
                for (; c + 31 < ow - img_border; c += 32) {
                    const __m512i vc = _mm512_loadu_si512(rc + c);
                    const __m512i av = _mm512_abs_epi16(vc);
                    __mmask32 cand = _mm512_cmpgt_epi16_mask(av, v_pt16);
                    if (!cand) continue;

                    const __m512i vl = _mm512_loadu_si512(rc + c - 1);
                    const __m512i vr = _mm512_loadu_si512(rc + c + 1);
                    const __m512i vu = _mm512_loadu_si512(rp + c);
                    const __m512i vd = _mm512_loadu_si512(rn + c);
                    const __m512i vs_p = _mm512_loadu_si512(p_row + c);
                    const __m512i vs_n = _mm512_loadu_si512(n_row + c);

                    const __m512i nmax = _mm512_max_epi16(
                        _mm512_max_epi16(_mm512_max_epi16(vl, vr), _mm512_max_epi16(vu, vd)),
                        _mm512_max_epi16(vs_p, vs_n)
                    );
                    const __m512i nmin = _mm512_min_epi16(
                        _mm512_min_epi16(_mm512_min_epi16(vl, vr), _mm512_min_epi16(vu, vd)),
                        _mm512_min_epi16(vs_p, vs_n)
                    );
                    const __mmask32 is_pos = _mm512_cmpgt_epi16_mask(vc, v_z16);
                    const __mmask32 ok_max = is_pos & _mm512_cmpge_epi16_mask(vc, nmax);
                    const __mmask32 ok_min = (~is_pos) & _mm512_cmple_epi16_mask(vc, nmin);
                    cand &= (ok_max | ok_min);
                    if (!cand) continue;

                    uint32_t bits = static_cast<uint32_t>(cand);
                    while (bits) {
                        const int32_t lane = vslam_ctz32(bits);
                        bits &= bits - 1;
                        refine_candidate(i, r, c + lane, prev_img, curr_img, next_img);
                    }
                }
#elif defined(__AVX2__)
                for (; c + 15 < ow - img_border; c += 16) {
                    const __m256i vc = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rc + c));
                    const __m256i av = _mm256_abs_epi16(vc);
                    const __m256i cand_v = _mm256_cmpgt_epi16(av, v_pt16);
                    if (_mm256_testz_si256(cand_v, cand_v)) continue;

                    const __m256i vl = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rc + c - 1));
                    const __m256i vr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rc + c + 1));
                    const __m256i vu = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rp + c));
                    const __m256i vd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rn + c));
                    const __m256i vs_p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p_row + c));
                    const __m256i vs_n = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(n_row + c));

                    const __m256i nmax = _mm256_max_epi16(
                        _mm256_max_epi16(_mm256_max_epi16(vl, vr), _mm256_max_epi16(vu, vd)),
                        _mm256_max_epi16(vs_p, vs_n)
                    );
                    const __m256i nmin = _mm256_min_epi16(
                        _mm256_min_epi16(_mm256_min_epi16(vl, vr), _mm256_min_epi16(vu, vd)),
                        _mm256_min_epi16(vs_p, vs_n)
                    );
                    const __m256i is_pos = _mm256_cmpgt_epi16(vc, v_z16);
                    // vc >= nmax  <=> !(nmax > vc)
                    const __m256i ok_max = _mm256_andnot_si256(_mm256_cmpgt_epi16(nmax, vc), is_pos);
                    // vc <= nmin  <=> !(vc > nmin)
                    const __m256i is_neg = _mm256_cmpgt_epi16(v_z16, vc);
                    const __m256i ok_min = _mm256_andnot_si256(_mm256_cmpgt_epi16(vc, nmin), is_neg);
                    const __m256i ok_v = _mm256_and_si256(cand_v, _mm256_or_si256(ok_max, ok_min));
                    uint32_t bmask = static_cast<uint32_t>(_mm256_movemask_epi8(ok_v)) & 0x55555555u;
                    while (bmask) {
                        const int32_t lane = vslam_ctz32(bmask) >> 1;
                        bmask &= bmask - 1;
                        refine_candidate(i, r, c + lane, prev_img, curr_img, next_img);
                    }
                }
#endif
                for (; c < ow - img_border; ++c) {
                    const int16_t val = rc[c];
                    const int16_t aval = (val < 0) ? static_cast<int16_t>(-val) : val;
                    if (aval <= pre_thresh_i16) continue;
                    if (val > 0) {
                        if (val < rc[c - 1] || val < rc[c + 1] || val < rp[c] || val < rn[c] ||
                            val < p_row[c] || val < n_row[c]) continue;
                    } else {
                        if (val > rc[c - 1] || val > rc[c + 1] || val > rp[c] || val > rn[c] ||
                            val > p_row[c] || val > n_row[c]) continue;
                    }
                    refine_candidate(i, r, c, prev_img, curr_img, next_img);
                }
            }
        }
    }

    const int32_t total_ext = static_cast<int32_t>(raw_ext.size());
    if (total_ext == 0) {
        *out_count = 0;
        return 1;
    }

    // 3. Response-cap and spatially pre-bucket raw extrema before orientation calculation
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

    // Spatially pre-select up to target_features extrema (plus 15% margin) before computing
    // orientation histograms, avoiding wasted orientation work on dense-cell surplus extrema.
    const int32_t pre_ori_cap = std::min(static_cast<int32_t>(ext_order.size()),
                                         std::max(target_features, static_cast<int32_t>(target_features * 1.15)));
    if (static_cast<int32_t>(ext_order.size()) > pre_ori_cap) {
        const int32_t cols = std::max(1, grid_cols);
        const int32_t rows = std::max(1, grid_rows);
        const int32_t n_cells = cols * rows;
        const int32_t cell_quota = std::max(1, static_cast<int32_t>(std::ceil(static_cast<double>(target_features) / n_cells)));
        std::vector<int32_t> cell_counts(n_cells, 0);
        std::vector<uint8_t> picked(ext_order.size(), 0);
        std::vector<int32_t> bucketed_ext;
        bucketed_ext.reserve(pre_ori_cap);
        const float sx = static_cast<float>(cols) / static_cast<float>(std::max(w, 1));
        const float sy = static_cast<float>(rows) / static_cast<float>(std::max(h, 1));
        for (size_t pos = 0; pos < ext_order.size() && static_cast<int32_t>(bucketed_ext.size()) < pre_ori_cap; ++pos) {
            const RawExtremum& ex = raw_ext[ext_order[pos]];
            int32_t gx = std::max(0, std::min(cols - 1, static_cast<int32_t>(ex.x * sx)));
            int32_t gy = std::max(0, std::min(rows - 1, static_cast<int32_t>(ex.y * sy)));
            int32_t cid = gy * cols + gx;
            if (cell_counts[cid] < cell_quota) {
                cell_counts[cid]++;
                picked[pos] = 1;
                bucketed_ext.push_back(ext_order[pos]);
            }
        }
        for (size_t pos = 0; pos < ext_order.size() && static_cast<int32_t>(bucketed_ext.size()) < pre_ori_cap; ++pos) {
            if (!picked[pos]) bucketed_ext.push_back(ext_order[pos]);
        }
        ext_order.swap(bucketed_ext);
    }

    std::vector<SiftKeypoint> kps;
    kps.reserve(raw_cap + 512);
    for (int32_t ei : ext_order) {
        const RawExtremum& ex = raw_ext[ei];
        float ohist[36];
        const float inv_oct_scale = (ex.octave_idx >= 0)
            ? (1.0f / static_cast<float>(1 << ex.octave_idx))
            : static_cast<float>(1 << (-ex.octave_idx));
        const float omax = calc_orientation_hist_grid_avx512<int16_t>(
            g_ws.gpyr_i16[ex.pyr_o][ex.layer].data(),
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
        const float adj_ori = (ori >= 360.0f ? 0.0f : ori);
        calc_sift_descriptor_grid_avx512<int16_t>(
            g_ws.gpyr_i16[o][kp.layer].data(),
            g_ws.oct_h[o], g_ws.oct_w[o],
            kp.x * inv_oct_scale, kp.y * inv_oct_scale,
            adj_ori, kp.scl_octv,
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
