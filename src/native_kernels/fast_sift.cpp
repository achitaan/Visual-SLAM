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

inline void blur_row_h_avx512(const float* srow, float* trow, int32_t w, const float* hw, int32_t rad) {
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

// L2-cache-fused ring-buffer separable 2D Gaussian blur!
// Never allocates or streams a full-frame `tmp` buffer to DRAM: keeps only a tiny
// (2*rad + 1) rolling window of horizontally-blurred rows (~200 KB) 100% inside L2 cache!
void gaussian_blur_ring_avx512(
    const float* src,
    float* dst,
    float* ring_buf,  // [64 * w]
    int32_t h,
    int32_t w,
    float sigma
) {
    float hw[25];
    int32_t rad = 0;
    make_gaussian_kernel(sigma, hw, rad);
    const int32_t ring_sz = 2 * rad + 2;  // power-friendly rolling slot count (<= 42)
    auto ring_ptr = [&](int32_t ry) -> float* {
        int32_t clamped = (ry < 0) ? 0 : (ry >= h ? h - 1 : ry);
        return ring_buf + static_cast<size_t>(clamped % ring_sz) * w;
    };

    // Prime horizontal blur for rows [0 .. min(rad, h-1)]
    int32_t h_done = -1;
    const int32_t init_end = std::min(rad, h - 1);
    for (int32_t ry = 0; ry <= init_end; ++ry) {
        blur_row_h_avx512(src + static_cast<size_t>(ry) * w, ring_ptr(ry), w, hw, rad);
        h_done = ry;
    }

    const __m512 w0 = _mm512_set1_ps(hw[0]);
    for (int32_t y = 0; y < h; ++y) {
        const int32_t need_ry = std::min(y + rad, h - 1);
        while (h_done < need_ry) {
            ++h_done;
            blur_row_h_avx512(src + static_cast<size_t>(h_done) * w, ring_ptr(h_done), w, hw, rad);
        }

        const float* r0 = ring_ptr(y);
        const float* r_up[25];
        const float* r_dn[25];
        for (int32_t k = 1; k <= rad; ++k) {
            r_up[k] = ring_ptr(y - k);
            r_dn[k] = ring_ptr(y + k);
        }
        float* drow = dst + static_cast<size_t>(y) * w;

        int32_t x = 0;
        for (; x + 31 < w; x += 32) {
            __m512 a0 = _mm512_mul_ps(w0, _mm512_loadu_ps(r0 + x));
            __m512 a1 = _mm512_mul_ps(w0, _mm512_loadu_ps(r0 + x + 16));
            for (int32_t k = 1; k <= rad; ++k) {
                __m512 wk = _mm512_set1_ps(hw[k]);
                a0 = _mm512_fmadd_ps(wk, _mm512_add_ps(_mm512_loadu_ps(r_up[k] + x), _mm512_loadu_ps(r_dn[k] + x)), a0);
                a1 = _mm512_fmadd_ps(wk, _mm512_add_ps(_mm512_loadu_ps(r_up[k] + x + 16), _mm512_loadu_ps(r_dn[k] + x + 16)), a1);
            }
            _mm512_storeu_ps(drow + x, a0);
            _mm512_storeu_ps(drow + x + 16, a1);
        }
        for (; x + 15 < w; x += 16) {
            __m512 a0 = _mm512_mul_ps(w0, _mm512_loadu_ps(r0 + x));
            for (int32_t k = 1; k <= rad; ++k) {
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(hw[k]),
                                     _mm512_add_ps(_mm512_loadu_ps(r_up[k] + x), _mm512_loadu_ps(r_dn[k] + x)), a0);
            }
            _mm512_storeu_ps(drow + x, a0);
        }
        for (; x < w; ++x) {
            float acc = hw[0] * r0[x];
            for (int32_t k = 1; k <= rad; ++k) acc += hw[k] * (r_up[k][x] + r_dn[k][x]);
            drow[x] = acc;
        }
    }
}

void upsample_2x_avx512(const uint8_t* src, int32_t h, int32_t w, float* dst, float* row_tmp) {
    const int32_t dh = h * 2;
    const int32_t dw = w * 2;
    for (int32_t y = 0; y < h; ++y) {
        const uint8_t* sr = src + static_cast<size_t>(y) * w;
        float* tr = row_tmp + static_cast<size_t>(y) * dw;
        tr[0] = static_cast<float>(sr[0]);
        tr[1] = 0.75f * sr[0] + 0.25f * sr[std::min(1, w - 1)];
        for (int32_t x = 1; x < w - 1; ++x) {
            const float c = static_cast<float>(sr[x]);
            tr[2 * x]     = 0.75f * c + 0.25f * static_cast<float>(sr[x - 1]);
            tr[2 * x + 1] = 0.75f * c + 0.25f * static_cast<float>(sr[x + 1]);
        }
        if (w > 1) {
            tr[2 * (w - 1)]     = 0.75f * sr[w - 1] + 0.25f * sr[w - 2];
            tr[2 * (w - 1) + 1] = static_cast<float>(sr[w - 1]);
        }
    }
    const __m512 v075 = _mm512_set1_ps(0.75f);
    const __m512 v025 = _mm512_set1_ps(0.25f);
    for (int32_t dy = 0; dy < dh; ++dy) {
        const int32_t sy = dy >> 1;
        const int32_t sy_other = (dy & 1) ? std::min(sy + 1, h - 1) : std::max(sy - 1, 0);
        const float* r_main = row_tmp + static_cast<size_t>(sy) * dw;
        const float* r_oth  = row_tmp + static_cast<size_t>(sy_other) * dw;
        float* dr = dst + static_cast<size_t>(dy) * dw;
        int32_t x = 0;
        for (; x + 15 < dw; x += 16) {
            __m512 v = _mm512_fmadd_ps(v075, _mm512_loadu_ps(r_main + x),
                                       _mm512_mul_ps(v025, _mm512_loadu_ps(r_oth + x)));
            _mm512_storeu_ps(dr + x, v);
        }
        for (; x < dw; ++x) dr[x] = 0.75f * r_main[x] + 0.25f * r_oth[x];
    }
}

void downsample_2x(const float* src, int32_t sh, int32_t sw, float* dst, int32_t dh, int32_t dw) {
    for (int32_t y = 0; y < dh; ++y) {
        const float* sr = src + static_cast<size_t>(y * 2) * sw;
        float* dr = dst + static_cast<size_t>(y) * dw;
        for (int32_t x = 0; x < dw; ++x) dr[x] = sr[x * 2];
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

// Precomputed constant 16x16 = 256-sample rotated SIFT descriptor grid!
// Each sample sits at canonical descriptor coordinates (r_rot, c_rot) in [-2, 2]^2
// corresponding to 4x4 spatial bins x 4x4 sub-samples per bin.
// Gaussian weights and 4 bilinear spatial bin weights are 100% keypoint-independent!
struct DescGridLUT {
    static constexpr int32_t N = 256;
    alignas(64) float u[N];        // col offset in hist_width units: [-1.875 .. +1.875]
    alignas(64) float v[N];        // row offset in hist_width units: [-1.875 .. +1.875]
    alignas(64) float gauss[N];    // exp(-(u^2 + v^2) / 8.0)
    int16_t idx00[N], idx01[N], idx10[N], idx11[N];  // base indices into 6x6x10 padded hist
    float w00[N], w01[N], w10[N], w11[N];            // gauss * bilinear spatial weights

    DescGridLUT() {
        constexpr int32_t d = 4;
        constexpr int32_t n = 8;
        int32_t k = 0;
        for (int32_t ri = 0; ri < 16; ++ri) {
            const float r_rot = -1.875f + 0.25f * static_cast<float>(ri);
            const float rb = r_rot + 1.5f;  // [-0.375 .. 3.375]
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

// Precomputed 11x11 = 121-sample scaled grid for 36-bin orientation histogram (padded to 128).
struct OriGridLUT {
    static constexpr int32_t N = 128;
    static constexpr int32_t N_VALID = 121;
    alignas(64) float du[N];
    alignas(64) float dv[N];
    alignas(64) float gauss[N];

    OriGridLUT() {
        int32_t k = 0;
        // Span [-3.0, +3.0] in units of ori_sigma = 1.5 * scl_octv (step 0.6 * ori_sigma)
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

// Ultra-fast AVX-512 36-bin orientation histogram using fixed 121-sample scaled grid (8 AVX-512 iterations!).
float calc_orientation_hist_grid_avx512(
    const float* gimg,
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
        const __m512i bins = _mm512_cvtps_epi32(_mm512_mul_ps(ori, v_bscale));

        _mm512_store_si512(bin_buf, bins);
        _mm512_store_ps(w_buf, wt);
        for (int32_t lane = 0; lane < 16; ++lane) {
            int32_t b = bin_buf[lane];
            if (b >= 36) b -= 36;
            if (b < 0) b += 36;
            temphist[b + 2] += w_buf[lane];
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

// Ultra-fast AVX-512 128-dim SIFT descriptor using the precomputed 256-sample rotated grid!
// Exactly 16 AVX-512 gather+grad+atan2 iterations per keypoint regardless of scale!
void calc_sift_descriptor_grid_avx512(
    const float* gimg,
    int32_t h,
    int32_t w,
    float ptx,
    float pty,
    float ori_deg,
    float scl,
    float* dst_desc
) {
    constexpr int32_t d = 4;
    constexpr int32_t n = 8;
    const float hist_width = 3.0f * scl;
    const float cos_s = std::cos(ori_deg * DEG2RAD) * hist_width;
    const float sin_s = std::sin(ori_deg * DEG2RAD) * hist_width;

    // Coordinate transform from canonical (c_rot=u, r_rot=v) to image (x, y):
    // In standard SIFT: c_rot = (j*cos + i*sin)/hw, r_rot = (i*cos - j*sin)/hw
    // Inverse: j = hw * (u*cos - v*sin), i = hw * (u*sin + v*cos)
    const __m512 v_ptx = _mm512_set1_ps(ptx);
    const __m512 v_pty = _mm512_set1_ps(pty);
    const __m512 v_cos = _mm512_set1_ps(cos_s);
    const __m512 v_sin = _mm512_set1_ps(sin_s);
    const __m512 v_ori = _mm512_set1_ps(ori_deg);
    const __m512 v_bpd = _mm512_set1_ps(static_cast<float>(n) / 360.0f);
    const __m512i v_one = _mm512_set1_epi32(1);
    const __m512i v_w_m2 = _mm512_set1_epi32(w - 2);
    const __m512i v_h_m2 = _mm512_set1_epi32(h - 2);
    const __m512i v_w = _mm512_set1_epi32(w);

    float hist[(d + 2) * (d + 2) * (n + 2)];
    std::memset(hist, 0, sizeof(hist));

    alignas(64) float buf_mag[16];
    alignas(64) float buf_ob[16];

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

        _mm512_store_ps(buf_mag, mag);
        _mm512_store_ps(buf_ob, ob);

        for (int32_t lane = 0; lane < 16; ++lane) {
            const float m = buf_mag[lane];
            if (m == 0.0f) continue;
            const int32_t sk = k + lane;
            const float obf = buf_ob[lane];
            int32_t o0 = static_cast<int32_t>(std::floor(obf));
            const float dob = obf - static_cast<float>(o0);
            const float o1w = dob;
            const float o0w = 1.0f - dob;
            o0 &= 7;  // since n == 8, ((o0 % 8) + 8) % 8 == o0 & 7!

            const float m00 = m * g_desc_lut.w00[sk];
            const float m01 = m * g_desc_lut.w01[sk];
            const float m10 = m * g_desc_lut.w10[sk];
            const float m11 = m * g_desc_lut.w11[sk];

            const int32_t p00 = g_desc_lut.idx00[sk] + o0;
            const int32_t p01 = g_desc_lut.idx01[sk] + o0;
            const int32_t p10 = g_desc_lut.idx10[sk] + o0;
            const int32_t p11 = g_desc_lut.idx11[sk] + o0;

            hist[p00] += m00 * o0w; hist[p00 + 1] += m00 * o1w;
            hist[p01] += m01 * o0w; hist[p01 + 1] += m01 * o1w;
            hist[p10] += m10 * o0w; hist[p10 + 1] += m10 * o1w;
            hist[p11] += m11 * o0w; hist[p11 + 1] += m11 * o1w;
        }
    }

    float raw[128];
    float nrm2 = 0.0f;
    for (int32_t i = 0; i < d; ++i) {
        for (int32_t j = 0; j < d; ++j) {
            const int32_t idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);
            hist[idx] += hist[idx + n];
            hist[idx + 1] += hist[idx + n + 1];
            for (int32_t k = 0; k < n; ++k) {
                float val = hist[idx + k];
                raw[(i * d + j) * n + k] = val;
                nrm2 += val * val;
            }
        }
    }
    const float thr = std::sqrt(nrm2) * 0.2f;
    float nrm2_clipped = 0.0f;
    for (int32_t k = 0; k < 128; ++k) {
        float val = std::min(raw[k], thr);
        raw[k] = val;
        nrm2_clipped += val * val;
    }
    const float scale = 512.0f / std::max(std::sqrt(nrm2_clipped), 1e-6f);
    for (int32_t k = 0; k < 128; ++k) {
        int32_t iv = static_cast<int32_t>(std::round(raw[k] * scale));
        if (iv < 0) iv = 0;
        else if (iv > 255) iv = 255;
        dst_desc[k] = static_cast<float>(iv);
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
            gpyr[o].resize(6);
            dogpyr[o].resize(5);
            for (int32_t i = 0; i < 6; ++i) gpyr[o][i].resize(npix);
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
    constexpr int32_t nD = nOctaveLayers + 2;
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

    // 1. L2-fused Gaussian + DoG pyramid
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
                for (size_t idx = 0; idx < npix; ++idx) g_ws.base_raw[idx] = static_cast<float>(gray[idx]);
                sig_diff = std::sqrt(std::max(sigma * sigma - 0.25f, 0.01f));
            }
            gaussian_blur_ring_avx512(g_ws.base_raw.data(), g_ws.gpyr[0][0].data(), g_ws.ring_buf.data(), oh, ow, sig_diff);
        } else {
            downsample_2x(g_ws.gpyr[o - 1][nOctaveLayers].data(), g_ws.oct_h[o - 1], g_ws.oct_w[o - 1],
                          g_ws.gpyr[o][0].data(), oh, ow);
        }

        for (int32_t i = 1; i < nG; ++i) {
            gaussian_blur_ring_avx512(g_ws.gpyr[o][i - 1].data(), g_ws.gpyr[o][i].data(), g_ws.ring_buf.data(),
                                      oh, ow, sig[i]);
        }

        for (int32_t i = 0; i < nD; ++i) {
            const float* g0 = g_ws.gpyr[o][i].data();
            const float* g1 = g_ws.gpyr[o][i + 1].data();
            float* dg = g_ws.dogpyr[o][i].data();
            size_t idx = 0;
            for (; idx + 15 < npix; idx += 16) {
                _mm512_storeu_ps(dg + idx, _mm512_sub_ps(_mm512_loadu_ps(g1 + idx), _mm512_loadu_ps(g0 + idx)));
            }
            for (; idx < npix; ++idx) dg[idx] = g1[idx] - g0[idx];
        }
    }

    // 2. AVX-512 3x3x3 Extrema detection (with 6-neighbor + 2-scale SIMD pre-filter)
    const float pre_thresh = std::floor(0.5f * contrast_threshold / nOctaveLayers * 255.0f);
    const float final_thresh = (contrast_threshold / nOctaveLayers) * 255.0f;
    const __m512 v_pre_thresh = _mm512_set1_ps(pre_thresh);
    const __m512 v_abs_mask = _mm512_set1_ps(-0.0f);

    std::vector<RawExtremum> raw_ext;
    raw_ext.reserve(8192);

    for (int32_t o = 0; o < n_octaves; ++o) {
        const int32_t oh = g_ws.oct_h[o];
        const int32_t ow = g_ws.oct_w[o];
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

                    // 6-neighbor SIMD filter (left, right, up, down, prev_scale, next_scale)
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
                    const __mmask16 is_pos = _mm512_cmp_ps_mask(vc, _mm512_setzero_ps(), _CMP_GT_OQ);
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
