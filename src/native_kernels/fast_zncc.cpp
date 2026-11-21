#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

// Rejection reason codes matching src/stereo_depth.py strings
enum RejectReason : int32_t {
    REASON_OK = 0,
    REASON_PATCH_BOUNDS = 1,
    REASON_LOW_TEXTURE = 2,
    REASON_SEARCH_BOUNDS = 3,
    REASON_CORRELATION = 4,
    REASON_INSUFFICIENT_ALTERNATIVES = 5,
    REASON_NEAR_TIE = 6,
    REASON_AMBIGUOUS = 7,
    REASON_UNREFINABLE_PEAK = 8,
    REASON_FLAT_PEAK = 9,
    REASON_UNSTABLE_PEAK = 10,
    REASON_REFINED_BOUNDS = 11,
    REASON_LOW_TARGET_TEXTURE = 12,
    REASON_REFINED_CORRELATION = 13,
    REASON_REVERSE_PREFIX_OFFSET = 100,  // reverse_<reason>
    REASON_ROUND_TRIP = 200,
};

// Exact bilinear sampling with BORDER_REPLICATE matching cv::remap(INTER_LINEAR, BORDER_REPLICATE).
inline float sample_bilinear_scalar(
    const float* img, int32_t h, int32_t w, float x, float y
) {
    int32_t ix0 = static_cast<int32_t>(std::floor(x));
    int32_t iy0 = static_cast<int32_t>(std::floor(y));
    const float fx = x - static_cast<float>(ix0);
    const float fy = y - static_cast<float>(iy0);
    int32_t ix1 = ix0 + 1;
    int32_t iy1 = iy0 + 1;
    ix0 = (ix0 < 0) ? 0 : (ix0 >= w ? w - 1 : ix0);
    ix1 = (ix1 < 0) ? 0 : (ix1 >= w ? w - 1 : ix1);
    iy0 = (iy0 < 0) ? 0 : (iy0 >= h ? h - 1 : iy0);
    iy1 = (iy1 < 0) ? 0 : (iy1 >= h ? h - 1 : iy1);
    const float* r0 = img + static_cast<size_t>(iy0) * w;
    const float* r1 = img + static_cast<size_t>(iy1) * w;
    const float v0 = r0[ix0] + fx * (r0[ix1] - r0[ix0]);
    const float v1 = r1[ix0] + fx * (r1[ix1] - r1[ix0]);
    return v0 + fy * (v1 - v0);
}

// Fast AVX-512 horizontal bilinear sampling of a contiguous row of length `len`
// at coordinates (x0 + 0, y), (x0 + 1, y), ..., (x0 + len - 1, y).
// Since x increments by exactly +1.0 per sample, the fractional part `fx` and `fy`
// are CONSTANT across the entire row! That reduces bilinear sampling to 4 aligned/unaligned
// AVX-512 vector loads + 3 FMA instructions for 16 pixels at once!
inline void sample_rect_fast(
    const float* img,
    int32_t h,
    int32_t w,
    float cx,
    float cy,
    int32_t out_w,
    int32_t out_h,
    float* dst  // [out_h, out_w]
) {
    const float x0 = cx - 0.5f * static_cast<float>(out_w - 1);
    const float y0 = cy - 0.5f * static_cast<float>(out_h - 1);
    const int32_t ix0_base = static_cast<int32_t>(std::floor(x0));
    const int32_t iy0_base = static_cast<int32_t>(std::floor(y0));
    const float fx = x0 - static_cast<float>(ix0_base);
    const float fy = y0 - static_cast<float>(iy0_base);
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = fx * (1.0f - fy);
    const float w10 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    const __m512 vw00 = _mm512_set1_ps(w00);
    const __m512 vw01 = _mm512_set1_ps(w01);
    const __m512 vw10 = _mm512_set1_ps(w10);
    const __m512 vw11 = _mm512_set1_ps(w11);

    const bool x_interior = (ix0_base >= 0) && (ix0_base + out_w < w);

    for (int32_t dy = 0; dy < out_h; ++dy) {
        int32_t iy0 = iy0_base + dy;
        int32_t iy1 = iy0 + 1;
        iy0 = (iy0 < 0) ? 0 : (iy0 >= h ? h - 1 : iy0);
        iy1 = (iy1 < 0) ? 0 : (iy1 >= h ? h - 1 : iy1);
        const float* r0 = img + static_cast<size_t>(iy0) * w;
        const float* r1 = img + static_cast<size_t>(iy1) * w;
        float* out_row = dst + static_cast<size_t>(dy) * out_w;

        if (x_interior) {
            const float* p00 = r0 + ix0_base;
            const float* p01 = p00 + 1;
            const float* p10 = r1 + ix0_base;
            const float* p11 = p10 + 1;
            int32_t dx = 0;
            for (; dx + 15 < out_w; dx += 16) {
                __m512 v = _mm512_mul_ps(vw00, _mm512_loadu_ps(p00 + dx));
                v = _mm512_fmadd_ps(vw01, _mm512_loadu_ps(p01 + dx), v);
                v = _mm512_fmadd_ps(vw10, _mm512_loadu_ps(p10 + dx), v);
                v = _mm512_fmadd_ps(vw11, _mm512_loadu_ps(p11 + dx), v);
                _mm512_storeu_ps(out_row + dx, v);
            }
            for (; dx < out_w; ++dx) {
                out_row[dx] = w00 * p00[dx] + w01 * p01[dx] + w10 * p10[dx] + w11 * p11[dx];
            }
        } else {
            for (int32_t dx = 0; dx < out_w; ++dx) {
                int32_t ix0 = ix0_base + dx;
                int32_t ix1 = ix0 + 1;
                ix0 = (ix0 < 0) ? 0 : (ix0 >= w ? w - 1 : ix0);
                ix1 = (ix1 < 0) ? 0 : (ix1 >= w ? w - 1 : ix1);
                out_row[dx] = w00 * r0[ix0] + w01 * r0[ix1] + w10 * r1[ix0] + w11 * r1[ix1];
            }
        }
    }
}

// Single epipolar ZNCC search matching _batch_search in src/stereo_depth.py.
// Returns REASON_OK and sets *out_x on success, or returns reject reason code.
int32_t zncc_search_one(
    const float* src_img,
    const float* tgt_img,
    int32_t h,
    int32_t w,
    double x,
    double y,
    double direction,
    int32_t patch_size,
    int32_t max_disp,
    double min_corr,
    double uniq_ratio,
    double min_tex_std,
    double disp_lo,
    double disp_hi,
    float* patch_buf,    // [patch_size * patch_size]
    float* strip_buf,    // [patch_size * (patch_size + max_disp)]
    double* scores_buf,  // [max_disp + 1]
    double* out_x
) {
    const int32_t radius = patch_size / 2;
    const double patch_area = static_cast<double>(patch_size * patch_size);
    const double tol = 8.0 * static_cast<double>(std::numeric_limits<float>::epsilon());
    const double flat_var_floor = 1e-6 * patch_area;

    if (!(x >= radius && x <= (w - 1 - radius) && y >= radius && y <= (h - 1 - radius))) {
        return REASON_PATCH_BOUNDS;
    }

    sample_rect_fast(src_img, h, w, static_cast<float>(x), static_cast<float>(y),
                     patch_size, patch_size, patch_buf);

    const int32_t npix = patch_size * patch_size;
    double p_sum = 0.0;
    for (int32_t k = 0; k < npix; ++k) {
        p_sum += static_cast<double>(patch_buf[k]);
    }
    const double p_mean = p_sum / patch_area;
    double p_ss = 0.0;
    for (int32_t k = 0; k < npix; ++k) {
        const double zm = static_cast<double>(patch_buf[k]) - p_mean;
        patch_buf[k] = static_cast<float>(zm);  // store zero-mean patch in float32 for AVX-512 FMA
        p_ss += zm * zm;
    }
    if (std::sqrt(p_ss / patch_area) < min_tex_std) {
        return REASON_LOW_TEXTURE;
    }

    const int32_t strip_w = patch_size + max_disp;
    const int32_t n_scores = max_disp + 1;
    const double strip_center = x + direction * static_cast<double>(max_disp) * 0.5;
    sample_rect_fast(tgt_img, h, w, static_cast<float>(strip_center), static_cast<float>(y),
                     strip_w, patch_size, strip_buf);

    // Column sums and squared sums of strip (in float64 for exact variance without cancellation)
    double col_s[256];
    double col_sq[256];
    for (int32_t c = 0; c < strip_w; ++c) {
        col_s[c] = 0.0;
        col_sq[c] = 0.0;
    }
    for (int32_t r = 0; r < patch_size; ++r) {
        const float* srow = strip_buf + static_cast<size_t>(r) * strip_w;
        for (int32_t c = 0; c < strip_w; ++c) {
            const double v = static_cast<double>(srow[c]);
            col_s[c] += v;
            col_sq[c] += v * v;
        }
    }

    // Prefix sums for sliding window sum & sum-of-squares
    double pref_s[257];
    double pref_sq[257];
    pref_s[0] = 0.0;
    pref_sq[0] = 0.0;
    for (int32_t c = 0; c < strip_w; ++c) {
        pref_s[c + 1] = pref_s[c] + col_s[c];
        pref_sq[c + 1] = pref_sq[c] + col_sq[c];
    }

    // AVX-512 cross-correlation: num[s] = sum_{r,c} p_zm[r,c] * strip[r, c + s] for s=0..n_scores-1
    alignas(64) float num_f[256];
    int32_t s = 0;
    for (; s + 15 < n_scores; s += 16) {
        __m512 acc = _mm512_setzero_ps();
        for (int32_t r = 0; r < patch_size; ++r) {
            const float* prow = patch_buf + static_cast<size_t>(r) * patch_size;
            const float* srow = strip_buf + static_cast<size_t>(r) * strip_w + s;
            for (int32_t c = 0; c < patch_size; ++c) {
                acc = _mm512_fmadd_ps(_mm512_set1_ps(prow[c]), _mm512_loadu_ps(srow + c), acc);
            }
        }
        _mm512_store_ps(num_f + s, acc);
    }
    if (s < n_scores) {
        const __mmask16 rem = static_cast<__mmask16>((1u << (n_scores - s)) - 1u);
        __m512 acc = _mm512_setzero_ps();
        for (int32_t r = 0; r < patch_size; ++r) {
            const float* prow = patch_buf + static_cast<size_t>(r) * patch_size;
            const float* srow = strip_buf + static_cast<size_t>(r) * strip_w + s;
            for (int32_t c = 0; c < patch_size; ++c) {
                acc = _mm512_fmadd_ps(
                    _mm512_set1_ps(prow[c]),
                    _mm512_maskz_loadu_ps(rem, srow + c),
                    acc
                );
            }
        }
        _mm512_mask_storeu_ps(num_f + s, rem, acc);
    }

    const double pos_base = strip_center - static_cast<double>(max_disp) * 0.5;
    int32_t best = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    uint8_t valid_mask[256];

    for (int32_t k = 0; k < n_scores; ++k) {
        const double ws = pref_s[k + patch_size] - pref_s[k];
        const double wsq = pref_sq[k + patch_size] - pref_sq[k];
        const double win_ss = std::max(wsq - (ws * ws) / patch_area, 0.0);
        const double denom = std::sqrt(p_ss * win_ss);
        double sc = 0.0;
        if (win_ss > flat_var_floor && denom > 0.0) {
            sc = static_cast<double>(num_f[k]) / denom;
            if (sc > 1.0) sc = 1.0;
            else if (sc < -1.0) sc = -1.0;
        }
        scores_buf[k] = sc;

        const double pos = pos_base + static_cast<double>(k);
        const double disp = direction * (pos - x);
        const bool ok = (
            pos >= radius && pos <= (w - 1 - radius)
            && disp > disp_lo && disp < disp_hi
            && std::isfinite(sc)
        );
        valid_mask[k] = ok ? 1 : 0;
        if (ok && sc > best_score) {
            best_score = sc;
            best = k;
        }
    }

    if (best < 0) {
        return REASON_SEARCH_BOUNDS;
    }
    if (best_score < min_corr) {
        return REASON_CORRELATION;
    }

    bool any_comp = false;
    double second_score = -std::numeric_limits<double>::infinity();
    for (int32_t k = 0; k < n_scores; ++k) {
        if (valid_mask[k] && std::abs(k - best) > 1) {
            any_comp = true;
            if (scores_buf[k] > second_score) {
                second_score = scores_buf[k];
            }
        }
    }
    if (!any_comp) {
        return REASON_INSUFFICIENT_ALTERNATIVES;
    }
    if (best_score - second_score <= tol) {
        return REASON_NEAR_TIE;
    }
    if (1.0 - best_score >= uniq_ratio * (1.0 - second_score)) {
        return REASON_AMBIGUOUS;
    }
    if (!(best > 0 && best < n_scores - 1 && valid_mask[best - 1] && valid_mask[best + 1])) {
        return REASON_UNREFINABLE_PEAK;
    }
    const double s_prev = scores_buf[best - 1];
    const double s_next = scores_buf[best + 1];
    const double curv = s_prev - 2.0 * best_score + s_next;
    if (curv >= -tol) {
        return REASON_FLAT_PEAK;
    }
    const double offset = 0.5 * (s_prev - s_next) / curv;
    if (std::abs(offset) > 0.5) {
        return REASON_UNSTABLE_PEAK;
    }
    const double res_x = pos_base + static_cast<double>(best) + offset;
    const double ref_disp = direction * (res_x - x);
    if (!(res_x >= radius && res_x <= (w - 1 - radius) && ref_disp > disp_lo && ref_disp < disp_hi)) {
        return REASON_REFINED_BOUNDS;
    }

    // Sample refined target patch and verify texture + correlation
    float actual_buf[256];
    sample_rect_fast(tgt_img, h, w, static_cast<float>(res_x), static_cast<float>(y),
                     patch_size, patch_size, actual_buf);
    double a_sum = 0.0;
    for (int32_t k = 0; k < npix; ++k) {
        a_sum += static_cast<double>(actual_buf[k]);
    }
    const double a_mean = a_sum / patch_area;
    double a_ss = 0.0;
    double a_num = 0.0;
    for (int32_t k = 0; k < npix; ++k) {
        const double azm = static_cast<double>(actual_buf[k]) - a_mean;
        a_ss += azm * azm;
        a_num += azm * static_cast<double>(patch_buf[k]);
    }
    if (std::sqrt(a_ss / patch_area) < min_tex_std) {
        return REASON_LOW_TARGET_TEXTURE;
    }
    const double a_denom = std::sqrt(p_ss * a_ss);
    if (!(a_denom > 0.0)) {
        return REASON_REFINED_CORRELATION;
    }
    const double corr = std::max(-1.0, std::min(1.0, a_num / a_denom));
    if (!std::isfinite(corr) || corr < min_corr) {
        return REASON_REFINED_CORRELATION;
    }

    *out_x = res_x;
    return REASON_OK;
}

}  // namespace

extern "C" {

// Full stereo verification with forward (L->R) + reverse (R->L) + round-trip gate.
// Supports patch_size <= 15 and max_disp <= 240.
int32_t verify_stereo_zncc_batch_f32(
    const float* left_img,
    const float* right_img,
    int32_t h,
    int32_t w,
    const double* pixels_xy,   // [n, 2]
    int32_t n,
    int32_t patch_size,
    int32_t max_disp,
    double min_corr,
    double uniq_ratio,
    double round_trip_px,
    double min_tex_std,
    double disp_lo,
    double disp_hi,
    double* out_right_u,       // [n]
    int32_t* out_reasons       // [n]
) {
    if (patch_size < 3 || patch_size > 15 || (patch_size & 1) == 0 || max_disp < 2 || max_disp > 240) {
        return 0;
    }
    std::vector<float> patch_buf(static_cast<size_t>(patch_size) * patch_size);
    std::vector<float> strip_buf(static_cast<size_t>(patch_size) * (patch_size + max_disp));
    std::vector<double> scores_buf(static_cast<size_t>(max_disp) + 1);

    for (int32_t i = 0; i < n; ++i) {
        const double px = pixels_xy[2 * i + 0];
        const double py = pixels_xy[2 * i + 1];
        out_right_u[i] = std::numeric_limits<double>::quiet_NaN();

        double right_x = 0.0;
        const int32_t fwd_code = zncc_search_one(
            left_img, right_img, h, w, px, py, -1.0,
            patch_size, max_disp, min_corr, uniq_ratio, min_tex_std, disp_lo, disp_hi,
            patch_buf.data(), strip_buf.data(), scores_buf.data(), &right_x
        );
        if (fwd_code != REASON_OK) {
            out_reasons[i] = fwd_code;
            continue;
        }

        double back_x = 0.0;
        const int32_t rev_code = zncc_search_one(
            right_img, left_img, h, w, right_x, py, 1.0,
            patch_size, max_disp, min_corr, uniq_ratio, min_tex_std, disp_lo, disp_hi,
            patch_buf.data(), strip_buf.data(), scores_buf.data(), &back_x
        );
        if (rev_code != REASON_OK) {
            out_reasons[i] = REASON_REVERSE_PREFIX_OFFSET + rev_code;
            continue;
        }

        if (std::abs(back_x - px) > round_trip_px) {
            out_reasons[i] = REASON_ROUND_TRIP;
            continue;
        }

        out_right_u[i] = right_x;
        out_reasons[i] = REASON_OK;
    }
    return 1;
}

}  // extern "C"
