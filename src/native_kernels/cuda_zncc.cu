// Sparse forward and reverse ZNCC for 11x11 grayscale patches and a 96-pixel
// rectified search. Rejection codes match the CPU verifier's reason table.
constexpr int radius = 5;
constexpr int side = 2 * radius + 1;
constexpr int area = side * side;
constexpr int max_disp = 96;
constexpr int strip_width = side + max_disp;
constexpr double tolerance = 8.0 * 1.1920928955078125e-7;

struct SearchBuffers {
    float patch[area];
    float strip[side * strip_width];
    double sum[strip_width + 1];
    double squares[strip_width + 1];
    double scores[max_disp + 1];
    double warp_sum[4];
    double warp_squares[4];
    int warp_best[4];
    double patch_variance;
    double match_x;
    int reason;
};

struct Interpolation {
    __device__ Interpolation(double x, double y) {
        int x32 = __float2int_rn(static_cast<float>(x) * 32);
        int y32 = __float2int_rn(static_cast<float>(y) * 32);
        x0 = x32 >> 5;
        y0 = y32 >> 5;
        int dx = x32 & 31;
        int dy = y32 & 31;
        w00 = (32 - dx) * (32 - dy);
        w01 = dx * (32 - dy);
        w10 = (32 - dx) * dy;
        w11 = dx * dy;
    }
    __device__ float sample(const unsigned char* image, int h, int w, int x, int y) const {
        int left = max(0, min(x0 + x, w - 1));
        int right = max(0, min(x0 + x + 1, w - 1));
        int top = max(0, min(y0 + y, h - 1)) * w;
        int bottom = max(0, min(y0 + y + 1, h - 1)) * w;
        int value = w00 * image[top + left] + w01 * image[top + right]
                  + w10 * image[bottom + left] + w11 * image[bottom + right];
        return value * (1.0f / 1024.0f);
    }
    int x0, y0, w00, w01, w10, w11;
};

__device__ int select_peak(SearchBuffers& b, int peak, double competitor,
                           int w, double x, double start, int direction,
                           double lower, double upper) {
    double score = b.scores[peak];
    if (competitor == -2.0) return 5;
    if (score - competitor <= tolerance) return 6;
    if (1.0 - score >= 0.7 * (1.0 - competitor)) return 7;
    if (peak == 0 || peak == max_disp || b.scores[peak - 1] == -2.0 || b.scores[peak + 1] == -2.0)
        return 8;
    double before = b.scores[peak - 1], after = b.scores[peak + 1];
    double curvature = before - 2.0 * score + after;
    if (curvature >= -tolerance) return 9;
    double correction = 0.5 * (before - after) / curvature;
    if (fabs(correction) > 0.5) return 10;
    double right_x = start + peak + correction;
    double disparity = direction * (right_x - x);
    if (right_x < radius || right_x > w - 1 - radius || disparity <= lower || disparity >= upper)
        return 11;
    b.match_x = right_x;
    return 0;
}

__device__ int validate_refined_patch(SearchBuffers& b) {
    double total = 0.0;
    for (int k = 0; k < area; ++k)
        total += b.strip[k];
    double mean = total / area;
    double squared = 0.0, dot = 0.0;
    for (int k = 0; k < area; ++k) {
        double centered = b.strip[k] - mean;
        squared += centered * centered;
        dot += centered * b.patch[k];
    }
    if (sqrt(squared / area) < 2.0) return 12;
    if (dot / sqrt(squared * b.patch_variance) < 0.85) return 13;
    return 0;
}

__device__ void search(SearchBuffers& b, const unsigned char* source,
                       const unsigned char* target, int h, int w, double x, double y,
                       int direction, double lower, double upper) {
    int lane = threadIdx.x;
    int warp = lane / 32;
    int in_warp = lane % 32;
    if (!(x >= radius && x <= w - 1 - radius && y >= radius && y <= h - 1 - radius)) {
        if (lane == 0) b.reason = 1;
        __syncthreads();
        return;
    }
    double start = x + (direction - 1) * (max_disp * 0.5);
    Interpolation patch_grid(x - radius, y - radius);
    Interpolation strip_grid(start - radius, y - radius);
    if (lane < area)
        b.patch[lane] = patch_grid.sample(source, h, w, lane % side, lane / side);
    for (int k = lane; k < side * strip_width; k += blockDim.x)
        b.strip[k] = strip_grid.sample(target, h, w, k % strip_width, k / strip_width);
    __syncthreads();

    if (lane == 0) {
        double mean = 0.0;
        for (float value : b.patch) mean += value;
        mean /= area;
        double variance = 0.0;
        for (float& value : b.patch) {
            double centered = value - mean;
            value = static_cast<float>(centered);
            variance += centered * centered;
        }
        b.patch_variance = variance;
        b.reason = sqrt(variance / area) < 2.0 ? 2 : 0;
    }
    double sum = 0.0, squares = 0.0;
    if (lane < strip_width) {
        for (int row = 0; row < side; ++row) {
            double v = b.strip[row * strip_width + lane];
            sum += v;
            squares += v * v;
        }
    }
    for (int offset = 1; offset < 32; offset *= 2) {
        double low_sum = __shfl_up_sync(0xffffffff, sum, offset);
        double low_squares = __shfl_up_sync(0xffffffff, squares, offset);
        if (in_warp >= offset) {
            sum += low_sum;
            squares += low_squares;
        }
    }
    if (in_warp == 31) {
        b.warp_sum[warp] = sum;
        b.warp_squares[warp] = squares;
    }
    __syncthreads();
    if (b.reason) return;
    if (lane == 0) b.sum[0] = b.squares[0] = 0;
    if (lane < strip_width) {
        // Quantized uint8 interpolation gives exact dyadic sums in float64.
        for (int i = 0; i < warp; ++i) {
            sum += b.warp_sum[i];
            squares += b.warp_squares[i];
        }
        b.sum[lane + 1] = sum;
        b.squares[lane + 1] = squares;
    }
    __syncthreads();
    double score = -2.0;
    if (lane <= max_disp) {
        double position = start + lane;
        double disp = direction * (position - x);
        if (position >= radius && position <= w - 1 - radius && disp > lower && disp < upper) {
            float dot = 0;
            for (int row = 0; row < side; ++row) {
                #pragma unroll
                for (int col = 0; col < side; ++col)
                    dot = fmaf(b.patch[row * side + col], b.strip[row * strip_width + col + lane], dot);
            }
            double sum = b.sum[lane + side] - b.sum[lane];
            double squares = b.squares[lane + side] - b.squares[lane];
            double variance = squares - sum * sum / area;
            score = variance > 1e-6 * area
                ? fmax(-1.0, fmin(1.0, dot / sqrt(b.patch_variance * variance)))
                : 0.0;
        }
        b.scores[lane] = score;
    }
    int best = lane;
    double value = score;
    for (int offset = 16; offset > 0; offset /= 2) {
        double candidate = __shfl_down_sync(0xffffffff, value, offset);
        int candidate_lane = __shfl_down_sync(0xffffffff, best, offset);
        if (candidate > value || (candidate == value && candidate_lane < best)) {
            value = candidate;
            best = candidate_lane;
        }
    }
    if (in_warp == 0) {
        b.warp_best[warp] = best;
        b.warp_sum[warp] = value;
    }
    __syncthreads();
    if (lane == 0) {
        for (int i = 1; i < 4; ++i) {
            if (b.warp_sum[i] > value) {
                value = b.warp_sum[i];
                best = b.warp_best[i];
            }
        }
        b.warp_best[0] = best;
        b.reason = value == -2.0 ? 3 : (value < 0.85 ? 4 : 0);
    }
    __syncthreads();
    if (b.reason) return;
    int peak = b.warp_best[0];
    double competitor = abs(lane - peak) > 1 ? score : -2.0;
    for (int offset = 16; offset > 0; offset /= 2)
        competitor = fmax(competitor, __shfl_down_sync(0xffffffff, competitor, offset));
    if (in_warp == 0) b.warp_sum[warp] = competitor;
    __syncthreads();
    if (lane == 0) {
        for (int i = 1; i < 4; ++i) competitor = fmax(competitor, b.warp_sum[i]);
        b.reason = select_peak(b, peak, competitor, w, x, start, direction, lower, upper);
    }
    __syncthreads();
    if (b.reason) return;
    if (lane < area) {
        Interpolation refined(b.match_x - radius, y - radius);
        b.strip[lane] = refined.sample(target, h, w, lane % side, lane / side);
    }
    __syncthreads();
    if (lane == 0) b.reason = validate_refined_patch(b);
    __syncthreads();
}

extern "C" __global__ void verify_zncc(const unsigned char* left, const unsigned char* right,
                                       const double* pixels, double* output, int* reasons,
                                       int h, int w, double lower, double upper) {
    __shared__ SearchBuffers b;
    __shared__ double right_x;
    int point = blockIdx.x;
    double x = pixels[2 * point], y = pixels[2 * point + 1];
    search(b, left, right, h, w, x, y, -1, lower, upper);
    if (b.reason) {
        if (threadIdx.x == 0) {
            output[point] = nan("");
            reasons[point] = b.reason;
        }
        return;
    }
    if (threadIdx.x == 0) right_x = b.match_x;
    __syncthreads();
    search(b, right, left, h, w, right_x, y, 1, lower, upper);
    if (threadIdx.x == 0) {
        int reason = b.reason ? 100 + b.reason : (fabs(b.match_x - x) > 0.5 ? 200 : 0);
        output[point] = reason ? nan("") : right_x;
        reasons[point] = reason;
    }
}
