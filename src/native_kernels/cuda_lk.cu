// Fused forward-backward Lucas-Kanade. Patch accumulation follows fast_lk.cpp.
constexpr int half_window = 10;
constexpr int side = 2 * half_window + 1;
constexpr int area = side * side;
__device__ float pixel(const float* image, int w, int h, int x, int y) {
    x = min(max(0, x), w - 1);
    y = min(max(0, y), h - 1);
    return image[y * w + x];
}

extern "C" __global__ void init_gray(const unsigned char* source, float* pyramid, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) pyramid[i] = static_cast<float>(source[i]);
}

__device__ float binomial(float a, float b, float c, float d, float e) {
    return fmaf(1.0f / 16.0f, a + e, fmaf(4.0f / 16.0f, b + d, (6.0f / 16.0f) * c));
}

extern "C" __global__ void down_gray(float* pyramid, int src, int dst,
                                    int w, int h, int out_w, int out_h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_w * out_h) return;
    int x = 2 * (i % out_w), y = 2 * (i / out_w);
    float rows[5];
    for (int j = 0; j < 5; ++j) {
        int yy = y + j - 2;
        rows[j] = binomial(pixel(pyramid + src, w, h, x - 2, yy),
                           pixel(pyramid + src, w, h, x - 1, yy),
                           pixel(pyramid + src, w, h, x, yy),
                           pixel(pyramid + src, w, h, x + 1, yy),
                           pixel(pyramid + src, w, h, x + 2, yy));
    }
    pyramid[dst + i] = binomial(rows[0], rows[1], rows[2], rows[3], rows[4]);
}

struct Bilinear {
    __device__ Bilinear(float x, float y) {
        ix = __float2int_rd(x);
        iy = __float2int_rd(y);
        float a = x - ix, b = y - iy;
        w00 = (1 - a) * (1 - b);
        w01 = a * (1 - b);
        w10 = (1 - a) * b;
        w11 = a * b;
    }
    __device__ bool bounds(int w, int h) const {
        return ix >= -half_window && iy >= -half_window
            && ix + side <= w + half_window && iy + side <= h + half_window;
    }
    __device__ bool interior(int w, int h, int border) const {
        return ix >= border && iy >= border && ix + side + border < w && iy + side + border < h;
    }
    __device__ float weights(float a, float b, float c, float d) const {
        return fmaf(w00, a, fmaf(w01, b, fmaf(w10, c, w11 * d)));
    }
    __device__ float tail_derivative(float a, float b, float c, float d) const {
        return fmaf(w00, a, w01 * b) + fmaf(w10, c, w11 * d);
    }
    __device__ float sample(const float* image, int w, int h, int x, int y, bool direct = false) const {
        x += ix;
        y += iy;
        if (direct) {
            const float* block = image + y * w + x;
            return weights(block[0], block[1], block[w], block[w + 1]);
        }
        return weights(pixel(image, w, h, x, y), pixel(image, w, h, x + 1, y),
                       pixel(image, w, h, x, y + 1), pixel(image, w, h, x + 1, y + 1));
    }
    __device__ void template_sample(const float* image, int w, int h, int x, int y,
                                    float& value, float& gx, float& gy,
                                    bool cpu_tail = false, bool direct = false) const {
        x += ix;
        y += iy;
        if (direct) {
            const float* p = image + y * w + x;
            value = weights(p[0], p[1], p[w], p[w + 1]);
            float dx0 = 0.5f * (p[1] - p[-1]);
            float dx1 = 0.5f * (p[2] - p[0]);
            float dx2 = 0.5f * (p[w + 1] - p[w - 1]);
            float dx3 = 0.5f * (p[w + 2] - p[w]);
            float dy0 = 0.5f * (p[w] - p[-w]);
            float dy1 = 0.5f * (p[w + 1] - p[-w + 1]);
            float dy2 = 0.5f * (p[2 * w] - p[0]);
            float dy3 = 0.5f * (p[2 * w + 1] - p[1]);
            gx = cpu_tail ? tail_derivative(dx0, dx1, dx2, dx3) : weights(dx0, dx1, dx2, dx3);
            gy = cpu_tail ? tail_derivative(dy0, dy1, dy2, dy3) : weights(dy0, dy1, dy2, dy3);
            return;
        }
        float p00 = pixel(image, w, h, x, y), p01 = pixel(image, w, h, x + 1, y);
        float p10 = pixel(image, w, h, x, y + 1), p11 = pixel(image, w, h, x + 1, y + 1);
        value = weights(p00, p01, p10, p11);
        float dx0 = 0.5f * (p01 - pixel(image, w, h, x - 1, y));
        float dx1 = 0.5f * (pixel(image, w, h, x + 2, y) - p00);
        float dx2 = 0.5f * (p11 - pixel(image, w, h, x - 1, y + 1));
        float dx3 = 0.5f * (pixel(image, w, h, x + 2, y + 1) - p10);
        float dy0 = 0.5f * (p10 - pixel(image, w, h, x, y - 1));
        float dy1 = 0.5f * (p11 - pixel(image, w, h, x + 1, y - 1));
        float dy2 = 0.5f * (pixel(image, w, h, x, y + 2) - p00);
        float dy3 = 0.5f * (pixel(image, w, h, x + 1, y + 2) - p01);
        gx = cpu_tail ? tail_derivative(dx0, dx1, dx2, dx3) : weights(dx0, dx1, dx2, dx3);
        gy = cpu_tail ? tail_derivative(dy0, dy1, dy2, dy3) : weights(dy0, dy1, dy2, dy3);
    }
    int ix, iy;
    float w00, w01, w10, w11;
};

struct Template {
    float value[area], gx[area], gy[area];
};

__device__ float sum_avx_lanes(float value) {
    // Same 8, 4, 2, 1 reduction tree as the MinGW AVX-512 DLL.
    unsigned mask = 0xffffU << (threadIdx.x & 16);
    for (int stride = 8; stride; stride /= 2) {
        value += __shfl_xor_sync(mask, value, stride, 16);
    }
    return __shfl_sync(mask, value, 0, 16);
}

// Parallelize the pixel sampling while retaining the CPU lane accumulation order.
struct CooperativePatch {
    Template patch;
    float diff[area];
    float x, y, previous_dx, previous_dy;
    double a, b, c, inverse;
    int usable, stop, valid;
};

__device__ bool track_cooperative(
    const float* source, const float* target, int w0, int h0, int levels,
    float px0, float py0, float& nx, float& ny, CooperativePatch& shared
) {
    if (!(isfinite(px0) && isfinite(py0) && isfinite(nx) && isfinite(ny))) return false;
    int offset[7], widths[7], heights[7];
    offset[0] = 0;
    widths[0] = w0;
    heights[0] = h0;
    for (int level = 1; level <= levels; ++level) {
        int w = widths[level - 1], h = heights[level - 1];
        offset[level] = offset[level - 1] + w * h;
        widths[level] = (w + 1) / 2;
        heights[level] = (h + 1) / 2;
    }
    int lane = threadIdx.x;
    float initial_scale = 1.0f / (1 << levels);
    if (lane == 0) {
        shared.x = nx * initial_scale;
        shared.y = ny * initial_scale;
        shared.valid = 1;
    }
    __syncthreads();
    for (int level = levels; level >= 0; --level) {
        if (level < levels && lane == 0) {
            shared.x *= 2.0f;
            shared.y *= 2.0f;
        }
        float scale = 1.0f / (1 << level);
        int w = widths[level], h = heights[level];
        const float* image0 = source + offset[level];
        const float* image1 = target + offset[level];
        Bilinear tmpl(px0 * scale - half_window, py0 * scale - half_window);
        if (!tmpl.bounds(w, h)) {
            if (level == 0 && lane == 0) shared.valid = 0;
            continue;
        }
        bool direct = tmpl.interior(w, h, 1);
        for (int i = lane; i < area; i += blockDim.x) {
            int col = i % side, row = i / side;
            tmpl.template_sample(image0, w, h, col, row,
                shared.patch.value[i], shared.patch.gx[i], shared.patch.gy[i], col >= 16, direct);
        }
        __syncthreads();
        if (lane < 16) {
            float xx = 0, xy = 0, yy = 0;
            #pragma unroll
            for (int row = 0; row < side; ++row) {
                int i = row * side + lane;
                float gx = shared.patch.gx[i], gy = shared.patch.gy[i];
                xx = fmaf(gx, gx, xx);
                xy = fmaf(gx, gy, xy);
                yy = fmaf(gy, gy, yy);
                if (lane < 5) {
                    gx = shared.patch.gx[i + 16];
                    gy = shared.patch.gy[i + 16];
                    xx = fmaf(gx, gx, xx);
                    xy = fmaf(gx, gy, xy);
                    yy = fmaf(gy, gy, yy);
                }
            }
            double a = sum_avx_lanes(xx), b = sum_avx_lanes(xy), c = sum_avx_lanes(yy);
            if (lane == 0) {
                double determinant = fma(a, c, -(b * b));
                double min_eigen = ((a + c) - sqrt(fma(a - c, a - c, 4.0 * (b * b))))
                    * (0.5 * (1.0f / area));
                shared.usable = min_eigen >= static_cast<float>(1e-4) && determinant >= 1e-7;
                if (!shared.usable && level == 0) shared.valid = 0;
                shared.a = a; shared.b = b; shared.c = c;
                shared.inverse = 1.0 / determinant;
                shared.previous_dx = shared.previous_dy = 0.0f;
            }
        }
        __syncthreads();
        if (!shared.usable) continue;
        for (int iteration = 0; iteration < 30; ++iteration) {
            Bilinear match(shared.x - half_window, shared.y - half_window);
            if (!match.bounds(w, h)) {
                __syncthreads();
                if (level == 0 && lane == 0) shared.valid = 0;
                break;
            }
            bool direct_match = match.interior(w, h, 0);
            for (int i = lane; i < area; i += blockDim.x) {
                shared.diff[i] = shared.patch.value[i] - match.sample(image1, w, h,
                    i % side, i / side, direct_match);
            }
            __syncthreads();
            if (lane < 16) {
                float du = 0, dv = 0;
                #pragma unroll
                for (int row = 0; row < side; ++row) {
                    int i = row * side + lane;
                    du = fmaf(shared.diff[i], shared.patch.gx[i], du);
                    dv = fmaf(shared.diff[i], shared.patch.gy[i], dv);
                    if (lane < 5) {
                        du = fmaf(shared.diff[i + 16], shared.patch.gx[i + 16], du);
                        dv = fmaf(shared.diff[i + 16], shared.patch.gy[i + 16], dv);
                    }
                }
                double u = sum_avx_lanes(du), v = sum_avx_lanes(dv);
                if (lane == 0) {
                    float dx = fma(shared.c, u, -(shared.b * v)) * shared.inverse;
                    float dy = fma(shared.a, v, -(shared.b * u)) * shared.inverse;
                    shared.x += dx; shared.y += dy;
                    shared.stop = fmaf(dx, dx, dy * dy) <= 0.0001f;
                    if (!shared.stop && iteration && fabsf(dx + shared.previous_dx) < 0.01f
                            && fabsf(dy + shared.previous_dy) < 0.01f) {
                        shared.x -= 0.5f * dx;
                        shared.y -= 0.5f * dy;
                        shared.stop = 1;
                    }
                    shared.previous_dx = dx; shared.previous_dy = dy;
                }
            }
            __syncthreads();
            if (shared.stop) break;
        }
    }
    __syncthreads();
    nx = shared.x; ny = shared.y;
    bool valid = shared.valid && nx >= 0.0f && ny >= 0.0f && nx < w0 && ny < h0;
    __syncthreads();
    return valid;
}

extern "C" __global__ void track_lk_pair(const float* prev, const float* next,
    const float* points, const float* seed, float* output, unsigned char* status,
    int w, int h, int levels) {
    __shared__ CooperativePatch shared;
    int point = blockIdx.x;
    float x = points[2 * point], y = points[2 * point + 1];
    float nx = seed[2 * point], ny = seed[2 * point + 1];
    bool ok = track_cooperative(prev, next, w, h, levels, x, y, nx, ny, shared);
    float bx = levels >= 4 ? x : nx, by = levels >= 4 ? y : ny;
    bool back_ok = track_cooperative(next, prev, w, h, levels, nx, ny, bx, by, shared);
    if (threadIdx.x == 0) {
        output[4 * point] = nx;
        output[4 * point + 1] = ny;
        output[4 * point + 2] = bx;
        output[4 * point + 3] = by;
        status[2 * point] = ok;
        status[2 * point + 1] = back_ok;
    }
}
