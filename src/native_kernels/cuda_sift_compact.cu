// Compact the existing SIFT extrema and the final local stencil on the GPU.
struct SiftExtremumSeed {
    int order, octave, row, column, layer;
    float xc, xr, xi, value, dx, dy, ds, dxx, dyy, dxy;
};
static_assert(sizeof(SiftExtremumSeed) == 60);

__device__ bool solve_seed(float dxx, float dyy, float dss, float dxy, float dxs,
                           float dys, float dx, float dy, float ds,
                           float& xc, float& xr, float& xi) {
    const double a = dxx, b = dxy, c = dxs, d = dyy, e = dys, f = dss;
    const double aa = d * f - e * e, bb = e * c - b * f, cc = b * e - d * c;
    const double dd = a * f - c * c, ee = c * b - a * e, ff = a * d - b * b;
    const double det = a * aa + b * bb + c * cc;
    if (fabs(det) < 1e-12) return false;
    const double inverse = 1.0 / det;
    xc = static_cast<float>((aa * -dx + bb * -dy + cc * -ds) * inverse);
    xr = static_cast<float>((bb * -dx + dd * -dy + ee * -ds) * inverse);
    xi = static_cast<float>((cc * -dx + ee * -dy + ff * -ds) * inverse);
    return isfinite(xc) && isfinite(xr) && isfinite(xi);
}

extern "C" __global__ void sift_sparse_seeds(
    const short* dog, SiftExtremumSeed* packet, int* count, int capacity, int h, int w,
    int octave, int seed_order, int pre_threshold
) {
    constexpr int border = 5;
    const int area = h * w;
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= 3 * area) return;
    const int i = 1 + tid / area, pixel = tid % area, r = pixel / w, c = pixel % w;
    const int end_column = border + 32 * ((w - 2 * border) / 32);
    if (r < border || r >= h - border || c < border || c >= end_column) return;

    const short* center = dog + i * area + pixel;
    const int value = *center;
    if (value == -32768 || abs(value) <= pre_threshold) return;
    const int sign = value > 0 ? 1 : -1;
    const int peak = value * sign;
    if (peak < sign * center[-1] || peak < sign * center[1]
        || peak < sign * center[-w] || peak < sign * center[w]
        || peak < sign * center[-area] || peak < sign * center[area]) return;
    for (int layer = -1; layer <= 1; ++layer) {
        for (int row = -1; row <= 1; ++row) {
            const short* adjacent = center + layer * area + row * w;
            if (peak < sign * adjacent[-1] || peak < sign * adjacent[0]
                || peak < sign * adjacent[1]) return;
        }
    }

    int row = r, col = c, layer = i;
    for (int step = 0; step < 5; ++step) {
        const int at = row * w + col;
        const short* dc = dog + layer * area + at;
        const short* dp = dc - area;
        const short* dn = dc + area;
        const float v0 = static_cast<float>(*dc);
        const float dx = 0.5f * (dc[1] - dc[-1]);
        const float dy = 0.5f * (dc[w] - dc[-w]);
        const float ds = 0.5f * (*dn - *dp);
        const float dxx = dc[1] + dc[-1] - 2.0f * v0;
        const float dyy = dc[w] + dc[-w] - 2.0f * v0;
        const float dss = *dn + *dp - 2.0f * v0;
        const float dxy = 0.25f * (dc[w + 1] - dc[w - 1] - dc[-w + 1] + dc[-w - 1]);
        const float dxs = 0.25f * (dn[1] - dn[-1] - dp[1] + dp[-1]);
        const float dys = 0.25f * (dn[w] - dn[-w] - dp[w] + dp[-w]);
        float xc, xr, xi;
        if (!solve_seed(dxx, dyy, dss, dxy, dxs, dys, dx, dy, ds, xc, xr, xi)) return;
        if (fabsf(xc) < 0.5f && fabsf(xr) < 0.5f && fabsf(xi) < 0.5f) {
            const int slot = atomicAdd(count, 1);
            if (slot < capacity) {
                packet[slot] = {
                    seed_order + tid, octave, row, col, layer, xc, xr, xi,
                    v0, dx, dy, ds, dxx, dyy, dxy
                };
            }
            return;
        }
        if (fabsf(xc) > 1e5f || fabsf(xr) > 1e5f || fabsf(xi) > 1e5f) return;
        col += static_cast<int>(roundf(xc));
        row += static_cast<int>(roundf(xr));
        layer += static_cast<int>(roundf(xi));
        if (layer < 1 || layer > 3 || col < border || col >= w - border
            || row < border || row >= h - border) return;
    }
}

extern "C" __global__ void sift_pack_gaussian(const short* input, short* output, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) output[i] = input[i];
}

// SIFT descriptor sampling and ordered histogram accumulation on a resident pyramid.
struct SiftDescriptorKeypoint {
    float x, y, cos_scale, sin_scale, angle;
    int offset, width, height;
};
static_assert(sizeof(SiftDescriptorKeypoint) == 32);

__device__ float descriptor_angle(float y, float x) {
    constexpr float pi = 3.14159265358979323846f;
    const float ax = fabsf(x), ay = fabsf(y);
    const float a = fminf(ax, ay) / (fmaxf(ax, ay) + 1e-12f);
    const float s = a * a;
    float r = fmaf(-0.0464964749f, s, 0.15931422f);
    r = fmaf(r, s, -0.327622764f);
    r = fmaf(r * s, a, a);
    if (ay > ax) r = .5f * pi - r;
    if (x < 0.f) r = pi - r;
    float deg = r * (180.f / pi);
    if (y < 0.f) deg = 360.f - deg;
    if (deg >= 360.f) deg -= 360.f;
    return deg;
}

__device__ float sum16(float value) {
    for (int stride = 8; stride > 0; stride /= 2)
        value += __shfl_down_sync(0xffffffff, value, stride, 16);
    return value;
}

extern "C" __global__ void sift_descriptors(const short* pyramid,
        const SiftDescriptorKeypoint* keypoints, const float* weights,
        float* descriptors) {
    __shared__ float magnitude[256], angle_fraction[256], raw[128];
    __shared__ int angle_bin[256];
    __shared__ float clip, normalization;
    const int point = blockIdx.x, k = threadIdx.x;
    const SiftDescriptorKeypoint kp = keypoints[point];
    const float u = -1.875f + .25f * (k % 16), v = -1.875f + .25f * (k / 16);
    const int x = __float2int_rn(kp.x + fmaf(u, kp.cos_scale, -(v * kp.sin_scale)));
    const int y = __float2int_rn(kp.y + fmaf(u, kp.sin_scale, v * kp.cos_scale));
    float m = 0.f, f = 0.f;
    int bin = 0;
    if (x >= 1 && x <= kp.width - 2 && y >= 1 && y <= kp.height - 2) {
        const short* center = pyramid + kp.offset + y * kp.width + x;
        const float gx = center[1] - center[-1];
        const float gy = center[-kp.width] - center[kp.width];
        m = sqrtf(fmaf(gx, gx, gy * gy));
        const float offset = (descriptor_angle(gy, gx) - kp.angle) * (8.f / 360.f);
        const int base = __float2int_rd(offset);
        f = offset - float(base);
        bin = base & 7;
    }
    magnitude[k] = m;
    angle_fraction[k] = f;
    angle_bin[k] = bin;
    __syncthreads();

    // Each lane owns one bin and traverses that cell's support in sample order.
    if (k < 128) {
        const int row = k / 32, col = k / 8 % 4, orientation = k % 8;
        float sum = 0.f, wrap = 0.f;
        for (int j = max(0, 4 * row - 2); j < min(16, 4 * row + 6); ++j) {
            const int row_channel = j < 4 * row + 2 ? 2 : 0;
            for (int i = max(0, 4 * col - 2); i < min(16, 4 * col + 6); ++i) {
                const int sample = j * 16 + i, direction = angle_bin[sample];
                const int channel = row_channel + int(i < 4 * col + 2);
                const float weighted = magnitude[sample] * weights[channel * 256 + sample];
                const float part = angle_fraction[sample];
                if (direction == orientation)
                    sum = sum + weighted * (1.f - part);
                if (direction + 1 == orientation)
                    sum = sum + weighted * part;
                if (orientation == 0 && direction == 7)
                    wrap = wrap + weighted * part;
            }
        }
        raw[k] = sum + wrap;
    }
    __syncthreads();

    // Use the CPU descriptor's groups of 16 for the two L2 reductions.
    if (k < 32) {
        float square_sum = 0.f;
        for (int i = (k & 15); i < 128; i += 16)
            square_sum = fmaf(raw[i], raw[i], square_sum);
        square_sum = sum16(square_sum);
        if (k == 0) clip = .2f * sqrtf(square_sum);
    }
    __syncthreads();
    if (k < 32) {
        float square_sum = 0.f;
        for (int i = (k & 15); i < 128; i += 16) {
            const float value = fminf(raw[i], clip);
            square_sum = fmaf(value, value, square_sum);
        }
        square_sum = sum16(square_sum);
        if (k == 0) normalization = 512.f / fmaxf(sqrtf(square_sum), 1e-6f);
    }
    __syncthreads();
    if (k < 128) {
        const float scaled = fminf(raw[k], clip) * normalization;
        descriptors[point * 128 + k] = float(max(0, min(255, __float2int_rn(scaled))));
    }
}
