// Native SIFT Q7 Gaussian and Q9 DoG planes with fixed-point rounding.
__device__ int clamp_index(int value, int length) {
    return max(0, min(value, length - 1));
}

extern "C" __global__ void sift_base(const unsigned char* image, short* pyramid,
                                     int width, int height, int doubled, int destination) {
    const int output_width = doubled ? 2 * width : width;
    const int output_height = doubled ? 2 * height : height;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= output_width * output_height) return;
    if (!doubled) {
        pyramid[destination + i] = image[i] << 7;
        return;
    }
    const int x = i % output_width;
    const int y = i / output_width;
    const int sx = x / 2;
    const int sy = y / 2;
    const int nx = clamp_index(sx + ((x & 1) ? 1 : -1), width);
    const int ny = clamp_index(sy + ((y & 1) ? 1 : -1), height);
    const int row = 96 * image[sy * width + sx] + 32 * image[sy * width + nx];
    const int other = 96 * image[ny * width + sx] + 32 * image[ny * width + nx];
    pyramid[destination + i] = (3 * row + other + 2) >> 2;
}

extern "C" __global__ void sift_down(short* pyramid, int source, int destination,
                                     int old_width, int width, int height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;
    pyramid[destination + i] = pyramid[source + 2 * (i / width) * old_width + 2 * (i % width)];
}

template<int Radius>
__device__ void sift_blur(short* pyramid, const short* weights, int weight_start,
                          int source, int gaussian, int dog, int width, int height) {
    constexpr int TileX = 32;
    constexpr int TileY = 8;
    constexpr int Rows = TileY + 2 * Radius;
    constexpr int Width = TileX + 2 * Radius;
    __shared__ short input[Rows][Width];
    __shared__ short horizontal[Rows][TileX];
    const short* coeff = weights + weight_start;
    const int lane = threadIdx.x + TileX * threadIdx.y;
    const int x = blockIdx.x * TileX;
    const int y = blockIdx.y * TileY;
    for (int i = lane; i < Rows * Width; i += TileX * TileY) {
        const int row = i / Width, col = i % Width;
        const int src_x = clamp_index(x + col - Radius, width);
        const int src_y = clamp_index(y + row - Radius, height);
        input[row][col] = pyramid[source + src_y * width + src_x];
    }
    __syncthreads();
    for (int row = threadIdx.y; row < Rows; row += TileY) {
        const short* samples = input[row] + threadIdx.x + Radius;
        int sum = coeff[0] * samples[0];
        #pragma unroll
        for (int k = 1; k <= Radius; ++k) sum += coeff[k] * (samples[-k] + samples[k]);
        horizontal[row][threadIdx.x] = (sum + 16384) >> 15;
    }
    __syncthreads();
    const int out_x = x + threadIdx.x, out_y = y + threadIdx.y;
    if (out_x >= width || out_y >= height) return;
    const int row = threadIdx.y + Radius, col = threadIdx.x;
    int sum = coeff[0] * horizontal[row][col];
    #pragma unroll
    for (int k = 1; k <= Radius; ++k) sum += coeff[k] * (horizontal[row - k][col] + horizontal[row + k][col]);
    const int i = out_y * width + out_x;
    if (gaussian >= 0) pyramid[gaussian + i] = (sum + 16384) >> 15;
    if (dog >= 0) {
        const int diff = (sum - (pyramid[source + i] << 15) + 4096) >> 13;
        pyramid[dog + i] = max(-32768, min(32767, diff));
    }
}

#define SIFT_BLUR(radius) \
extern "C" __global__ void sift_blur_##radius(short* p, const short* weights, int offset, \
    int src, int gaussian, int dog, int width, int height) { \
    sift_blur<radius>(p, weights, offset, src, gaussian, dog, width, height); \
}

SIFT_BLUR(3)
SIFT_BLUR(4)
SIFT_BLUR(5)
SIFT_BLUR(6)
SIFT_BLUR(7)
SIFT_BLUR(9)
