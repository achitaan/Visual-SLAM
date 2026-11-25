using RankKey = unsigned long long;

__device__ inline void insert_rank(RankKey candidate, RankKey& first, RankKey& second) {
    if (candidate < first) {
        second = first;
        first = candidate;
    } else if (candidate < second) {
        second = candidate;
    }
}

__device__ inline void merge_ranks(RankKey other_first, RankKey other_second,
                                   RankKey& first, RankKey& second) {
    if (other_first < first) {
        second = other_second < first ? other_second : first;
        first = other_first;
    } else if (other_first < second) {
        second = other_first;
    }
}

__device__ inline void reduce_warp(RankKey& first, RankKey& second) {
    for (int offset = 16; offset > 0; offset /= 2) {
        RankKey other_first = __shfl_down_sync(0xffffffff, first, offset);
        RankKey other_second = __shfl_down_sync(0xffffffff, second, offset);
        merge_ranks(other_first, other_second, first, second);
    }
}

__device__ inline RankKey square_distance(float dot, float first, float second, int index) {
    // Quantized descriptors make every intermediate exactly representable as float32.
    const unsigned int distance = static_cast<unsigned int>(first + second - 2.0f * dot);
    return (static_cast<RankKey>(distance) << 32) | static_cast<unsigned int>(index);
}

__device__ inline void export_ranks(int* output, int row, RankKey first, RankKey second) {
    output[3 * row] = static_cast<int>(first >> 32);
    output[3 * row + 1] = static_cast<int>(second >> 32);
    output[3 * row + 2] = static_cast<int>(first & 0xffffffffu);
}

extern "C" __global__ void rank_descriptor_dots(
    const float* dots, const float* left_norm, const float* right_norm,
    int* output, int rows, int columns
) {
    constexpr RankKey no_rank = ~RankKey(0);
    constexpr int column_tile = 16;
    constexpr int row_workers = 128 / column_tile;
    __shared__ RankKey firsts[128];
    __shared__ RankKey seconds[128];
    int tile = static_cast<int>(blockIdx.x);
    int lane = static_cast<int>(threadIdx.x);
    RankKey first = no_rank;
    RankKey second = no_rank;
    if (tile < rows) {
        const float norm = left_norm[tile];
        for (int i = lane; i < columns; i += 128) {
            auto key = square_distance(dots[tile * columns + i], norm, right_norm[i], i);
            insert_rank(key, first, second);
        }
        reduce_warp(first, second);
        if (lane % 32 == 0) {
            firsts[lane / 32] = first;
            seconds[lane / 32] = second;
        }
        __syncthreads();
        if (lane == 0) {
            for (int i = 1; i < 4; ++i) {
                merge_ranks(firsts[i], seconds[i], first, second);
            }
            export_ranks(output, tile, first, second);
        }
        return;
    }

    // Neighboring lanes read neighboring columns of the same descriptor row.
    const int column = (tile - rows) * column_tile + lane % column_tile;
    if (column < columns) {
        const float norm = right_norm[column];
        for (int row = lane / column_tile; row < rows; row += row_workers) {
            auto key = square_distance(dots[row * columns + column], left_norm[row], norm, row);
            insert_rank(key, first, second);
        }
    }
    firsts[lane] = first;
    seconds[lane] = second;
    __syncthreads();
    if (lane < column_tile && column < columns) {
        for (int slot = lane + column_tile; slot < 128; slot += column_tile) {
            merge_ranks(firsts[slot], seconds[slot], first, second);
        }
        export_ranks(output, rows + column, first, second);
    }
}
