/**
 * @NOTE: This file is adapted from
 * https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py
 * We:
 * 1. adapt from tilelang to pure cuda
 * 2. optimize the performance a little
 * 3. fix the potential illegal memory access
 */
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <optional>

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;
constexpr int kThreadsPerBlockVllm = 512;
constexpr size_t kSmem = 32 * 1024 * sizeof(uint32_t);  // 128KB

struct FastTopKParams {
  const float* __restrict__ input;  // [B, input_stride]
  int32_t* __restrict__ indices;    // [B, TopK]
  int32_t* __restrict__ lengths;    // [B]
  int64_t input_stride;
};

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

static inline __device__ uint16_t extractBinIdx(float x) {
  union {
    __half h;
    uint16_t u16;
  } tmp;
  tmp.h = __float2half_rn(x);
  tmp.u16 = (x < 0.f) ? (~tmp.u16 & 0xffff) : (tmp.u16 | 0x8000);
  return 511 - (tmp.u16 >> 7);
}

__device__ void fast_topk_cuda_tl(const float* __restrict__ input, int* __restrict__ index, int length) {
  // An optimized topk kernel copied from tilelang kernel
  // We assume length > TopK here, or it will crash
  int topk = TopK;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(convert_to_uint8(input[idx]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        /// NOTE: (dark) fuse the histogram computation here
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[TopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              /// NOTE: (dark) fuse the histogram computation here
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

// Follows 11 - 11 - 10 bit iterations
template <int kNumThreadsPerBlock = 512>
static __device__ void topKPerRowOptimized(const float* logits, int* outIndices, int length) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 2048;

  static constexpr int kNumBinsPerThread = kNumBins / kNumThreadsPerBlock;
  // The top-k width.
  static constexpr int kTopK = 2048;
  // The number of elements per thread for the final top-k sort.
  static constexpr int kNumTopKItemsPerThread = kTopK / kNumThreadsPerBlock;

  // The number of slots for the final pass.
  static constexpr int kNumFinalItems = 2048;
  // The number of elements per thread for the final sort.
  static constexpr int kNumFinalItemsPerThread = kNumFinalItems / kNumThreadsPerBlock;
  // The class to sort the elements during the final pass.
  using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;

  // The class to compute the inclusive prefix-sum over the histogram.
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

  // Shared memory to compute the block scan.
  __shared__ typename Scan::TempStorage smemScan;

  // The structure to store the final items (for the final pass).
  struct FinalItems {
    // Shared memory to store the indices for the final pass.
    int indices[kNumFinalItems];
    // Shared memory to store the logits for the final pass.
    float logits[kNumFinalItems];
  };

  // Shared memory to compute the block sort.
  __shared__ union {
    FinalItems items;
    typename FinalSort::TempStorage finalSort;
    // typename TopKSort::TempStorage topKSort;
  } smemFinal;

  // Shared memory to store the histogram.
  __shared__ int smemHistogram[kNumBins];
  // Shared memory to store the selected indices.
  __shared__ int smemIndices[kTopK];
  // Shared memory to store the threshold bin.
  __shared__ int smemThresholdBinIdx[1];
  // Shared memory counter to register the candidates for the final phase.
  __shared__ int smemFinalDstIdx[1];
  // Shared memory to determine if the threshold bin fits in the final items.
  __shared__ int smemFinalBinSize[1];
  // Shared memory to keep track of the top-k values found so far by the previous iterations
  __shared__ int smemFoundTopKValues[1];

  // Shortcut if the length of the row is smaller than Top-K. Indices are not
  // sorted by their corresponding logit.
  if (length <= kTopK) {
    for (int rowIt = threadIdx.x; rowIt < length; rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = rowIt;
    }
    for (int rowIt = length + threadIdx.x; rowIt < kTopK; rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = -1;
    }
    return;
  }

  // Clear the histogram.
#pragma unroll
  for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smemHistogram[idx] = 0;
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Fetch elements one-by-one.
  for (int idx = threadIdx.x; idx < length; idx += kNumThreadsPerBlock) {
    uint32_t binIdx = convert_to_uint32(logits[idx]);
    binIdx = 2047 - (binIdx >> 21);  // Take first 11 bits
    atomicAdd(&smemHistogram[binIdx], 1);
  }
  // Initialize values
  if (threadIdx.x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }

  // Make sure the histogram is ready.
  __syncthreads();
  const auto exclusiveScan = [&] {
    // Reads the value of the starting position in the smemIndices array
    int lastValue = smemFoundTopKValues[0];

    for (int round = 0; round < kNumBinsPerThread; round++) {
      // Read the values from SMEM.
      int idx = threadIdx.x + kNumThreadsPerBlock * round;
      int binCount{0};
      binCount = smemHistogram[idx];

      // Make sure each thread has read its value.
      __syncthreads();

      // Compute the prefix sum.
      int prefixSum{0}, totalSum{0};
      Scan(smemScan).ExclusiveSum(binCount, prefixSum, totalSum);

      // Update the histogram with the prefix sums.
      smemHistogram[idx] = prefixSum + lastValue;

      // Make sure the data is in shared memory.
      __syncthreads();

      // Find the last valid bin.
      bool foundThreshold = false;
      if (prefixSum + lastValue < kTopK) {
        int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1 ? totalSum + lastValue : smemHistogram[idx + 1];

        if (nextPrefixSum >= kTopK) {
          smemThresholdBinIdx[0] = idx;
          smemFinalBinSize[0] = nextPrefixSum - prefixSum - lastValue;
          smemFoundTopKValues[0] = prefixSum + lastValue;
          foundThreshold = true;
        }
      }

      // Early exit: if any thread found the threshold, we can skip remaining rounds
      if (__syncthreads_or(foundThreshold)) {
        break;
      }

      lastValue = totalSum + lastValue;
    }
  };

  exclusiveScan();

  // Make sure the data is in shared memory.
  __syncthreads();

  // The threshold bin.
  int thresholdBinIdx = smemThresholdBinIdx[0];

  // Iteration 1: Start with the first 11 bits
  // Fetch elements one-by-one and populate the shared memory buffers.
  for (int idx = threadIdx.x; idx < length; idx += kNumThreadsPerBlock) {
    float logit = logits[idx];
    uint32_t binIdx = convert_to_uint32(logits[idx]);
    binIdx = 2047 - (binIdx >> 21);

    if (binIdx < thresholdBinIdx) {
      int dstIdx = atomicAdd(&smemHistogram[binIdx], 1);
      smemIndices[dstIdx] = idx;
    } else if (binIdx == thresholdBinIdx) {
      // Only fill the final items if the threshold bin fits
      if (smemFinalBinSize[0] <= kNumFinalItems) {
        int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
        smemFinal.items.logits[dstIdx] = logit;
        smemFinal.items.indices[dstIdx] = idx;
      }
    }
  }

  // Make sure the elements are in shared memory.
  __syncthreads();

  const auto sortFinalItems = [&] {
    // The logits of the elements to be sorted in the final pass.
    float finalLogits[kNumFinalItemsPerThread];
    // The indices of the elements to be sorted in the final pass.
    int finalIndices[kNumFinalItemsPerThread];

#pragma unroll
    for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
      finalLogits[ii] = -FLT_MAX;
    }

// Read the elements from SMEM.
#pragma unroll
    for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
      int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
      if (srcIdx < smemFinalDstIdx[0]) {
        finalLogits[ii] = smemFinal.items.logits[srcIdx];
        finalIndices[ii] = smemFinal.items.indices[srcIdx];
      }
    }

    // Make sure the shared memory has been read.
    __syncthreads();

    // Sort the elements.
    FinalSort(smemFinal.finalSort).SortDescendingBlockedToStriped(finalLogits, finalIndices);

    // Copy the data back to the shared memory storage.
    int baseIdx = thresholdBinIdx > 0 ? smemHistogram[thresholdBinIdx - 1] : smemHistogram[0];

#pragma unroll
    for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
      int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
      int dstIdx = baseIdx + srcIdx;
      if (dstIdx < kTopK) {
        smemIndices[dstIdx] = finalIndices[ii];
      }
    }
  };

  if (smemFinalBinSize[0] <= kNumFinalItems) {
    sortFinalItems();
    // Make sure the data is in shared memory.
    __syncthreads();
  } else {
    // Iteration 2: Continue with the next 11 bits
    // Clear the histogram.
#pragma unroll
    for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
      smemHistogram[idx] = 0;
    }

    // Make sure the histogram is ready.
    __syncthreads();

    // Fetch elements one-by-one.
    // Only process elements that were in the Round 1 threshold bin
    int it1ThresholdBinIdx = smemThresholdBinIdx[0];
    for (int idx = threadIdx.x; idx < length; idx += kNumThreadsPerBlock) {
      uint32_t it1BinIdx = convert_to_uint32(logits[idx]);
      it1BinIdx = 2047 - (it1BinIdx >> 21);

      // Only process if this element was in the Round 1 threshold bin
      if (it1BinIdx == it1ThresholdBinIdx) {
        uint32_t binIdx = convert_to_uint32(logits[idx]);
        binIdx = 2047 - ((binIdx >> 10) & 0x7FF);  // Take next 11 bits
        atomicAdd(&smemHistogram[binIdx], 1);
      }
    }

    // Make sure the histogram is ready.
    __syncthreads();
    exclusiveScan();
    __syncthreads();

    thresholdBinIdx = smemThresholdBinIdx[0];
    // Fetch elements one-by-one and populate the shared memory buffers.
    for (int idx = threadIdx.x; idx < length; idx += kNumThreadsPerBlock) {
      uint32_t it1BinIdx = convert_to_uint32(logits[idx]);
      it1BinIdx = 2047 - (it1BinIdx >> 21);

      // Only process if this element was in the Round 1 threshold bin
      if (it1BinIdx == it1ThresholdBinIdx) {
        float logit = logits[idx];
        uint32_t binIdx = convert_to_uint32(logits[idx]);
        binIdx = 2047 - ((binIdx >> 10) & 0x7FF);  // Take next 11 bits

        if (binIdx < thresholdBinIdx) {
          int dstIdx = atomicAdd(&smemHistogram[binIdx], 1);
          smemIndices[dstIdx] = idx;
        } else if (binIdx == thresholdBinIdx) {
          // Only fill the final items if the threshold bin fits
          if (smemFinalBinSize[0] <= kNumFinalItems) {
            int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
            smemFinal.items.logits[dstIdx] = logit;
            smemFinal.items.indices[dstIdx] = idx;
          }
        }
      }
    }

    // Make sure the elements are in shared memory.
    __syncthreads();

    if (smemFinalBinSize[0] <= kNumFinalItems) {
      sortFinalItems();
      // Make sure the data is in shared memory.
      __syncthreads();
    } else {
      // Iteration 3: Continue with the last 10 bits
      // Clear the histogram.
#pragma unroll
      for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
        smemHistogram[idx] = 0;
      }

      // Make sure the histogram is ready.
      __syncthreads();

      // Fetch elements one-by-one.
      // Only process elements that were in both Round 1 AND Round 2 threshold bins
      int it2ThresholdBinIdx = smemThresholdBinIdx[0];
      for (int idx = threadIdx.x; idx < length; idx += kNumThreadsPerBlock) {
        uint32_t it1BinIdx = convert_to_uint32(logits[idx]);
        it1BinIdx = 2047 - (it1BinIdx >> 21);

        uint32_t it2BinIdx = convert_to_uint32(logits[idx]);
        it2BinIdx = 2047 - ((it2BinIdx >> 10) & 0x7FF);

        // Only process if this element was in both threshold bins
        if (it1BinIdx == it1ThresholdBinIdx && it2BinIdx == it2ThresholdBinIdx) {
          uint32_t binIdx = convert_to_uint32(logits[idx]);
          binIdx = 1023 - (binIdx & 0x3FF);  // Take last 10 bits
          atomicAdd(&smemHistogram[binIdx], 1);
        }
      }

      // Make sure the histogram is ready.
      __syncthreads();
      exclusiveScan();
      __syncthreads();

      thresholdBinIdx = smemThresholdBinIdx[0];
      // Fetch elements one-by-one and populate the shared memory buffers.
      for (int idx = threadIdx.x; idx < length; idx += kNumThreadsPerBlock) {
        uint32_t it1BinIdx = convert_to_uint32(logits[idx]);
        it1BinIdx = 2047 - (it1BinIdx >> 21);

        uint32_t it2BinIdx = convert_to_uint32(logits[idx]);
        it2BinIdx = 2047 - ((it2BinIdx >> 10) & 0x7FF);

        // Only process if this element was in both threshold bins
        if (it1BinIdx == it1ThresholdBinIdx && it2BinIdx == it2ThresholdBinIdx) {
          float logit = logits[idx];
          uint32_t binIdx = convert_to_uint32(logits[idx]);
          binIdx = 1023 - (binIdx & 0x3FF);

          if (binIdx < thresholdBinIdx) {
            int dstIdx = atomicAdd(&smemHistogram[binIdx], 1);
            smemIndices[dstIdx] = idx;
          } else if (binIdx == thresholdBinIdx) {
            int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
            if (dstIdx < kNumFinalItems) {
              smemFinal.items.logits[dstIdx] = logit;
              smemFinal.items.indices[dstIdx] = idx;
            }
          }
        }
      }
      __syncthreads();

      sortFinalItems();

      __syncthreads();
    }
  }

// Store to global memory.
#pragma unroll
  for (int ii = 0; ii < kNumTopKItemsPerThread; ++ii) {
    int offset = ii * kNumThreadsPerBlock + threadIdx.x;
    outIndices[offset] = smemIndices[offset];
  }
}

// copied from vllm
template <int kNumThreadsPerBlock = 512>
static __device__ void topKPerRow(const float* logits, int* outIndices, int length) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 512;

  // The top-k width.
  static constexpr int kTopK = 2048;
  // The number of elements per thread for the final top-k sort.
  static constexpr int kNumTopKItemsPerThread = kTopK / kNumThreadsPerBlock;
  // The class to sort the elements during the final top-k sort.
  using TopKSort = cub::BlockRadixSort<float, kNumThreadsPerBlock, kNumTopKItemsPerThread, int>;

  // The number of slots for the final pass.
  static constexpr int kNumFinalItems = 3072;
  // The number of elements per thread for the final sort.
  static constexpr int kNumFinalItemsPerThread = kNumFinalItems / kNumThreadsPerBlock;
  // The class to sort the elements during the final pass.
  using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;

  // The class to compute the inclusive prefix-sum over the histogram.
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

  // Shared memory to compute the block scan.
  __shared__ typename Scan::TempStorage smemScan;

  // The structure to store the final items (for the final pass).
  struct FinalItems {
    // Shared memory to store the indices for the final pass.
    int indices[kNumFinalItems];
    // Shared memory to store the logits for the final pass.
    float logits[kNumFinalItems];
  };

  // Shared memory to compute the block sort.
  __shared__ union {
    FinalItems items;
    typename FinalSort::TempStorage finalSort;
    typename TopKSort::TempStorage topKSort;
  } smemFinal;

  // Shared memory to store the histogram.
  __shared__ int smemHistogram[kNumBins];
  // Shared memory to store the selected indices.
  __shared__ int smemIndices[kTopK];
  // Shared memory to store the selected logits.
  __shared__ float smemLogits[kTopK];
  // Shared memory to store the threshold bin.
  __shared__ int smemThresholdBinIdx[1];
  // Shared memory counter to register the candidates for the final phase.
  __shared__ int smemFinalDstIdx[1];

  // Shortcut if the length of the row is smaller than Top-K. Indices are not
  // sorted by their corresponding logit.
  if (length <= kTopK) {
    for (int rowIt = threadIdx.x; rowIt < length; rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = rowIt;
    }
    for (int rowIt = length + threadIdx.x; rowIt < kTopK; rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = -1;
    }
    return;
  }

  // Clear the histogram.
  if (threadIdx.x < kNumBins) {
    smemHistogram[threadIdx.x] = 0;
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Fetch elements one-by-one.
  for (int idx = threadIdx.x; idx < length; idx += kNumThreadsPerBlock) {
    uint16_t binIdx = extractBinIdx(logits[idx]);
    atomicAdd(&smemHistogram[binIdx], 1);
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Read the values from SMEM.
  int binCount{0};
  if (threadIdx.x < kNumBins) {
    binCount = smemHistogram[threadIdx.x];
  }

  // Make sure each thread has read its value.
  __syncthreads();

  // Compute the prefix sum.
  int prefixSum{0}, totalSum{0};
  Scan(smemScan).ExclusiveSum(binCount, prefixSum, totalSum);

  // Update the histogram with the prefix sums.
  if (threadIdx.x < kNumBins) {
    smemHistogram[threadIdx.x] = prefixSum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // Find the last valid bin.
  if (threadIdx.x < kNumBins) {
    int nextPrefixSum = threadIdx.x == kNumBins - 1 ? totalSum : smemHistogram[threadIdx.x + 1];
    if (prefixSum < kTopK && nextPrefixSum >= kTopK) {
      smemThresholdBinIdx[0] = threadIdx.x;
    }
  }

  // Clear the counter to store the items for the final phase.
  if (threadIdx.x == 0) {
    smemFinalDstIdx[0] = 0;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The threshold bin.
  int thresholdBinIdx = smemThresholdBinIdx[0];

  // Fetch elements one-by-one and populate the shared memory buffers.
  for (int idx = threadIdx.x; idx < length; idx += kNumThreadsPerBlock) {
    float logit = logits[idx];
    uint16_t binIdx = extractBinIdx(logit);
    if (binIdx < thresholdBinIdx) {
      int dstIdx = atomicAdd(&smemHistogram[binIdx], 1);
      smemLogits[dstIdx] = logit;
      smemIndices[dstIdx] = idx;
    } else if (binIdx == thresholdBinIdx) {
      int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
      if (dstIdx < kNumFinalItems) {
        smemFinal.items.logits[dstIdx] = logit;
        smemFinal.items.indices[dstIdx] = idx;
      }
    }
  }

  // Make sure the elements are in shared memory.
  __syncthreads();

  // The logits of the elements to be sorted in the final pass.
  float finalLogits[kNumFinalItemsPerThread];
  // The indices of the elements to be sorted in the final pass.
  int finalIndices[kNumFinalItemsPerThread];

// Init.
#pragma unroll
  for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
    finalLogits[ii] = -FLT_MAX;
  }

// Read the elements from SMEM.
#pragma unroll
  for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
    int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
    if (srcIdx < smemFinalDstIdx[0]) {
      finalLogits[ii] = smemFinal.items.logits[srcIdx];
      finalIndices[ii] = smemFinal.items.indices[srcIdx];
    }
  }

  // Make sure the shared memory has been read.
  __syncthreads();

  // Sort the elements.
  FinalSort(smemFinal.finalSort).SortDescendingBlockedToStriped(finalLogits, finalIndices);

  // Copy the data back to the shared memory storage.
  int baseIdx = thresholdBinIdx > 0 ? smemHistogram[thresholdBinIdx - 1] : 0;
#pragma unroll
  for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
    int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
    int dstIdx = baseIdx + srcIdx;
    if (dstIdx < kTopK) {
      smemLogits[dstIdx] = finalLogits[ii];
      smemIndices[dstIdx] = finalIndices[ii];
    }
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The topK logits.
  float topKLogits[kNumTopKItemsPerThread];
  // The topK indices.
  int topKIndices[kNumTopKItemsPerThread];

// Load from shared memory.
#pragma unroll
  for (int ii = 0; ii < kNumTopKItemsPerThread; ++ii) {
    topKLogits[ii] = smemLogits[ii * kNumThreadsPerBlock + threadIdx.x];
    topKIndices[ii] = smemIndices[ii * kNumThreadsPerBlock + threadIdx.x];
  }

  // Sort the elements.
  TopKSort(smemFinal.topKSort).SortDescendingBlockedToStriped(topKLogits, topKIndices);

// Store to global memory.
#pragma unroll
  for (int ii = 0; ii < kNumTopKItemsPerThread; ++ii) {
    int offset = ii * kNumThreadsPerBlock + threadIdx.x;
    outIndices[offset] = topKIndices[ii];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // topk
    void topk_kernel(const FastTopKParams params) {
  const auto& [input, indices, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length);
  } else {
    return fast_topk_cuda_tl(score, indice, length);
  }
}
__global__ __launch_bounds__(kThreadsPerBlockVllm)  // topk
    void vllm_topk_kernel(const FastTopKParams params) {
  const auto& [input, indices, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;
  return topKPerRow<kThreadsPerBlockVllm>(score, indice, length);
}

__global__ __launch_bounds__(kThreadsPerBlockVllm)  // topk
    void vllm_topk_kernel_optimized(const FastTopKParams params) {
  const auto& [input, indices, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;
  return topKPerRowOptimized<512>(score, indice, length);
}

__global__ __launch_bounds__(kThreadsPerBlock)  // decode
    void topk_transform_decode_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride) {
  const auto& [input, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill
    void topk_transform_prefill_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride,
        const int32_t* __restrict__ cu_seqlens_q,
        const int64_t prefill_bs) {
  const auto& [input, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;

  /// NOTE: prefill bs is usually small, we can just use a simple loop here
  /// We ensure that last cu_seqlens is equal to number of blocks launched
  __shared__ const int32_t* s_src_page_entry;
  if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
    if (tid < prefill_bs) {
      if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const auto src_page_entry = s_src_page_entry;

  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

auto get_params(at::Tensor score, at::Tensor lengths, std::optional<at::Tensor> indices_opt = std::nullopt)
    -> FastTopKParams {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(lengths.size(0) == B);
  int32_t* indices_data_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    TORCH_CHECK(indices.size(1) == TopK);
    indices_data_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .indices = indices_data_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result =
      [] { return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem); }();
  TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void fast_topk_interface(at::Tensor score, at::Tensor indices, at::Tensor lengths) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  CHECK_CUDA(lengths);
  const auto params = get_params(score, lengths, indices);
  const auto B = score.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  setup_kernel_smem_once<topk_kernel, kSmem>();
  topk_kernel<<<grid, block, kSmem, stream>>>(params);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void vllm_fast_topk_interface(at::Tensor score, at::Tensor indices, at::Tensor lengths) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  CHECK_CUDA(lengths);
  const auto params = get_params(score, lengths, indices);
  const auto B = score.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlockVllm};
  vllm_topk_kernel<<<grid, block, 0, stream>>>(params);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void vllm_fast_topk_optimized_interface(at::Tensor score, at::Tensor indices, at::Tensor lengths) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  CHECK_CUDA(lengths);
  const auto params = get_params(score, lengths, indices);
  const auto B = score.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlockVllm};
  vllm_topk_kernel_optimized<<<grid, block, 0, stream>>>(params);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_interface(
    at::Tensor score,
    at::Tensor lengths,
    at::Tensor dst_page_table,
    at::Tensor src_page_table,
    at::Tensor cu_seqlens_q) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(dst_page_table);
  CHECK_CUDA(src_page_table);
  CHECK_CUDA(cu_seqlens_q);
  const auto params = get_params(score, lengths);
  const auto B = score.size(0);
  TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
  TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.stride(1) == 1);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous());
  const auto prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(dst_page_table.size(0) == B);
  TORCH_CHECK(dst_page_table.size(1) == TopK);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(prefill_bs <= B);  // prefill_bs should be smaller than expanded bs

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  const auto src_stride = src_page_table.stride(0);

  // dispatch to decode or prefill
  const auto is_decode = (prefill_bs == B);
  if (is_decode) {
    setup_kernel_smem_once<topk_transform_decode_kernel, kSmem>();
    topk_transform_decode_kernel<<<grid, block, kSmem, stream>>>(
        params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride);
  } else {
    setup_kernel_smem_once<topk_transform_prefill_kernel, kSmem>();
    topk_transform_prefill_kernel<<<grid, block, kSmem, stream>>>(
        params,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride,
        cu_seqlens_q.data_ptr<int32_t>(),
        prefill_bs);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}
