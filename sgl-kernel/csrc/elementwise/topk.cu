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
constexpr int WARP_SIZE = 32;
using WideT = float4;

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
}

template <int step>
static inline __device__ uint32_t extractBinIdx(float x) {
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;

  if constexpr (step == 0) {
    return bits >> 21;
  } else if constexpr (step == 1) {
    return (bits >> 10) & 0x7ff;
  } else {
    return bits & 0x3ff;
  }
}

template <int shift>
static inline __device__ bool isPartialMatch(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  uint32_t bits = __float_as_uint(x);
  bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
  return (bits ^ pattern) >> shift == 0;
}

/**
 * Map a Func over the input data, using vectorized load instructions if
 * possible.
 *
 * NB: in future, we should move this to
 * cpp/include/raft/linalg/detail/unary_op.cuh, which currently does not support
 * the second lambda argument (index of an element)
 *
 * @tparam T element type
 * @tparam IdxT indexing type
 * @tparam Func void (T x, IdxT idx)
 *
 * @param thread_rank rank of the calling thread among all participating threads
 * @param num_threads number of the threads that participate in processing
 * @param in the input data
 * @param len the number of elements to read
 * @param f the lambda taking two arguments (T x, IdxT idx)
 */
template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads, const T* in, idxT len, Func f) {
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (idxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    // TODO: it's UB
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
                       ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
                       : 0;
    if (skip_cnt > len) {
      skip_cnt = len;
    }
    const WideT* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const idxT len_cast = (len - skip_cnt) / items_per_scalar;

    for (idxT i = thread_rank; i < len_cast; i += num_threads) {
      wide.scalar = in_cast[i];
      const idxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    static_assert(WARP_SIZE >= items_per_scalar);
    // and because items_per_scalar > skip_cnt, WARP_SIZE > skip_cnt
    // no need to use loop
    if (thread_rank < skip_cnt) {
      f(in[thread_rank], thread_rank);
    }
    // because len_cast = (len - skip_cnt) / items_per_scalar,
    // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
    // and so
    // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <=
    // WARP_SIZE no need to use loop
    const idxT remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
    if (remain_i < len) {
      f(in[remain_i], remain_i);
    }
  }
}

template <
    int step,
    int kNumThreadsPerBlock,
    int kNumBins,
    int kTopK,
    int kNumFinalItems,
    bool multipleBlocksPerRow,
    bool mergeBlocks,
    typename SmemFinalType,
    typename SmemOutputType>
__device__ bool processHistogramStep(
    const int* indices,
    const float* logits,
    int rowEnd,
    uint32_t& logitPattern,
    int& thresholdBinIdx,
    SmemOutputType& smemOutput,
    int* smemThresholdBinIdx,
    int* smemFinalDstIdx,
    int* smemFinalBinSize,
    int* smemFoundTopKValues,
    SmemFinalType& smemFinal,
    int stride1,
    int rowStart,
    int offset = 0) {
  // Clear the histogram.
#pragma unroll
  for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock) {
    smemFinal.histo.data[idx] = 0;
  }

  // Make sure the histogram is ready.
  __syncthreads();

  // Update pattern
  constexpr auto patternShift = step == 0 ? 0 : step == 1 ? 21 : 10;
  if constexpr (step == 1) {
    logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
  } else if constexpr (step == 2) {
    logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
  }

  auto distributeToBins = [&](float logit, int /* idx */ = 0) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      atomicAdd(&smemFinal.histo.data[binIdx], 1);
    }
  };

  // Distribute the elements to the histogram bins.
  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, distributeToBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      distributeToBins(logit, idx);
    }
  }
  // Make sure the histogram is ready.
  __syncthreads();

  // Reads the value of the starting position in the smemOutput array
  int lastValue = smemFoundTopKValues[0];

  for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
    // Read the values from SMEM.
    int idx = threadIdx.x + kNumThreadsPerBlock * round;
    int binCount{0};
    binCount = smemFinal.histo.data[idx];

    // Make sure each thread has read its value.
    __syncthreads();

    // Compute the prefix sum.
    int prefixSum{0}, totalSum{0};
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
    Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);

    // Update the histogram with the prefix sums.
    prefixSum += lastValue;
    totalSum += lastValue;
    smemFinal.histo.data[idx] = prefixSum;

    // Make sure the data is in shared memory.
    __syncthreads();

    // Find the last valid bin.
    bool foundThreshold = false;
    if (prefixSum < kTopK) {
      int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1 ? totalSum : smemFinal.histo.data[idx + 1];

      if (nextPrefixSum >= kTopK) {
        smemThresholdBinIdx[0] = idx;
        smemFinalBinSize[0] = nextPrefixSum - prefixSum;
        foundThreshold = true;
      }
    }

    // Early exit: if any thread found the threshold, we can skip remaining rounds
    if (__syncthreads_or(foundThreshold)) {
      break;
    }

    lastValue = totalSum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The threshold bin.
  thresholdBinIdx = smemThresholdBinIdx[0];

  auto processBins = [&](float logit, int idx) {
    if (isPartialMatch<patternShift>(logit, logitPattern)) {
      uint32_t binIdx = extractBinIdx<step>(logit);
      if (binIdx < thresholdBinIdx) {
        // The element is part of the top-k selection
        int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);

        if constexpr (mergeBlocks) {
          smemOutput.indices[dstIdx] = indices[idx];
        } else if constexpr (multipleBlocksPerRow) {
          smemOutput.indices[dstIdx] = idx + rowStart;
          smemOutput.logits[dstIdx] = logit;
        } else {
          smemOutput.indices[dstIdx] = idx;
        }
      }
      if constexpr (step < 2) {
        // Only fill the final items for sorting if the threshold bin fits
        if (binIdx == thresholdBinIdx && smemFinalBinSize[0] <= kNumFinalItems) {
          int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
          smemFinal.items.logits[dstIdx] = logit;
          if constexpr (mergeBlocks) {
            smemFinal.items.indices[dstIdx] = indices[idx];
          } else if constexpr (multipleBlocksPerRow) {
            smemFinal.items.indices[dstIdx] = idx + rowStart;
          } else {
            smemFinal.items.indices[dstIdx] = idx;
          }
        } else if (binIdx == thresholdBinIdx && smemFinalBinSize[0] > kNumFinalItems) {
          // Load elements for next it
        }
      } else {
        if (binIdx == thresholdBinIdx) {
          // The elements in the threshold bin share the same 32 bits at step 2
          int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
          if (dstIdx < kTopK) {
            if constexpr (mergeBlocks) {
              smemOutput.indices[dstIdx] = indices[idx];

            } else if constexpr (multipleBlocksPerRow) {
              smemOutput.indices[dstIdx] = idx + rowStart;
              smemOutput.logits[dstIdx] = logit;
            } else {
              smemOutput.indices[dstIdx] = idx;
            }
          }
        }
      }
    }
  };

  if (stride1 == 1) {
    vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, processBins);
  } else {
    for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock) {
      float logit = logits[idx * stride1];
      processBins(logit, idx);
    }
  }

  // Make sure the elements are in shared memory.
  __syncthreads();

  // Check if we should continue to next step
  return smemFinalBinSize[0] > kNumFinalItems;
}

template <
    int kNumThreadsPerBlock,
    int kNumBins,
    int kTopK,
    bool useRadixSort,
    bool multipleBlocksPerRow,
    bool mergeBlocks,
    bool fuseTransformDecode,
    bool fuseTransformPrefill>
static __device__ void topKPerRowJob(
    const int* indices,
    const float* logits,
    int rowStart,
    int rowEnd,
    int* outIndices,
    float* outLogits,
    int stride1,
    int offset = 0,
    const int32_t* __restrict__ src_page_table = nullptr,
    int src_stride = 0,
    const int32_t* __restrict__ cu_seqlens_q = nullptr,
    int prefill_bs = 0) {
  // The number of slots for the final pass.
  static constexpr int kNumFinalItems = 2048;
  // The number of elements per thread for the final sort.
  static constexpr int kNumFinalItemsPerThread = kNumFinalItems / kNumThreadsPerBlock;
  // The class to sort the elements during the final pass.
  using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;
  using FinalSortTempStorage = std::conditional_t<useRadixSort, typename FinalSort::TempStorage, int>;
  // The class to compute the inclusive prefix-sum over the histogram.
  using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

  // The structure to store the final items (for the final pass).
  struct FinalItems {
    // Shared memory to store the indices for the final pass.
    int indices[kNumFinalItems];
    // Shared memory to store the logits for the final pass.
    float logits[kNumFinalItems];
  };

  struct Histogram {
    typename Scan::TempStorage scan;
    int data[kNumBins];
  };

  // Shared memory to compute the block sort.
  __shared__ union {
    FinalItems items;
    FinalSortTempStorage finalSort;
    Histogram histo;
  } smemFinal;

  // Shared memory to store the selected indices.
  // If we are processing using multiple blocks, we need to store the logits and indices.
  struct SmemOutputIndices {
    int indices[kTopK];
  };
  struct SmemOutputLogitsAndIndices {
    int indices[kTopK];
    float logits[kTopK];
  };
  using SmemOutput_t = std::conditional_t<multipleBlocksPerRow, SmemOutputLogitsAndIndices, SmemOutputIndices>;
  __shared__ SmemOutput_t smemOutput;

  // Shared memory to store the threshold bin.
  __shared__ int smemThresholdBinIdx[1];
  // Shared memory counter to register the candidates for the final phase.
  __shared__ int smemFinalDstIdx[1];
  // Shared memory to determine if the threshold bin fits in the final items.
  __shared__ int smemFinalBinSize[1];
  // Shared memory to keep track of the top-k values found so far by the previous iterations
  __shared__ int smemFoundTopKValues[1];

  // The length of the row.
  int rowLen = rowEnd - rowStart;

  __shared__ const int32_t* s_src_page_entry;
  if constexpr (fuseTransformPrefill) {
    for (int64_t i = threadIdx.x; i < prefill_bs; i += kNumThreadsPerBlock) {
      if (blockIdx.x >= cu_seqlens_q[i] && blockIdx.x < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  if constexpr (fuseTransformDecode) {
    s_src_page_entry = src_page_table + blockIdx.x * src_stride;
  }

  __syncthreads();

  // Shortcut if the length of the row is smaller than Top-K. Indices are not
  // sorted by their corresponding logit.
  if (rowLen <= kTopK) {
    for (int rowIt = threadIdx.x; rowIt < rowLen; rowIt += kNumThreadsPerBlock) {
      if constexpr (fuseTransformDecode || fuseTransformPrefill) {
        outIndices[rowIt] = s_src_page_entry[rowIt + offset];

      } else {
        if constexpr (multipleBlocksPerRow) {
          outIndices[rowIt] = rowIt + offset + rowStart;
          outLogits[rowIt] = logits[rowIt + rowStart];
        } else {
          outIndices[rowIt] = rowIt + offset;
        }
      }
    }
    // maybe not execute when not fusetrans
    for (int rowIt = rowLen + threadIdx.x; rowIt < kTopK; rowIt += kNumThreadsPerBlock) {
      outIndices[rowIt] = -1;

      if constexpr (multipleBlocksPerRow) {
        outLogits[rowIt] = -FLT_MAX;
      }
    }

    return;
  }
  // Initialize values
  if (threadIdx.x == 0) {
    smemFinalDstIdx[0] = 0;
    smemFoundTopKValues[0] = 0;
  }
  __syncthreads();
  int thresholdBinIdx = -1;
  uint32_t logitPattern = 0;

  // Step 0: Process first 11 bits
  bool continueToNextStep =
      processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kTopK, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
          indices,
          logits,
          rowEnd,
          logitPattern,
          thresholdBinIdx,
          smemOutput,
          smemThresholdBinIdx,
          smemFinalDstIdx,
          smemFinalBinSize,
          smemFoundTopKValues,
          smemFinal,
          stride1,
          rowStart);

  if (continueToNextStep) {
    // Step 1: Process next 11 bits
    continueToNextStep = processHistogramStep<
        1,
        kNumThreadsPerBlock,
        kNumBins,
        kTopK,
        kNumFinalItems,
        multipleBlocksPerRow,
        mergeBlocks>(
        indices,
        logits,
        rowEnd,
        logitPattern,
        thresholdBinIdx,
        smemOutput,
        smemThresholdBinIdx,
        smemFinalDstIdx,
        smemFinalBinSize,
        smemFoundTopKValues,
        smemFinal,
        stride1,
        rowStart);

    if (continueToNextStep) {
      // Step 2: Process final 10 bits
      processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kTopK, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
          indices,
          logits,
          rowEnd,
          logitPattern,
          thresholdBinIdx,
          smemOutput,
          smemThresholdBinIdx,
          smemFinalDstIdx,
          smemFinalBinSize,
          smemFoundTopKValues,
          smemFinal,
          stride1,
          rowStart);
    }
  }

  if (!continueToNextStep) {
    // The histogram did not proceed to the final 10 bits, therefore we need to sort the final
    // items The logits of the elements to be sorted in the final pass.
    if constexpr (useRadixSort) {
      // Sorting with radix sort
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
      int baseIdx = smemFoundTopKValues[0];

#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
        int dstIdx = baseIdx + srcIdx;

        if (dstIdx < kTopK) {
          smemOutput.indices[dstIdx] = finalIndices[ii];
          if constexpr (multipleBlocksPerRow) {
            smemOutput.logits[dstIdx] = finalLogits[ii];
          }
        }
      }
    } else {
      // Sorting with insertion sort
      auto baseIdx = smemFoundTopKValues[0];
      for (int i = threadIdx.x; i < smemFinalDstIdx[0]; i += kNumThreadsPerBlock) {
        int outIndex = 0;
        auto logit = smemFinal.items.logits[i];

        for (int j = 0; j < smemFinalDstIdx[0]; j++) {
          auto otherLogit = smemFinal.items.logits[j];
          if (logit < otherLogit || (logit == otherLogit && i < j)) {
            outIndex++;
          }
        }
        // Store if outIndex is in bounds
        if (outIndex + baseIdx < kTopK) {
          smemOutput.indices[outIndex + baseIdx] = smemFinal.items.indices[i];
          if constexpr (multipleBlocksPerRow) {
            smemOutput.logits[outIndex + baseIdx] = smemFinal.items.logits[i];
          }
        }
      }
    }
    __syncthreads();
  }

  // Store to global memory.
  for (int i = threadIdx.x; i < kTopK; i += kNumThreadsPerBlock) {
    if constexpr (fuseTransformDecode || fuseTransformPrefill) {
      outIndices[i] = s_src_page_entry[smemOutput.indices[i] + offset];
    } else {
      if constexpr (multipleBlocksPerRow) {
        outIndices[i] = smemOutput.indices[i];
        outLogits[i] = smemOutput.logits[i];
      } else {
        outIndices[i] = smemOutput.indices[i] + offset;
      }
    }
  }
}

template <
    int kNumThreadsPerBlock,
    bool useRadixSort,
    bool multipleBlocksPerRow = false,
    bool mergeBlocks = false,
    bool fuseTransformDecode = false,
    bool fuseTransformPrefill = false>
static __global__ __launch_bounds__(kNumThreadsPerBlock) void topKPerRow(
    const float* logits,
    const int* rowStarts,
    const int* rowEnds,
    int* outIndices,
    float* outLogits,
    int stride0,
    int stride1,
    const int offsetIndex = 0,
    const int numBlocksToMerge = 0,
    const int* indices = nullptr,
    const int* offsetPerRow = nullptr,
    const int32_t* __restrict__ src_page_table = nullptr,
    int src_stride = 0,
    const int32_t* __restrict__ cu_seqlens_q = nullptr,
    int prefill_bs = 0) {
  // The number of bins in the histogram.
  static constexpr int kNumBins = 2048;

  // The top-k width.
  static constexpr int kTopK = 2048;

  // The row computed by this block.
  int rowIdx = blockIdx.x + offsetIndex;

  // The range of logits within the row.
  int offset = offsetPerRow ? offsetPerRow[rowIdx] : 0;
  int rowStart = rowStarts ? rowStarts[rowIdx] : 0;
  int rowEnd = rowEnds[rowIdx];

  if constexpr (!multipleBlocksPerRow && !mergeBlocks) {
    outIndices += rowIdx * kTopK;
  } else if constexpr (multipleBlocksPerRow) {
    const auto blockSize = rowEnd / gridDim.y;  // 16384 / 2 = 8192
    rowStart = blockSize * blockIdx.y;          // 8192 * 1 = 8192
    rowEnd = gridDim.y == blockIdx.y + 1 ? rowEnd : rowStart + blockSize;
    outIndices += rowIdx * gridDim.y * kTopK + blockIdx.y * kTopK;
    outLogits += rowIdx * gridDim.y * kTopK + blockIdx.y * kTopK;
  } else if constexpr (mergeBlocks) {
    rowEnd = numBlocksToMerge * kTopK;
    indices += rowIdx * numBlocksToMerge * kTopK;
    outIndices += rowIdx * kTopK;
  }

  // Local pointers to this block
  logits += rowIdx * stride0;

  topKPerRowJob<
      kNumThreadsPerBlock,
      kNumBins,
      kTopK,
      useRadixSort,
      multipleBlocksPerRow,
      mergeBlocks,
      fuseTransformDecode,
      fuseTransformPrefill>(
      indices,
      logits,
      rowStart,
      rowEnd,
      outIndices,
      outLogits,
      stride1,
      offset,
      src_page_table,
      src_stride,
      cu_seqlens_q,
      prefill_bs);
}
}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void fast_topk_interface(
    at::Tensor logits,
    at::Tensor outIndices,
    at::Tensor rowStarts,
    at::Tensor rowEnds,
    at::Tensor outIndicesAux,
    at::Tensor outLogitsAux,
    bool isPrefill) {
  CHECK_CUDA(logits);
  CHECK_CUDA(outIndices);
  CHECK_CUDA(rowEnds);

  constexpr int kTopK = 2048;
  // Threshold of number of elements to use radix / insertion sort.
  constexpr int kSortingAlgorithmThreshold = 12288;
  const auto logitsPtr = logits.data_ptr<float>();

  const auto outIndicesPtr = outIndices.data_ptr<int32_t>();
  const auto rowEndsPtr = rowEnds.data_ptr<int32_t>();
  const auto numRows = logits.size(0);
  const auto numColumns = logits.size(1);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int stride0 = numColumns;
  const int stride1 = 1;
  const auto multipleBlocksPerRowConfig = 10;
  const int splitWorkThreshold = 200 * 1000;

  if (isPrefill) {
    CHECK_CUDA(rowStarts);
    const auto rowStartsPtr = rowStarts.data_ptr<int32_t>();
    constexpr int kNumThreadsPerBlock = 512;
    int numInsertionBlocks = std::min(static_cast<int>(numRows), kSortingAlgorithmThreshold);
    topKPerRow<kNumThreadsPerBlock, false><<<numInsertionBlocks, kNumThreadsPerBlock, 0, stream>>>(
        logitsPtr, rowStartsPtr, rowEndsPtr, outIndicesPtr, nullptr, stride0, stride1);

    if (numRows > kSortingAlgorithmThreshold) {
      int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
      topKPerRow<kNumThreadsPerBlock, true><<<numRadixBlocks, kNumThreadsPerBlock, 0, stream>>>(
          logitsPtr, rowStartsPtr, rowEndsPtr, outIndicesPtr, nullptr, stride0, stride1, kSortingAlgorithmThreshold);
    }
  } else {
    CHECK_CUDA(outIndicesAux);
    CHECK_CUDA(outLogitsAux);
    const auto rowStartsPtr = nullptr;
    const auto outIndicesAuxPtr = outIndicesAux.data_ptr<int32_t>();
    const auto outLogitsAuxPtr = outLogitsAux.data_ptr<float>();
    if (numColumns < kSortingAlgorithmThreshold) {
      constexpr int kNumThreadsPerBlock = 512;
      topKPerRow<kNumThreadsPerBlock, false><<<numRows, kNumThreadsPerBlock, 0, stream>>>(
          logitsPtr, rowStartsPtr, rowEndsPtr, outIndicesPtr, nullptr, stride0, stride1);
    } else if (numColumns < splitWorkThreshold) {
      if (numRows >= 1024) {
        constexpr int kNumThreadsPerBlock = 256;
        topKPerRow<kNumThreadsPerBlock, true><<<numRows, kNumThreadsPerBlock, 0, stream>>>(
            logitsPtr, rowStartsPtr, rowEndsPtr, outIndicesPtr, nullptr, stride0, stride1);
      } else {
        constexpr int kNumThreadsPerBlock = 512;
        topKPerRow<kNumThreadsPerBlock, true><<<numRows, kNumThreadsPerBlock, 0, stream>>>(
            logitsPtr, rowStartsPtr, rowEndsPtr, outIndicesPtr, nullptr, stride0, stride1);
      }
    } else {
      constexpr int kNumThreadsPerBlock = 512;
      topKPerRow<kNumThreadsPerBlock, true, true>
          <<<dim3(numRows, multipleBlocksPerRowConfig), kNumThreadsPerBlock, 0, stream>>>(
              logitsPtr, rowStartsPtr, rowEndsPtr, outIndicesAuxPtr, outLogitsAuxPtr, stride0, stride1);

      constexpr int kNumThreadsPerBlockStep2 = 1024;
      topKPerRow<kNumThreadsPerBlockStep2, true, false, true><<<numRows, kNumThreadsPerBlockStep2, 0, stream>>>(
          outLogitsAuxPtr,
          rowStartsPtr,
          rowEndsPtr,
          outIndicesPtr,
          nullptr,
          multipleBlocksPerRowConfig * kTopK,
          1,
          0,
          multipleBlocksPerRowConfig,
          outIndicesAuxPtr);
    }
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_interface(
    at::Tensor logits,
    at::Tensor rowStarts,
    at::Tensor rowEnds,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    at::Tensor outIndicesAux,
    at::Tensor outLogitsAux,
    bool isPrefill) {
  CHECK_CUDA(logits);
  CHECK_CUDA(dst_page_table);
  CHECK_CUDA(rowEnds);

  constexpr int kTopK = 2048;
  // Threshold of number of elements to use radix / insertion sort.
  constexpr int kSortingAlgorithmThreshold = 12288;
  const auto logitsPtr = logits.data_ptr<float>();

  const auto outIndicesPtr = dst_page_table.data_ptr<int32_t>();

  const auto rowEndsPtr = rowEnds.data_ptr<int32_t>();
  const auto numRows = logits.size(0);
  const auto numColumns = logits.size(1);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int stride0 = numColumns;
  const int stride1 = 1;
  const auto multipleBlocksPerRowConfig = 10;
  const int splitWorkThreshold = 200 * 1000;
  const auto src_page_tablePtr = src_page_table.data_ptr<int32_t>();

  if (isPrefill) {
    CHECK_CUDA(rowStarts);
    CHECK_CUDA(cu_seqlens_q);
    constexpr int kNumThreadsPerBlock = 512;
    const auto src_stride = src_page_table.stride(0);
    const auto rowStartsPtr = rowStarts.data_ptr<int32_t>();
    const auto cu_seqlens_qPtr = cu_seqlens_q.data_ptr<int32_t>();
    const auto prefill_bs = cu_seqlens_q.size(0) - 1;
    int numInsertionBlocks = std::min(static_cast<int>(numRows), kSortingAlgorithmThreshold);
    topKPerRow<kNumThreadsPerBlock, false, false, false, false, true>
        <<<numInsertionBlocks, kNumThreadsPerBlock, 0, stream>>>(
            logitsPtr,
            rowStartsPtr,
            rowEndsPtr,
            outIndicesPtr,
            nullptr,
            stride0,
            stride1,
            0,
            0,
            nullptr,
            nullptr,
            src_page_tablePtr,
            src_stride,
            cu_seqlens_qPtr,
            prefill_bs);

    if (numRows > kSortingAlgorithmThreshold) {
      int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
      topKPerRow<kNumThreadsPerBlock, true, false, false, false, true>
          <<<numRadixBlocks, kNumThreadsPerBlock, 0, stream>>>(
              logitsPtr,
              rowStartsPtr,
              rowEndsPtr,
              outIndicesPtr,
              nullptr,
              stride0,
              stride1,
              kSortingAlgorithmThreshold,
              0,
              nullptr,
              nullptr,
              src_page_tablePtr,
              src_stride,
              cu_seqlens_qPtr,
              prefill_bs);
    }

  } else {
    CHECK_CUDA(outIndicesAux);
    CHECK_CUDA(outLogitsAux);
    const auto outIndicesAuxPtr = outIndicesAux.data_ptr<int32_t>();
    const auto outLogitsAuxPtr = outLogitsAux.data_ptr<float>();
    const auto rowStartsPtr = nullptr;
    if (numColumns < kSortingAlgorithmThreshold) {
      constexpr int kNumThreadsPerBlock = 512;
      topKPerRow<kNumThreadsPerBlock, false, false, false, true, false><<<numRows, kNumThreadsPerBlock, 0, stream>>>(
          logitsPtr,
          rowStartsPtr,
          rowEndsPtr,
          outIndicesPtr,
          nullptr,
          stride0,
          stride1,
          0,
          0,
          nullptr,
          nullptr,
          src_page_tablePtr);
    } else if (numColumns < splitWorkThreshold) {
      if (numRows >= 1024) {
        constexpr int kNumThreadsPerBlock = 256;
        topKPerRow<kNumThreadsPerBlock, true, false, false, true, false><<<numRows, kNumThreadsPerBlock, 0, stream>>>(
            logitsPtr,
            rowStartsPtr,
            rowEndsPtr,
            outIndicesPtr,
            nullptr,
            stride0,
            stride1,
            0,
            0,
            nullptr,
            nullptr,
            src_page_tablePtr);
      } else {
        constexpr int kNumThreadsPerBlock = 512;
        topKPerRow<kNumThreadsPerBlock, true, false, false, true, false><<<numRows, kNumThreadsPerBlock, 0, stream>>>(
            logitsPtr,
            rowStartsPtr,
            rowEndsPtr,
            outIndicesPtr,
            nullptr,
            stride0,
            stride1,
            0,
            0,
            nullptr,
            nullptr,
            src_page_tablePtr);
      }
    } else {
      constexpr int kNumThreadsPerBlock = 512;
      topKPerRow<kNumThreadsPerBlock, true, true, false, false, false>
          <<<dim3(numRows, multipleBlocksPerRowConfig), kNumThreadsPerBlock, 0, stream>>>(
              logitsPtr,
              rowStartsPtr,
              rowEndsPtr,
              outIndicesAuxPtr,
              outLogitsAuxPtr,
              stride0,
              stride1,
              0,
              0,
              nullptr,
              nullptr,
              src_page_tablePtr);

      constexpr int kNumThreadsPerBlockStep2 = 1024;
      topKPerRow<kNumThreadsPerBlockStep2, true, false, true, true, false>
          <<<numRows, kNumThreadsPerBlockStep2, 0, stream>>>(
              outLogitsAuxPtr,
              rowStartsPtr,
              rowEndsPtr,
              outIndicesPtr,
              nullptr,
              multipleBlocksPerRowConfig * kTopK,
              1,
              0,
              multipleBlocksPerRowConfig,
              outIndicesAuxPtr,
              nullptr,
              src_page_tablePtr);
    }
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_ragged_interface(
    at::Tensor score,
    at::Tensor outIndicesAux,
    at::Tensor outLogitsAux,
    at::Tensor topk_indices_ragged,
    at::Tensor rowStarts,
    at::Tensor lengths,
    const at::Tensor& topk_indices_offset,
    bool isPrefill) {
  constexpr int kTopK = 2048;

  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_indices_ragged);
  CHECK_CUDA(topk_indices_offset);
  const auto B = score.size(0);
  TORCH_CHECK(topk_indices_ragged.dim() == 2 && topk_indices_ragged.is_contiguous());
  TORCH_CHECK(topk_indices_offset.dim() == 1);
  TORCH_CHECK(topk_indices_ragged.size(0) == B);
  TORCH_CHECK(topk_indices_ragged.size(1) == kTopK);
  TORCH_CHECK(topk_indices_offset.size(0) == B);
  // Threshold of number of elements to use radix / insertion sort.
  constexpr int kSortingAlgorithmThreshold = 12288;
  const auto logitsPtr = score.data_ptr<float>();
  const auto outIndicesPtr = topk_indices_ragged.data_ptr<int32_t>();

  const auto rowEndsPtr = lengths.data_ptr<int32_t>();
  const auto offsetPtr = topk_indices_offset.data_ptr<int32_t>();
  const auto numRows = score.size(0);
  const auto numColumns = score.size(1);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int stride0 = numColumns;
  const int stride1 = 1;
  const auto multipleBlocksPerRowConfig = 10;
  const int splitWorkThreshold = 200 * 1000;

  if (isPrefill) {
    CHECK_CUDA(rowStarts);
    const auto rowStartsPtr = rowStarts.data_ptr<int32_t>();
    constexpr int kNumThreadsPerBlock = 512;
    int numInsertionBlocks = std::min(static_cast<int>(numRows), kSortingAlgorithmThreshold);
    topKPerRow<kNumThreadsPerBlock, false><<<numInsertionBlocks, kNumThreadsPerBlock, 0, stream>>>(
        logitsPtr, rowStartsPtr, rowEndsPtr, outIndicesPtr, nullptr, stride0, stride1, 0, 0, nullptr, offsetPtr);

    if (numRows > kSortingAlgorithmThreshold) {
      int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
      topKPerRow<kNumThreadsPerBlock, true><<<numRadixBlocks, kNumThreadsPerBlock, 0, stream>>>(
          logitsPtr,
          rowStartsPtr,
          rowEndsPtr,
          outIndicesPtr,
          nullptr,
          stride0,
          stride1,
          kSortingAlgorithmThreshold,
          0,
          nullptr,
          offsetPtr);
    }

  } else {
    CHECK_CUDA(outIndicesAux);
    CHECK_CUDA(outLogitsAux);
    const auto outIndicesAuxPtr = outIndicesAux.data_ptr<int32_t>();
    const auto outLogitsAuxPtr = outLogitsAux.data_ptr<float>();
    const auto rowStartsPtr = nullptr;
    if (numColumns < kSortingAlgorithmThreshold) {
      constexpr int kNumThreadsPerBlock = 512;
      topKPerRow<kNumThreadsPerBlock, false, false, false><<<numRows, kNumThreadsPerBlock, 0, stream>>>(
          logitsPtr, rowStartsPtr, rowEndsPtr, outIndicesPtr, nullptr, stride0, stride1, 0, 0, nullptr, offsetPtr);
    } else if (numColumns < splitWorkThreshold) {
      if (numRows >= 1024) {
        constexpr int kNumThreadsPerBlock = 256;
        topKPerRow<kNumThreadsPerBlock, true, false, false><<<numRows, kNumThreadsPerBlock, 0, stream>>>(
            logitsPtr, rowStartsPtr, rowEndsPtr, outIndicesPtr, nullptr, stride0, stride1, 0, 0, nullptr, offsetPtr);
      } else {
        constexpr int kNumThreadsPerBlock = 512;
        topKPerRow<kNumThreadsPerBlock, true, false, false><<<numRows, kNumThreadsPerBlock, 0, stream>>>(
            logitsPtr, rowStartsPtr, rowEndsPtr, outIndicesPtr, nullptr, stride0, stride1, 0, 0, nullptr, offsetPtr);
      }
    } else {
      constexpr int kNumThreadsPerBlock = 512;
      topKPerRow<kNumThreadsPerBlock, true, true, false>
          <<<dim3(numRows, multipleBlocksPerRowConfig), kNumThreadsPerBlock, 0, stream>>>(
              logitsPtr,
              rowStartsPtr,
              rowEndsPtr,
              outIndicesAuxPtr,
              outLogitsAuxPtr,
              stride0,
              stride1,
              0,
              0,
              nullptr,
              offsetPtr);

      constexpr int kNumThreadsPerBlockStep2 = 1024;
      topKPerRow<kNumThreadsPerBlockStep2, true, false, true><<<numRows, kNumThreadsPerBlockStep2, 0, stream>>>(
          outLogitsAuxPtr,
          rowStartsPtr,
          rowEndsPtr,
          outIndicesPtr,
          nullptr,
          multipleBlocksPerRowConfig * kTopK,
          1,
          0,
          multipleBlocksPerRowConfig,
          outIndicesAuxPtr,
          offsetPtr);
    }
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}
