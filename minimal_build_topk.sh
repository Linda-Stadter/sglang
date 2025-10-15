#!/bin/bash
# Minimal build script - ONLY topk.cu + minimal bindings
# Bypasses CMake completely for fast iteration

set -e

echo "========================================================================"
echo "Minimal Build: topk.cu kernels ONLY"
echo "========================================================================"
echo ""

# Paths
TORCH_PATH=$(python -c "import torch; print(torch.__path__[0])")
TORCH_INCLUDE="${TORCH_PATH}/include"
TORCH_LIB="${TORCH_PATH}/lib"
PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

# Detect PyTorch ABI
TORCH_CXX11_ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))" 2>/dev/null || echo "1")
echo "Detected PyTorch CXX11 ABI: $TORCH_CXX11_ABI"

SRC_DIR="/sglang/sgl-kernel"
BUILD_DIR="/tmp/topk_minimal_build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "Configuration:"
echo "  Source:      $SRC_DIR"
echo "  Build:       $BUILD_DIR"
echo "  GPU Arch:    sm_$GPU_ARCH"
echo "  PyTorch:     $TORCH_PATH"
echo "  Python Inc:  $PYTHON_INCLUDE"
echo ""

# Step 1: Compile topk.cu
echo "Step 1/4: Compiling topk.cu..."
${CUDA_HOME}/bin/nvcc \
  -c ${SRC_DIR}/csrc/elementwise/topk.cu \
  -o "${BUILD_DIR}/topk.o" \
  -I${SRC_DIR}/include \
  -I"${TORCH_INCLUDE}" \
  -I"${TORCH_INCLUDE}/torch/csrc/api/include" \
  -I"${CUDA_HOME}/include" \
  -std=c++17 -O3 \
  --expt-relaxed-constexpr \
  --expt-extended-lambda \
  --use_fast_math \
  -DUSE_FAST_MATH=1 \
  -gencode=arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH} \
  -Xcompiler -fPIC \
  -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}

echo "✓ topk.o compiled ($(du -h ${BUILD_DIR}/topk.o | cut -f1))"
echo ""

# Step 2: Create minimal Python binding
echo "Step 2/4: Creating minimal Python bindings..."
cat > ${BUILD_DIR}/topk_bindings.cc << 'EOFCPP'
#include <torch/extension.h>
#include "sgl_kernel_ops.h"

PYBIND11_MODULE(topk_ops, m) {
  m.doc() = "Minimal topk operations module";
  m.def("fast_topk", &fast_topk_interface, "Fast TopK interface");
  m.def("vllm_fast_topk", &vllm_fast_topk_interface, "VLLM Fast TopK interface");
  m.def("vllm_fast_topk_optimized", &vllm_fast_topk_optimized_interface, "VLLM Fast TopK Optimized interface");
  m.def("fast_topk_transform_fused", &fast_topk_transform_interface, "Fast TopK Transform interface");
}
EOFCPP

echo "✓ Bindings created"
echo ""

# Step 3: Compile bindings
echo "Step 3/4: Compiling bindings..."
g++ \
  -c ${BUILD_DIR}/topk_bindings.cc \
  -o "${BUILD_DIR}/topk_bindings.o" \
  -I${SRC_DIR}/include \
  -I"${TORCH_INCLUDE}" \
  -I"${TORCH_INCLUDE}/torch/csrc/api/include" \
  -I"${PYTHON_INCLUDE}" \
  -I"${CUDA_HOME}/include" \
  -std=c++17 -O3 -fPIC \
  -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}

echo "✓ Bindings compiled"
echo ""

# Step 4: Link into shared library
echo "Step 4/4: Linking shared library..."

g++ -shared \
  "${BUILD_DIR}/topk.o" \
  "${BUILD_DIR}/topk_bindings.o" \
  -o "${BUILD_DIR}/topk_ops.so" \
  -fPIC \
  -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI} \
  -L"${TORCH_LIB}" \
  -L"${CUDA_HOME}/lib64" \
  -Wl,--no-as-needed \
  ${TORCH_LIB}/libtorch_python.so \
  ${TORCH_LIB}/libtorch.so \
  ${TORCH_LIB}/libc10.so \
  ${TORCH_LIB}/libc10_cuda.so \
  -Wl,--as-needed \
  -lcudart \
  -Wl,-rpath,${TORCH_LIB} \
  -Wl,-rpath,${CUDA_HOME}/lib64

echo "✓ Shared library created ($(du -h ${BUILD_DIR}/topk_ops.so | cut -f1))"
echo ""


echo "Build Complete!"
