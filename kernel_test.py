import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/tmp/topk_minimal_build")
import torch
from topk_ops import fast_topk, vllm_fast_topk, vllm_fast_topk_optimized

MAX_SEQ_LEN = 131072
MAX_PERMIT_ERROR = 0
RANDOM_LENGTHS = False


# SGLang kernel
def fast_topk_v2(score, lengths, k):
    indices = torch.empty((score.size(0), k), dtype=torch.int32, device="cuda")
    fast_topk(score, indices, lengths)
    return indices


# VLLM kernel
def vllm_fast_topk_v2(score, lengths, k):
    indices = torch.empty((score.size(0), k), dtype=torch.int32, device="cuda")
    vllm_fast_topk(score, indices, lengths)
    return indices


# VLLM optimized kernel
def vllm_fast_topk_optimized_v2(score, lengths, k):
    indices = torch.empty((score.size(0), k), dtype=torch.int32, device="cuda")
    vllm_fast_topk_optimized(score, indices, lengths)
    return indices


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()
    for i in range(bs):
        indices_ref_set_i = set(indices_ref_cpu[i])
        indices_our_set_i = set(indices_our_cpu[i])
        more = indices_our_set_i - indices_ref_set_i
        less = indices_ref_set_i - indices_our_set_i
        if len(more) > MAX_PERMIT_ERROR or len(less) > MAX_PERMIT_ERROR:

            # check whether more values are the same with less values
            # if so, either one is acceptable, since their values are the same
            more_values = sorted(score[i, idx].item() for idx in more)
            less_values = sorted(score[i, idx].item() for idx in less)
            assert (
                more_values == less_values
            ), f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, with {more_values=}, {less_values=}"


def _ref_torch_impl(score: torch.Tensor, seq_len: int, topk: int) -> torch.Tensor:
    assert score.dim() == 2
    return torch.topk(score, topk, dim=-1, sorted=False).indices


def _ref_torch_impl_with_lengths(
    score: torch.Tensor, lengths: torch.Tensor, topk: int
) -> torch.Tensor:
    """Reference implementation that respects per-batch lengths using vectorized masking."""
    assert score.dim() == 2

    # Run topk on the full tensor (score already has -inf for invalid positions)
    max_length = lengths.max().item()
    k_actual = min(topk, max_length)
    torch_values, torch_indices = score.topk(k_actual, dim=-1, sorted=False)

    # Convert to int32 to match kernel output
    torch_indices = torch_indices.to(torch.int32)

    # Create mask: indices must be < valid length for each batch
    mask = torch_indices < lengths.unsqueeze(1)

    # Set invalid indices to -1
    torch_indices = torch_indices.masked_fill(~mask, -1)

    # If k_actual < topk, pad with -1
    if k_actual < topk:
        bs = score.size(0)
        padded_indices = torch.full(
            (bs, topk), -1, dtype=torch.int32, device=score.device
        )
        padded_indices[:, :k_actual] = torch_indices
        torch_indices = padded_indices

    return torch_indices


def benchmark_kernel(
    kernel_fn, score, lengths, k, num_warmup=10, num_iters=100, kernel_name="kernel"
):
    """Benchmark a kernel using CUDA events for accurate timing."""
    # Warmup
    for _ in range(num_warmup):
        _ = kernel_fn(score, lengths, k)
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        _ = kernel_fn(score, lengths, k)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return {
        "mean": np.mean(times),
        "median": np.median(times),
        "min": np.min(times),
        "max": np.max(times),
        "std": np.std(times),
        "times": times,
    }


def get_data(
    bs: int, k: int, seq_len: int, random_lengths: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    if random_lengths:
        lengths = torch.randint(1, seq_len + 1, (bs,), dtype=torch.int32, device="cuda")
    else:
        lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    mask = torch.arange(MAX_SEQ_LEN, device="cuda").unsqueeze(0) >= lengths.unsqueeze(1)
    score[mask] = float("-inf")
    return score, lengths


# Returns float values that differ in the last 22 bits
def get__22bit_data(
    bs: int, k: int, seq_len: int, random_lengths: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)

    # Generate random floats
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")

    score_bits = score.view(torch.int32)

    top_11_bits_mask = 0xFFE00000
    bottom_21_bits_mask = 0x001FFFFF

    fixed_top_11_bits = 0x3F900000

    # Generate random bits for the bottom 21 bits
    random_bottom_bits = torch.randint(
        0, 2**21, (bs, MAX_SEQ_LEN), dtype=torch.int32, device="cuda"
    )

    # Combine: fixed top 11 bits with random bottom 21 bits
    score_bits = (fixed_top_11_bits & top_11_bits_mask) | (
        random_bottom_bits & bottom_21_bits_mask
    )

    # Convert back to float
    score = score_bits.view(torch.float32)

    if random_lengths:
        lengths = torch.randint(1, seq_len + 1, (bs,), dtype=torch.int32, device="cuda")
    else:
        lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    mask = torch.arange(MAX_SEQ_LEN, device="cuda").unsqueeze(0) >= lengths.unsqueeze(1)
    score[mask] = float("-inf")
    return score, lengths


# Returns float values that differ in the last 10 bits
def get_10bit_data(
    bs: int, k: int, seq_len: int, random_lengths: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)

    top_22_bits_mask = 0xFFFFFC00
    last_10_bits_mask = 0x000003FF
    fixed_top_22_bits = 0x3F900000

    # Generate random bits for the last 10 bits
    random_bottom_bits = torch.randint(
        0, 2**10, (bs, MAX_SEQ_LEN), dtype=torch.int32, device="cuda"
    )

    # Combine: fixed top 22 bits with random last 10 bits
    score_bits = (fixed_top_22_bits & top_22_bits_mask) | (
        random_bottom_bits & last_10_bits_mask
    )

    # Convert back to float
    score = score_bits.view(torch.float32)

    if random_lengths:
        lengths = torch.randint(1, seq_len + 1, (bs,), dtype=torch.int32, device="cuda")
    else:
        lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    mask = torch.arange(MAX_SEQ_LEN, device="cuda").unsqueeze(0) >= lengths.unsqueeze(1)
    score[mask] = float("-inf")
    return score, lengths


def get_tiered_data(
    bs: int, k: int, seq_len: int, random_lengths: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates float data with three tiers to test all rounds of the topk kernel:
    - 10 values: differ in first 11 bits (testing round 1)
    - Next 600 values: same first 11 bits, differ in next 11 bits (testing round 2)
    - Remaining values: same first 22 bits, differ in last 10 bits (testing round 3)
    """
    torch.manual_seed(42)

    score = torch.zeros(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")

    # IEEE 754 float32 format:
    # Bit 31: Sign (1 bit)
    # Bits 30-23: Exponent (8 bits)
    # Bits 22-0: Fraction (23 bits)

    for batch_idx in range(bs):
        # Tier 1: 10 values - differ in first 11 bits
        for i in range(10):
            exponent = 130 - i
            first_2_frac_bits = (9 - i) % 4

            bits_pattern = (exponent << 23) | (first_2_frac_bits << 21) | 0x1FFFFF
            score[batch_idx, i] = torch.tensor(bits_pattern, dtype=torch.int32).view(
                torch.float32
            )

        # Tier 2: Next 600 values - same first 11 bits, differ in next 11 bits (bits 20-10)
        fixed_first_11_bits = 0x3F900000  # 0 | 01111111 | 01 | ...
        for i in range(600):
            next_11_bits = (2047 - i) & 0x7FF

            # Build the bit pattern: fixed first 11 bits | varying next 11 bits | random last 10 bits
            bits_pattern = (
                (fixed_first_11_bits & 0xFFE00000) | (next_11_bits << 10) | (i & 0x3FF)
            )
            score[batch_idx, 10 + i] = torch.tensor(
                bits_pattern, dtype=torch.int32
            ).view(torch.float32)

        # Tier 3: Remaining values - same first 22 bits, differ in last 10 bits
        fixed_first_22_bits = 0x3F900000  # Same as tier 2
        num_remaining = min(MAX_SEQ_LEN - 610, seq_len - 610)
        if num_remaining > 0:
            for i in range(num_remaining):
                # Vary only the last 10 bits
                last_10_bits = i & 0x3FF

                # Build the bit pattern: fixed first 22 bits | varying last 10 bits
                bits_pattern = (fixed_first_22_bits & 0xFFFFFC00) | last_10_bits
                score[batch_idx, 610 + i] = torch.tensor(
                    bits_pattern, dtype=torch.int32
                ).view(torch.float32)

    if random_lengths:
        lengths = torch.randint(1, seq_len + 1, (bs,), dtype=torch.int32, device="cuda")
    else:
        lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

    mask = torch.arange(MAX_SEQ_LEN, device="cuda").unsqueeze(0) >= lengths.unsqueeze(1)
    score[mask] = float("-inf")

    return score, lengths


def test_topk_kernel(
    bs: int,
    k: int,
    seq_len: int,
    run_benchmark: bool = False,
    random_lengths: bool = False,
) -> None:
    torch.manual_seed(42)

    # Print more values instead of the default 6
    torch.set_printoptions(edgeitems=5, linewidth=200)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    # score, lengths = get_data(bs, k, seq_len, random_lengths=random_lengths)
    # score, lengths = get__22bit_data(bs, k, seq_len, random_lengths=random_lengths)
    # score, lengths = get_10bit_data(bs, k, seq_len, random_lengths=random_lengths)
    score, lengths = get_tiered_data(bs, k, seq_len, random_lengths=random_lengths)

    print(f"\n{'='*80}")
    print(f"Testing Top-K Kernels")
    print(f"{'='*80}")
    print(f"Configuration: bs={bs}, k={k}, seq_len={seq_len}")
    print(f"{'='*80}\n")

    # Correctness check
    if random_lengths:
        indices_ref = _ref_torch_impl_with_lengths(score, lengths, k)
    else:
        indices_ref = _ref_torch_impl(score[:, :seq_len], seq_len, k)
    torch.cuda.synchronize()

    indices_fast_topk_v2 = fast_topk_v2(score, lengths, k)
    torch.cuda.synchronize()

    indices_vllm_fast_topk_v2 = vllm_fast_topk_v2(score, lengths, k)
    torch.cuda.synchronize()

    indices_vllm_fast_optimized = vllm_fast_topk_optimized_v2(score, lengths, k)
    torch.cuda.synchronize()

    # Check fast_topk_v2
    indices_ref_sorted = torch.sort(indices_ref, dim=-1).values
    indices_fast_sorted = torch.sort(indices_fast_topk_v2, dim=-1).values
    indices_vllm_fast_topk_optimized_sorted = torch.sort(
        indices_vllm_fast_optimized, dim=-1
    ).values

    try:
        assert_equal(score, indices_ref_sorted, indices_fast_sorted, bs, k, seq_len)
        print("✓ indices_fast_sorted: PASSED")
    except AssertionError as e:
        print(f"✗ indices_fast_sorted: FAILED")

    # Check vllm_fast_topk_v2
    indices_vllm_sorted = torch.sort(indices_vllm_fast_topk_v2, dim=-1).values

    try:
        assert_equal(score, indices_ref_sorted, indices_vllm_sorted, bs, k, seq_len)
        print("✓ vllm_fast_topk_v2: PASSED")
    except AssertionError as e:
        print(f"✗ vllm_fast_topk_v2: FAILED")

    try:
        assert_equal(
            score,
            indices_ref_sorted,
            indices_vllm_fast_topk_optimized_sorted,
            bs,
            k,
            seq_len,
        )
        print("✓ vllm_fast_optimized_topk_v2: PASSED")
    except AssertionError as e:
        print(f"✗ vllm_fast_optimized_topk_v2: FAILED {e}")

    if run_benchmark:
        print(f"\n{'='*80}")
        print("Running Performance Benchmark...")
        print(f"{'='*80}\n")

        # Enable profiler API range for nsys
        torch.cuda.cudart().cudaProfilerStart()

        # Benchmark fast_topk_v2
        print("Benchmarking fast_topk_v2...")
        stats_fast = benchmark_kernel(
            fast_topk_v2,
            score,
            lengths,
            k,
            num_warmup=10,
            num_iters=100,
            kernel_name="fast_topk_v2",
        )

        # Benchmark vllm_fast_topk_v2
        print("Benchmarking vllm_fast_topk_v2...")
        stats_vllm = benchmark_kernel(
            vllm_fast_topk_optimized_v2,
            score,
            lengths,
            k,
            num_warmup=10,
            num_iters=100,
            kernel_name="vllm_fast_topk_v2",
        )

        # Benchmark torch reference
        print("Benchmarking torch.topk (reference)...")

        def torch_topk_wrapper(score, lengths, k):
            return _ref_torch_impl(score, seq_len, k)

        score_tensor = score[:, :seq_len]
        stats_torch = benchmark_kernel(
            torch_topk_wrapper,
            score_tensor,
            lengths,
            k,
            num_warmup=10,
            num_iters=100,
            kernel_name="torch_topk",
        )

        # Stop profiler
        torch.cuda.cudart().cudaProfilerStop()

        print(f"\n{'='*80}")
        print("Performance Results (times in milliseconds)")
        print(f"{'='*80}\n")

        print(
            f"{'Kernel':<30} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'Std':>10}"
        )
        print(f"{'-'*80}")
        print(
            f"{'torch.topk (reference)':<30} {stats_torch['mean']:>10.4f} {stats_torch['median']:>10.4f} {stats_torch['min']:>10.4f} {stats_torch['max']:>10.4f} {stats_torch['std']:>10.4f}"
        )
        print(
            f"{'fast_topk_v2':<30} {stats_fast['mean']:>10.4f} {stats_fast['median']:>10.4f} {stats_fast['min']:>10.4f} {stats_fast['max']:>10.4f} {stats_fast['std']:>10.4f}"
        )
        print(
            f"{'vllm_fast_topk_v2':<30} {stats_vllm['mean']:>10.4f} {stats_vllm['median']:>10.4f} {stats_vllm['min']:>10.4f} {stats_vllm['max']:>10.4f} {stats_vllm['std']:>10.4f}"
        )

        print(f"\n{'='*80}")
        print("Speedup vs torch.topk")
        print(f"{'='*80}\n")
        speedup_fast = stats_torch["mean"] / stats_fast["mean"]
        speedup_vllm = stats_torch["mean"] / stats_vllm["mean"]
        print(f"fast_topk_v2:      {speedup_fast:>6.2f}x")
        print(f"vllm_fast_topk_v2: {speedup_vllm:>6.2f}x")

        print(f"\n{'='*80}")
        print("Comparison: vllm_fast_topk_v2 vs fast_topk_v2")
        print(f"{'='*80}\n")
        ratio = stats_vllm["mean"] / stats_fast["mean"]
        if ratio < 1.0:
            print(f"vllm_fast_topk_v2 is {1.0/ratio:.2f}x FASTER than fast_topk_v2")
        else:
            print(f"vllm_fast_topk_v2 is {ratio:.2f}x SLOWER than fast_topk_v2")
        print()


def run_for_ncu(bs: int, k: int, seq_len: int, random_lengths: bool = False) -> None:
    """Simple run for Nsight Compute profiling."""
    torch.manual_seed(42)
    torch.set_printoptions(edgeitems=5, linewidth=200)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    score, lengths = get_data(bs, k, seq_len, random_lengths=random_lengths)
    # score, lengths = get_tiered_data(bs, k, seq_len, random_lengths=random_lengths)

    print(f"\nRunning kernels for Nsight Compute profiling...")
    print(f"Configuration: bs={bs}, k={k}, seq_len={seq_len}\n")

    # Run kernels (ncu will profile these)
    print("Running fast_topk_v2...")
    _ = fast_topk_v2(score, lengths, k)
    torch.cuda.synchronize()

    print("Running vllm_fast_topk_v2...")
    _ = vllm_fast_topk_v2(score, lengths, k)
    torch.cuda.synchronize()

    print("Running vllm_fast_topk_optimized_v2...")
    _ = vllm_fast_topk_optimized_v2(score, lengths, k)
    torch.cuda.synchronize()

    print("✓ Complete.\n")


if __name__ == "__main__":
    import sys

    # Check flags
    run_benchmark = "--benchmark" in sys.argv or "-b" in sys.argv
    run_profile = "--profile" in sys.argv or "-p" in sys.argv
    run_ncu = "--ncu" in sys.argv or "--compute" in sys.argv
    random_lengths = "--random_lengths" in sys.argv or "--random-lengths" in sys.argv

    # Example: python kernel_test.py --benchmark --bs 4096 --k 2048 --seq_len 65536 --random_lengths
    bs = 4
    # bs = 4096
    k = 2048
    seq_len = 65536
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--bs" and i + 1 < len(sys.argv) - 1:
            bs = int(sys.argv[i + 2])
        elif arg == "--k" and i + 1 < len(sys.argv) - 1:
            k = int(sys.argv[i + 2])
        elif arg == "--seq_len" and i + 1 < len(sys.argv) - 1:
            seq_len = int(sys.argv[i + 2])

    if run_ncu:
        run_for_ncu(bs, k, seq_len, random_lengths=random_lengths)
    else:
        test_topk_kernel(
            bs, k, seq_len, run_benchmark=run_benchmark, random_lengths=random_lengths
        )
