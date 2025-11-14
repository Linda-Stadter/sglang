from typing import Any, Optional

import numpy as np
import pytest
import torch
from sgl_kernel import (
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
)


def _ref_torch_impl(
    score: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor],
) -> torch.Tensor:
    assert score.dim() == 2
    batch_size = score.shape[0]
    max_seq_len = seq_lens.max().item()

    new_score = torch.full(
        (batch_size, max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=score.device,
    )

    col_indices = torch.arange(max_seq_len, device=score.device).unsqueeze(0)
    valid_mask = col_indices < seq_lens.unsqueeze(1)

    src_col_indices = col_indices + row_starts.unsqueeze(1)
    src_col_indices = torch.clamp(src_col_indices, 0, score.shape[1] - 1)
    gathered = torch.gather(score, 1, src_col_indices)

    new_score = torch.where(valid_mask, gathered, new_score)

    k = min(topk, max(seq_lens))
    torch_indices = new_score.topk(k, dim=-1)[1]
    result = torch.full(
        (batch_size, topk), -1, dtype=torch_indices.dtype, device=score.device
    )

    mask = torch_indices < seq_lens.unsqueeze(1)

    result[:, :k] = torch_indices.masked_fill(~mask, -1)
    return result


def _ref_torch_transform_decode_impl(
    score: torch.Tensor,
    seq_len: torch.Tensor,
    src_page_table: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, _ = score.shape
    assert score.shape[0] == src_page_table.shape[0]
    indices = _ref_torch_impl(score, seq_len, topk, row_starts=row_starts)

    return indices


def _ref_torch_transform_impl(
    score: torch.Tensor,
    seq_len: torch.Tensor,
    src_page_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, _ = score.shape
    num_sequences = len(cu_seqlens_q) - 1

    indices = _ref_torch_impl(
        score, seq_len, topk, row_starts=row_starts
    )  # + row_starts.unsqueeze(1)

    topk_indices = torch.full(
        (batch_size, topk), -1, dtype=torch.int32, device=score.device
    )

    query_positions = torch.arange(batch_size, device=score.device)
    seq_ids = torch.searchsorted(cu_seqlens_q, query_positions, right=True) - 1

    valid_mask = indices != -1

    seq_ids_expanded = seq_ids.unsqueeze(1).expand(-1, topk)
    indices_clamped = torch.clamp(indices, min=0)

    topk_indices = src_page_table[seq_ids_expanded, indices_clamped]

    topk_indices = torch.where(valid_mask, topk_indices, -1)

    return topk_indices


def _ref_torch_transform_ragged_impl(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor,
) -> torch.Tensor:
    assert score.shape[0] == topk_indices_offset.shape[0]
    indices = _ref_torch_impl(score, lengths, topk, row_starts=row_starts)
    mask = indices != -1
    topk_indices_offset = topk_indices_offset.unsqueeze(1)

    return torch.where(mask, indices + topk_indices_offset, indices)


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_lens: torch.Tensor,
    topk_indices_offset: Optional[torch.Tensor] = None,
    max_permit_error: int = 0,
    row_starts: Optional[torch.Tensor] = None,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()

    wrong_values = 0
    for i in range(bs):
        indices_ref_set_i = set(indices_ref_cpu[i])
        indices_our_set_i = set(indices_our_cpu[i])
        more = indices_our_set_i - indices_ref_set_i
        less = indices_ref_set_i - indices_our_set_i
        offset = topk_indices_offset[i].item() if topk_indices_offset is not None else 0
        row_start = row_starts[i].item() if row_starts is not None else 0
        if len(more) > 0 or len(less) > 0:
            # check whether more values are the same with less values
            # if so, either one is acceptable, since their values are the same
            more_values = sorted(
                score[i, idx - offset + row_start].item() for idx in more if idx != -1
            )
            less_values = sorted(
                score[i, idx - offset + row_start].item() for idx in less if idx != -1
            )
            if more_values != less_values:
                wrong_values += len(more)
                print(
                    f"{bs=}, {k=}, {seq_lens[i].item()=}, {i=}, {len(more)=}, {len(less)=}, failed, with {len(more_values)=}, {len(less_values)=}, {len(more)=}, {len(less)=}, {len(indices_our_cpu[i])=}, {len(indices_ref_cpu[i])=}"
                )

        assert wrong_values <= max_permit_error, f"{wrong_values=}, {max_permit_error=}"


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
    data_generation: str,
) -> torch.Tensor:
    """Create random logits tensor for testing.

    Args:
        row_starts: Starting column index for each row
        row_ends: Length of valid data for each row (not absolute end position)
        dtype: Data type for the tensor
        seed: Random seed
        data_generation: Type of data generation ("random" or "10LSBits")

    Returns:
        Tensor of shape (batch_size, max_columns) where valid data for row i
        is in columns [row_starts[i], row_starts[i] + row_ends[i])
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = row_starts.shape[0]
    # Calculate the maximum column needed: max(row_start + row_end) across all rows
    max_columns = (row_starts + row_ends).max().item()

    # Generate logits with some structure to make testing more meaningful
    if data_generation == "random":
        logits = torch.randn(batch_size, max_columns, dtype=dtype, device="cuda")
    elif data_generation == "10LSBits":
        top_22_bits_mask = 0xFFFFFC00
        last_10_bits_mask = 0x000003FF
        fixed_top_22_bits = 0x3F900000
        # Generate random bits for the last 10 bits
        random_bottom_bits = torch.randint(
            0,
            2**10,
            (batch_size, max_columns),
            dtype=torch.int32,
            device="cuda",
        )
        # Combine: fixed top 22 bits with random last 10 bits
        logits_bits = (fixed_top_22_bits & top_22_bits_mask) | (
            random_bottom_bits & last_10_bits_mask
        )
        logits = logits_bits.view(dtype)

    # For each row, set invalid regions to -inf
    for i in range(batch_size):
        start = row_starts[i].item()
        length = row_ends[i].item()
        end = start + length

        # Set columns before row_start to -inf (should be ignored by kernel)
        if start > 0:
            logits[i, :start] = float("inf")

        # Valid data is in [start, end) - leave it as generated

        # Set columns after row_start + length to -inf
        if end < max_columns:
            logits[i, end:] = float("inf")

    return logits


@pytest.mark.parametrize("bs", [1024, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 8192, 210000])
@pytest.mark.parametrize("data_generation", ["random", "10LSBits"])
@torch.inference_mode()
def test_topk_decode_kernel(
    bs: int, k: int, seq_len: int, data_generation: str
) -> None:
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    lengths = torch.randint(1, seq_len + 1, (bs,), dtype=torch.int32, device="cuda")

    row_starts = torch.zeros(bs, dtype=torch.int32, device="cuda")
    score = create_random_logits(
        row_starts, lengths, torch.float32, 42, data_generation
    )
    indices_ref = _ref_torch_impl(score, lengths, k, row_starts=row_starts)
    indices_our = fast_topk_v2(score, lengths, k, row_starts=None, is_prefill=False)

    # sort and compare
    indices_ref = torch.sort(indices_ref, dim=-1).values
    indices_our = torch.sort(indices_our, dim=-1).values

    assert_equal(score, indices_ref, indices_our, bs, k, lengths)


@pytest.mark.parametrize("bs", [2048, 8192])
@pytest.mark.parametrize("k", [2048])
@pytest.mark.parametrize("num_sequences", [1, 5, 10])
@pytest.mark.parametrize("data_generation", ["random", "10LSBits"])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_topk_prefill_kernel(
    bs: int, k: int, num_sequences: int, has_row_starts: bool, data_generation: str
) -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed(24)
    np.random.seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    # Randomly partition bs into num_sequences parts
    # Tests prefill with pattern of increasing sequence lengths for each sequence
    seq_lens_list = []
    remaining = bs
    for i in range(num_sequences - 1):
        max_val = remaining - (num_sequences - i - 1)
        val = np.random.randint(1, max_val + 1)
        seq_lens_list.append(val)
        remaining -= val
    seq_lens_list.append(remaining)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")

    cu_seqlens_q = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ]
    )

    if has_row_starts:
        row_starts = torch.randint(
            0, bs - seq_lens.max().item() + 1, (bs,), dtype=torch.int32, device="cuda"
        )
    else:
        row_starts = torch.zeros(bs, dtype=torch.int32, device="cuda")
    # Create lengths tensor: [1,2,3,...,max_len] for each sequence
    lengths = torch.cat(
        [
            torch.arange(1, seq_len + 1, dtype=torch.int32, device="cuda")
            for seq_len in seq_lens
        ]
    )
    max_len = lengths.max().item()
    score = create_random_logits(
        row_starts, lengths, torch.float32, 42, data_generation
    )
    indices_ref = _ref_torch_impl(score, lengths, k, row_starts=row_starts)

    indices_our = fast_topk_v2(
        score, lengths, k, row_starts=row_starts, is_prefill=True
    )

    # sort and compare
    indices_ref = torch.sort(indices_ref, dim=-1).values
    indices_our = torch.sort(indices_our, dim=-1).values

    assert_equal(score, indices_ref, indices_our, bs, k, lengths, row_starts=row_starts)


@pytest.mark.parametrize("bs", [1024, 4096])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [2048, 8196, 210000])
@torch.inference_mode()
def test_topk_transform_decode_kernel_2(bs: int, k: int, seq_len: int) -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed(24)
    np.random.seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")

    row_starts = torch.zeros(bs, dtype=torch.int32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

    score = create_random_logits(row_starts, lengths, torch.float32, 42, "random")
    src_page_table = torch.arange(0, seq_len, dtype=torch.int32, device="cuda")
    src_page_table = src_page_table.unsqueeze(0).expand(bs, -1)

    dst_page_table_ref = _ref_torch_transform_decode_impl(
        score=score,
        seq_len=lengths,
        src_page_table=src_page_table,
        topk=k,
        row_starts=row_starts,
    )
    dst_page_table_our = fast_topk_transform_fused(
        score=score,
        lengths=lengths,
        page_table_size_1=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=k,
        row_starts=row_starts,
        is_prefill=False,
    )

    # sort and compare
    dst_page_table_our = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref = torch.sort(dst_page_table_ref, dim=-1).values

    assert_equal(
        score=score,
        indices_ref=dst_page_table_ref,
        indices_our=dst_page_table_our,
        bs=bs,
        k=k,
        seq_lens=lengths,
        row_starts=row_starts,
    )


@pytest.mark.parametrize("bs", [1024, 2048, 8192])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("num_sequences", [1, 8])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_topk_transform_prefill_kernel(
    bs: int, k: int, num_sequences: int, has_row_starts: bool
) -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed(24)
    np.random.seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    # Randomly partition bs into num_sequences parts
    # Tests prefill with pattern of increasing sequence lengths for each sequence
    seq_lens_list = []
    remaining = bs
    for i in range(num_sequences - 1):
        max_val = remaining - (num_sequences - i - 1)
        val = np.random.randint(1, max_val + 1)
        seq_lens_list.append(val)
        remaining -= val
    seq_lens_list.append(remaining)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")

    cu_seqlens_q = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ]
    )

    if has_row_starts:
        row_starts = torch.randint(
            0, bs - seq_lens.max().item() + 1, (bs,), dtype=torch.int32, device="cuda"
        )
    else:
        row_starts = torch.zeros(bs, dtype=torch.int32, device="cuda")
    lengths = torch.cat(
        [
            torch.arange(1, seq_len + 1, dtype=torch.int32, device="cuda")
            for seq_len in seq_lens
        ]
    )

    max_len = lengths.max().item()
    score = create_random_logits(row_starts, lengths, torch.float32, 42, "random")
    src_page_table = (
        torch.arange(max_len, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(num_sequences, -1)
    )

    dst_page_table_ref = _ref_torch_transform_impl(
        score=score,
        seq_len=lengths,
        src_page_table=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=k,
        row_starts=row_starts,
    )
    dst_page_table_our = fast_topk_transform_fused(
        score=score,
        row_starts=row_starts,
        lengths=lengths,
        page_table_size_1=src_page_table,
        cu_seqlens_q=cu_seqlens_q,
        topk=k,
        is_prefill=True,
    )

    # sort and compare
    dst_page_table_our = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref = torch.sort(dst_page_table_ref, dim=-1).values

    assert_equal(
        score=score,
        indices_ref=dst_page_table_ref,
        indices_our=dst_page_table_our,
        bs=bs,
        k=k,
        seq_lens=lengths,
        row_starts=row_starts,
    )


@pytest.mark.parametrize("bs", [2048])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("seq_len", [1000, 8192, 210000])
@torch.inference_mode()
def test_topk_decode_transform_ragged_kernel(bs: int, k: int, seq_len: int) -> None:
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    row_starts = torch.zeros(bs, dtype=torch.int32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    score = create_random_logits(row_starts, lengths, torch.float32, 42, "random")

    topk_indices_offset = torch.randint(
        0, 1024, (bs,), dtype=torch.int32, device="cuda"
    )

    dst_page_table_ref = _ref_torch_transform_ragged_impl(
        score=score,
        lengths=lengths,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        row_starts=row_starts,
    )
    dst_page_table_our = fast_topk_transform_ragged_fused(
        score=score,
        lengths=lengths,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        is_prefill=False,
        row_starts=row_starts,
    )

    # sort and compare
    dst_page_table_our = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref = torch.sort(dst_page_table_ref, dim=-1).values

    assert_equal(
        score,
        dst_page_table_ref,
        dst_page_table_our,
        bs,
        k,
        lengths,
        topk_indices_offset,
    )


@pytest.mark.parametrize("bs", [2048, 8192])
@pytest.mark.parametrize("k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("num_sequences", [1, 5, 10])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_topk_prefill_transform_ragged_kernel(
    bs: int, k: int, num_sequences: int, has_row_starts: bool
) -> None:
    torch.manual_seed(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    # Randomly partition bs into num_sequences parts
    # Tests prefill with pattern of increasing sequence lengths for each sequence
    seq_lens_list = []
    remaining = bs
    for i in range(num_sequences - 1):
        max_val = remaining - (num_sequences - i - 1)
        val = np.random.randint(1, max_val + 1)
        seq_lens_list.append(val)
        remaining -= val
    seq_lens_list.append(remaining)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")

    if has_row_starts:
        row_starts = torch.randint(
            0, bs - seq_lens.max().item() + 1, (bs,), dtype=torch.int32, device="cuda"
        )
    else:
        row_starts = torch.zeros(bs, dtype=torch.int32, device="cuda")
    # Create lengths tensor: [1,2,3,...,max_len] for each sequence
    lengths = torch.cat(
        [
            torch.arange(1, seq_len + 1, dtype=torch.int32, device="cuda")
            for seq_len in seq_lens
        ]
    )

    score = create_random_logits(row_starts, lengths, torch.float32, 42, "random")

    topk_indices_offset = torch.randint(
        0, 1024, (bs,), dtype=torch.int32, device="cuda"
    )

    dst_page_table_ref = _ref_torch_transform_ragged_impl(
        score=score,
        lengths=lengths,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        row_starts=row_starts,
    )
    dst_page_table_our = fast_topk_transform_ragged_fused(
        score=score,
        row_starts=row_starts,
        lengths=lengths,
        topk_indices_offset=topk_indices_offset,
        topk=k,
        is_prefill=True,
    )

    dst_page_table_our = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref = torch.sort(dst_page_table_ref, dim=-1).values

    assert_equal(
        score,
        dst_page_table_ref,
        dst_page_table_our,
        bs,
        k,
        lengths,
        topk_indices_offset,
        row_starts=row_starts,
    )


if __name__ == "__main__":
    pytest.main([__file__])
