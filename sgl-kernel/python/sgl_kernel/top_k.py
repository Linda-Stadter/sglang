from typing import Optional

import torch


def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        # TODO: implement faster cuda kernels for large vocab sizes
        return torch.topk(values, topk, dim=dim)


def fast_topk_v2(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    is_prefill: bool = False,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor.
    Args:
        score: The score tensor of shape (B, L). The score tensor is the logits
            between the query and the key whose layout is either ragged or paged.
            row_starts is only required when the key is ragged.
        lengths: The lengths tensor of shape (B)
        topk: The number of topk indices to get
        row_starts: The start index of each row in the score tensor of shape (B).
            For each row i, topk only applies to section [row_starts[i], row_starts[i] + lengths[i]]
            of the score tensor.
        is_prefill: Whether the input is from prefill or not.
    Returns:
        The topk indices tensor of shape (B, topk)
    """
    assert (
        topk == 2048
    ), "fast_topk_v2 is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2

    topk_indices = score.new_empty((score.size(0), topk), dtype=torch.int32)

    if is_prefill:
        if row_starts is None:
            row_starts = torch.zeros(
                (score.size(0),), dtype=torch.int32, device=score.device
            )
        row_ends = row_starts + lengths
        torch.ops.sgl_kernel.fast_topk(
            score, topk_indices, row_starts, row_ends, None, None, is_prefill
        )
    else:
        # Auxiliary tensors for long sequences (used when vocab_size >= 200,000)
        splitWorkThreshold = 200 * 1000
        aux_indices = torch.empty((0,), dtype=torch.int32, device=score.device)
        aux_logits = torch.empty((0,), dtype=torch.float32, device=score.device)
        if score.shape[1] >= splitWorkThreshold:
            aux_indices = torch.empty(
                (score.size(0), 10 * 2048), dtype=torch.int32, device=score.device
            )
            aux_logits = torch.empty(
                (score.size(0), 10 * 2048), dtype=torch.float32, device=score.device
            )
        torch.ops.sgl_kernel.fast_topk(
            score,
            topk_indices,
            row_starts,
            lengths,
            aux_indices,
            aux_logits,
            is_prefill,
        )

    return topk_indices


def fast_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,  # NOTE: page size should be 1
    cu_seqlens_q: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    is_prefill: bool = False,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor and then transform the topk indices
    to indices to the page table (page_size = 1)
    Args:
        score: The score tensor of shape (B, L). The score tensor is the logits
            between the query and the key whose layout is either ragged or paged.
            row_starts is only required when the key is ragged.
        lengths: The lengths tensor of shape (B)
        page_table_size_1: The page table tensor of shape (Batch, topk)
        cu_seqlens_q: The cumulative sequence lengths tensor of shape (Batch + 1)
        topk: The number of topk indices to get
        row_starts: The start index of each row in the score tensor of shape (B).
            For each row i, topk only applies to section [row_starts[i], row_starts[i] + lengths[i]]
            of the score tensor. It's only used for cases where the key is
            ragged, i.e. during extend and draft extend.
        is_prefill: Whether the input is from prefill or not.
    Returns:
        The topk indices tensor of shape (B, topk)
    """
    assert (
        topk == 2048
    ), "fast_topk_transform_fused is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    src_page_table = page_table_size_1
    dst_page_table = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    if is_prefill:
        if row_starts is None:
            row_starts = torch.zeros(
                (score.size(0),), dtype=torch.int32, device=score.device
            )
        row_ends = row_starts + lengths
        torch.ops.sgl_kernel.fast_topk_transform_fused(
            score,
            row_starts,
            row_ends,
            dst_page_table,
            src_page_table,
            cu_seqlens_q,
            None,
            None,
            is_prefill,
        )
    else:
        # Auxiliary tensors for long sequences (used when vocab_size >= 200,000)
        splitWorkThreshold = 200 * 1000
        aux_indices = torch.empty((0,), dtype=torch.int32, device=score.device)
        aux_logits = torch.empty((0,), dtype=torch.float32, device=score.device)
        if score.shape[1] >= splitWorkThreshold:
            aux_indices = torch.empty(
                (score.size(0), 10 * 2048), dtype=torch.int32, device=score.device
            )
            aux_logits = torch.empty(
                (score.size(0), 10 * 2048), dtype=torch.float32, device=score.device
            )
        torch.ops.sgl_kernel.fast_topk_transform_fused(
            score,
            row_starts,
            lengths,
            dst_page_table,
            src_page_table,
            cu_seqlens_q,
            aux_indices,
            aux_logits,
            is_prefill,
        )
    return dst_page_table


def fast_topk_transform_ragged_fused(
    score: torch.Tensor,
    row_starts: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices_offset: torch.Tensor,  # ragged kv
    topk: int,
    is_prefill: bool = False,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor and then transform the topk indices to
    indices to ragged kv (non-paged). This function is only used for extend,
    not including draft extend.
    Args:
        score: The score tensor of shape (B, L). The score tensor is the logits
            between the query and the key which can be ragged or paged.
            row_starts is only required when the key is ragged.
        lengths: The lengths tensor of shape (B)
        topk_indices_offset: The offset of topk indices in ragged kv of shape (B)
        topk: The number of topk indices to get
        row_starts: The start index of each row in the score tensor of shape (B).
            For each row i, topk only applies to section [row_starts[i], row_starts[i] + lengths[i]]
            of the score tensor. It can be None if only the fast path is triggered,
            in the case of all values in lengths <= topk (not checked in the kernel,
            guaranteed by the caller).
        is_prefill: Whether the input is from prefill or not.
    Returns:
        The topk indices tensor of shape (B, topk)
    """
    assert (
        topk == 2048
    ), "fast_topk_transform_ragged_fused is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    topk_indices_ragged = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    if is_prefill:
        if row_starts is None:
            row_starts = torch.zeros(
                (score.size(0),), dtype=torch.int32, device=score.device
            )
        row_ends = row_starts + lengths
        torch.ops.sgl_kernel.fast_topk_transform_ragged_fused(
            score,
            None,
            None,
            topk_indices_ragged,
            row_starts,
            row_ends,
            topk_indices_offset,
            is_prefill,
        )
    else:
        # Auxiliary tensors for long sequences (used when vocab_size >= 200,000)
        splitWorkThreshold = 200 * 1000
        aux_indices = torch.empty((0,), dtype=torch.int32, device=score.device)
        aux_logits = torch.empty((0,), dtype=torch.float32, device=score.device)
        if score.shape[1] >= splitWorkThreshold:
            aux_indices = torch.empty(
                (score.size(0), 10 * 2048), dtype=torch.int32, device=score.device
            )
            aux_logits = torch.empty(
                (score.size(0), 10 * 2048), dtype=torch.float32, device=score.device
            )

        torch.ops.sgl_kernel.fast_topk_transform_ragged_fused(
            score,
            aux_indices,
            aux_logits,
            topk_indices_ragged,
            row_starts,
            lengths,
            topk_indices_offset,
            is_prefill,
        )
    return topk_indices_ragged
