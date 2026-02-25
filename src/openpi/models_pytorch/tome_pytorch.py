"""Token Merging (ToMe) implementation for PyTorch.

ToMe is a technique to reduce the number of tokens in vision transformers
by merging similar tokens, which can significantly speed up inference
while maintaining performance.
"""

import math
import torch
import torch.nn.functional as F


def bipartite_soft_merge(
    metric: torch.Tensor,
    x: torch.Tensor,
    r: int,
    mode: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    高效版 ToMe bipartite soft matching，使用奇偶分组和 scatter_reduce。
    
    Args:
        metric: [B, T, C] 特征向量（用于计算相似度，通常是 K）
        x:      [B, T, C] 要合并的 token 特征
        r:      要合并的 token 对数（最多 50%）
        mode:   合并模式 ("mean" 或 "sum")
    
    Returns:
        merged_x: 合并后的 token 特征 [B, T-r, C]
        merged_idx: 合并后 token 在合并前的绝对位置索引 [B, T-r]
    """
    B, T, C = x.shape
    r = min(r, T // 2)
    if r <= 0:
        return x, torch.arange(T, device=x.device)[None, :].expand(B, -1)

    # 本层的原始顺序索引
    idx = torch.arange(T, device=x.device)[None, :].expand(B, -1)

    with torch.no_grad():
        # 归一化用于计算相似度
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # 奇偶分组
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        # 计算相似度矩阵
        scores = a @ b.transpose(-1, -2)  # [B, T//2, T//2]
        # 找到每个 A 组 token 的最佳匹配（B 组中的位置）
        node_max, node_idx = scores.max(dim=-1)  # node_max: [B, T//2], node_idx: [B, T//2]
        # 按相似度排序，选择 top-r 对进行合并
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # [B, T//2, 1]
        unm_idx = edge_idx[..., r:, :]   # 未合并的 A 集 [B, T//2-r, 1]
        src_idx = edge_idx[..., :r, :]   # 要合并的 A 集 [B, r, 1]
        # 找到对应的 B 集位置
        # src_idx 是 A 组中的索引（0 到 T//2-1），需要找到对应的 B 组位置
        # node_idx[b, a_idx] 表示 batch b 中 A 组 token a_idx 对应的 B 组位置
        batch_indices = torch.arange(B, device=x.device)[:, None]  # [B, 1]
        src_idx_flat = src_idx.squeeze(-1)  # [B, r] - A 组中的索引
        # 使用 gather 从 node_idx 中获取对应的 B 组位置
        dst_idx = torch.gather(
            node_idx.unsqueeze(-1),  # [B, T//2, 1]
            dim=1,
            index=src_idx  # [B, r, 1] - 在 dim=1 上索引
        )  # [B, r, 1]

    # 同步位置索引
    idx_a, idx_b = idx[..., ::2], idx[..., 1::2]
    unm_pos = idx_a.gather(dim=-1, index=unm_idx.squeeze(-1))  # 未合并 A 集的原位置 [B, T//2-r]
    dst_pos = idx_b  # B 集 token 的原位置（合并后继承）[B, T//2]

    # 特征合并
    src, dst = x[..., ::2, :], x[..., 1::2, :]
    # 提取未合并的 A 集特征
    unm = src.gather(dim=-2, index=unm_idx.expand(B, math.ceil(T / 2) - r, C))  # [B, T//2-r, C]
    # 提取要合并的 A 集特征
    src_to_merge = src.gather(dim=-2, index=src_idx.expand(B, r, C))  # [B, r, C]
    # 使用 scatter_reduce 将 A 集特征合并到 B 集
    dst = dst.scatter_reduce(
        -2,
        dst_idx.expand(B, r, C),
        src_to_merge,
        reduce=mode,
        include_self=True,
    )  # [B, T//2, C]

    # 合并后的索引（顺序与输出 token 一致）
    merged_idx = torch.cat([unm_pos, dst_pos], dim=1)  # [B, T-r]

    # 合并后的特征
    merged_x = torch.cat([unm, dst], dim=1)  # [B, T-r, C]

    return merged_x, merged_idx


def bipartite_soft_matching(
    tokens: torch.Tensor,
    r: int,
    metric: str = "cosine",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bipartite soft matching algorithm for token merging.
    使用高效的奇偶分组和 scatter_reduce 实现。
    
    Args:
        tokens: Input tokens of shape (batch, num_tokens, dim)
        r: Number of tokens to remove (merge)
        metric: Similarity metric to use ("cosine" or "euclidean")
    
    Returns:
        merged_tokens: Merged tokens of shape (batch, num_tokens - r, dim)
        merge_weights: Merge weights for tracking (dummy, for compatibility)
    """
    batch_size, num_tokens, dim = tokens.shape
    
    if r <= 0 or num_tokens <= r:
        return tokens, torch.ones((batch_size, num_tokens, 1), device=tokens.device, dtype=tokens.dtype)
    
    # 使用 tokens 本身作为 metric（用于计算相似度）
    if metric == "cosine":
        # 对于 cosine，使用归一化的 tokens
        metric_tokens = F.normalize(tokens, p=2, dim=-1, eps=1e-8)
    else:
        # 对于 euclidean，直接使用 tokens
        metric_tokens = tokens
    
    # 使用高效的 bipartite_soft_merge
    merged_tokens, _ = bipartite_soft_merge(
        metric=metric_tokens,
        x=tokens,
        r=r,
        mode="mean",
    )
    
    # 返回 dummy weights 以保持接口兼容
    merge_weights = torch.ones((batch_size, num_tokens - r, 1), device=tokens.device, dtype=tokens.dtype)
    
    return merged_tokens, merge_weights


def merge_tokens(
    tokens: torch.Tensor,
    r: int,
    metric: str = "cosine",
) -> torch.Tensor:
    """
    Merge tokens using ToMe algorithm.
    
    Args:
        tokens: Input tokens of shape (batch, num_tokens, dim)
        r: Number of tokens to remove (merge)
        metric: Similarity metric ("cosine" or "euclidean")
    
    Returns:
        Merged tokens of shape (batch, num_tokens - r, dim)
    """
    merged, _ = bipartite_soft_matching(tokens, r, metric)
    return merged


def apply_tome(
    tokens: torch.Tensor,
    ratio: float = 0.5,
    metric: str = "cosine",
    enabled: bool = True,
) -> torch.Tensor:
    """
    Apply Token Merging (ToMe) to reduce token count.
    
    This is the main function to use for ToMe integration.
    It can be easily enabled/disabled via the `enabled` parameter.
    
    Args:
        tokens: Input tokens of shape (batch, num_tokens, dim)
        ratio: Ratio of tokens to keep (0.0 to 1.0). 
               For example, 0.75 means keep 75% of tokens (merge 25%)
        metric: Similarity metric ("cosine" or "euclidean")
        enabled: Whether to apply ToMe. If False, returns original tokens.
    
    Returns:
        Merged tokens (or original if disabled)
    
    Example:
        >>> tokens = torch.randn(1, 100, 768)
        >>> merged = apply_tome(tokens, ratio=0.75, enabled=True)
        >>> # merged.shape = (1, 75, 768)
    """
    if not enabled:
        return tokens
    
    batch_size, num_tokens, dim = tokens.shape
    
    if num_tokens <= 1:
        return tokens
    
    # Calculate number of tokens to remove
    num_keep = int(num_tokens * ratio)
    r = num_tokens - num_keep
    
    if r <= 0:
        return tokens
    
    return merge_tokens(tokens, r, metric)

