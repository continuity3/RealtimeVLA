"""Token Filtering/Fusion (ToFu) implementation for PyTorch.

ToFu is a technique to reduce the number of tokens in vision transformers
by filtering low-importance tokens and fusing similar tokens.
It can be used together with ToMe for additional speedup.
"""

import math
import torch
import torch.nn.functional as F


def compute_token_importance(
    tokens: torch.Tensor,
    method: str = "norm",
) -> torch.Tensor:
    """
    Compute importance scores for each token.
    
    Args:
        tokens: Input tokens of shape (batch, num_tokens, dim)
        method: Method to compute importance ("norm", "variance", "attention")
    
    Returns:
        importance_scores: Importance scores of shape (batch, num_tokens)
    """
    if method == "norm":
        # Use L2 norm as importance measure
        importance = torch.norm(tokens, dim=-1)  # [B, T]
    elif method == "variance":
        # Use variance across feature dimensions as importance
        importance = torch.var(tokens, dim=-1)  # [B, T]
    elif method == "attention":
        # Use self-attention-like importance (simplified)
        # Compute dot product with mean token
        mean_token = tokens.mean(dim=1, keepdim=True)  # [B, 1, C]
        importance = (tokens * mean_token).sum(dim=-1)  # [B, T]
    else:
        raise ValueError(f"Unknown importance method: {method}")
    
    return importance


def filter_tokens(
    tokens: torch.Tensor,
    importance_scores: torch.Tensor,
    ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Filter tokens based on importance scores.
    
    Args:
        tokens: Input tokens of shape (batch, num_tokens, dim)
        importance_scores: Importance scores of shape (batch, num_tokens)
        ratio: Ratio of tokens to keep (0.0 to 1.0)
    
    Returns:
        filtered_tokens: Filtered tokens of shape (batch, num_keep, dim)
        keep_indices: Indices of kept tokens of shape (batch, num_keep)
    """
    B, T, C = tokens.shape
    num_keep = max(1, int(T * ratio))
    
    if num_keep >= T:
        return tokens, torch.arange(T, device=tokens.device)[None, :].expand(B, -1)
    
    # Select top-k tokens by importance
    _, top_indices = torch.topk(importance_scores, k=num_keep, dim=-1)  # [B, num_keep]
    
    # Gather filtered tokens
    batch_indices = torch.arange(B, device=tokens.device)[:, None]  # [B, 1]
    filtered_tokens = tokens[batch_indices, top_indices]  # [B, num_keep, C]
    
    return filtered_tokens, top_indices


def fuse_similar_tokens(
    tokens: torch.Tensor,
    metric: torch.Tensor,
    r: int,
    mode: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fuse similar tokens using bipartite matching (similar to ToMe).
    
    Args:
        tokens: Input tokens of shape (batch, num_tokens, dim)
        metric: Metric tokens for similarity computation (batch, num_tokens, dim)
        r: Number of token pairs to fuse
        mode: Fusion mode ("mean" or "sum")
    
    Returns:
        fused_tokens: Fused tokens of shape (batch, num_tokens - r, dim)
        fused_indices: Indices mapping of shape (batch, num_tokens - r)
    """
    B, T, C = tokens.shape
    r = min(r, T // 2)
    if r <= 0:
        return tokens, torch.arange(T, device=tokens.device)[None, :].expand(B, -1)
    
    # Use the same bipartite matching as ToMe
    from openpi.models_pytorch.tome_pytorch import bipartite_soft_merge
    
    fused_tokens, fused_indices = bipartite_soft_merge(metric, tokens, r, mode)
    return fused_tokens, fused_indices


def apply_tofu(
    tokens: torch.Tensor,
    ratio: float = 0.5,
    method: str = "norm",
    use_fusion: bool = True,
    fusion_ratio: float = 0.5,
    enabled: bool = True,
) -> torch.Tensor:
    """
    Apply Token Filtering/Fusion (ToFu) to reduce token count.
    
    Args:
        tokens: Input tokens of shape (batch, num_tokens, dim)
        ratio: Ratio of tokens to keep after filtering (0.0 to 1.0)
        method: Method to compute importance ("norm", "variance", "attention")
        use_fusion: Whether to apply fusion after filtering
        fusion_ratio: Ratio of tokens to fuse (applied to filtered tokens)
        enabled: Whether to apply ToFu. If False, returns original tokens.
    
    Returns:
        Filtered/fused tokens (or original if disabled)
    
    Example:
        >>> tokens = torch.randn(1, 100, 768)
        >>> filtered = apply_tofu(tokens, ratio=0.75, enabled=True)
        >>> # filtered.shape = (1, 75, 768) if use_fusion=False
    """
    if not enabled:
        return tokens
    
    batch_size, num_tokens, dim = tokens.shape
    
    if num_tokens <= 1:
        return tokens
    
    # Step 1: Compute importance scores
    importance_scores = compute_token_importance(tokens, method=method)
    
    # Step 2: Filter tokens based on importance
    filtered_tokens, keep_indices = filter_tokens(
        tokens, importance_scores, ratio=ratio
    )
    
    # Step 3: Optionally fuse similar tokens
    if use_fusion and filtered_tokens.shape[1] > 1:
        # Calculate number of tokens to fuse
        num_filtered = filtered_tokens.shape[1]
        num_fuse = max(0, int(num_filtered * (1 - fusion_ratio)))
        
        if num_fuse > 0:
            # Use filtered tokens as metric for fusion
            fused_tokens, _ = fuse_similar_tokens(
                filtered_tokens,
                filtered_tokens,  # Use same tokens as metric
                r=num_fuse,
                mode="mean",
            )
            return fused_tokens
    
    return filtered_tokens

