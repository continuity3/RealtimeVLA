"""V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models.

Based on: https://github.com/xuyang-liu16/V2Drop
Paper: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models

V2Drop measures token-wise variation across adjacent LLM layers and progressively
drops vision tokens with minimal variation, achieving plug-and-play inference acceleration.
"""

import torch
import torch.nn.functional as F


def compute_token_variation(
    tokens_before: torch.Tensor,
    tokens_after: torch.Tensor,
    method: str = "l2",
) -> torch.Tensor:
    """
    Compute variation (change) of tokens between two layers.
    
    Args:
        tokens_before: Tokens from previous layer [B, T, D]
        tokens_after: Tokens from current layer [B, T, D]
        method: Variation computation method ("l2", "cosine", "abs")
    
    Returns:
        variation: Token-wise variation scores [B, T]
    """
    if method == "l2":
        # L2 distance between tokens
        variation = torch.norm(tokens_after - tokens_before, p=2, dim=-1)  # [B, T]
    elif method == "cosine":
        # Cosine distance (1 - cosine similarity)
        tokens_before_norm = F.normalize(tokens_before, p=2, dim=-1)
        tokens_after_norm = F.normalize(tokens_after, p=2, dim=-1)
        cosine_sim = (tokens_before_norm * tokens_after_norm).sum(dim=-1)  # [B, T]
        variation = 1.0 - cosine_sim  # Variation = 1 - similarity
    elif method == "abs":
        # L1 distance
        variation = torch.norm(tokens_after - tokens_before, p=1, dim=-1)  # [B, T]
    else:
        raise ValueError(f"Unknown variation method: {method}")
    
    return variation


def drop_tokens_by_variation(
    tokens: torch.Tensor,
    variation_scores: torch.Tensor,
    num_vision_tokens: int,
    drop_ratio: float = 0.5,
    min_tokens: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Drop vision tokens with minimal variation.
    
    Args:
        tokens: All tokens [B, T, D] (includes vision + language tokens)
        variation_scores: Variation scores [B, T]
        num_vision_tokens: Number of vision tokens (from start)
        drop_ratio: Ratio of vision tokens to drop (0.0 to 1.0)
        min_tokens: Minimum number of vision tokens to keep
    
    Returns:
        filtered_tokens: Tokens after dropping [B, T_new, D]
        keep_mask: Boolean mask indicating kept tokens [B, T]
    """
    B, T, D = tokens.shape
    
    # Only drop from vision tokens (first num_vision_tokens)
    num_keep = max(min_tokens, int(num_vision_tokens * (1.0 - drop_ratio)))
    num_keep = min(num_keep, num_vision_tokens)  # Can't keep more than we have
    
    if num_keep >= num_vision_tokens:
        # No dropping needed
        keep_mask = torch.ones(B, T, dtype=torch.bool, device=tokens.device)
        return tokens, keep_mask
    
    # Get variation scores for vision tokens only
    vision_variation = variation_scores[:, :num_vision_tokens]  # [B, num_vision_tokens]
    
    # Select top-k tokens with highest variation (most important)
    _, top_indices = torch.topk(vision_variation, k=num_keep, dim=-1)  # [B, num_keep]
    
    # Create keep mask for vision tokens
    vision_keep_mask = torch.zeros(B, num_vision_tokens, dtype=torch.bool, device=tokens.device)
    batch_indices = torch.arange(B, device=tokens.device)[:, None]  # [B, 1]
    vision_keep_mask[batch_indices, top_indices] = True
    
    # Language tokens are always kept
    lang_keep_mask = torch.ones(B, T - num_vision_tokens, dtype=torch.bool, device=tokens.device)
    
    # Concatenate masks
    keep_mask = torch.cat([vision_keep_mask, lang_keep_mask], dim=1)  # [B, T]
    
    # Filter tokens using gather
    # For each batch, get indices of kept tokens
    filtered_tokens_list = []
    for b in range(B):
        batch_keep_mask = keep_mask[b]  # [T]
        batch_keep_indices = torch.where(batch_keep_mask)[0]  # [num_keep]
        filtered_tokens_list.append(tokens[b, batch_keep_indices])  # [num_keep, D]
    
    filtered_tokens = torch.stack(filtered_tokens_list, dim=0)  # [B, num_keep, D]
    
    return filtered_tokens, keep_mask


def apply_v2drop(
    tokens_before: torch.Tensor,
    tokens_after: torch.Tensor,
    num_vision_tokens: int,
    drop_ratio: float = 0.5,
    method: str = "l2",
    enabled: bool = True,
    min_tokens: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply V2Drop: drop vision tokens with minimal variation.
    
    Args:
        tokens_before: Tokens from previous LLM layer [B, T, D]
        tokens_after: Tokens from current LLM layer [B, T, D]
        num_vision_tokens: Number of vision tokens (from start of sequence)
        drop_ratio: Ratio of vision tokens to drop (0.0 to 1.0)
        method: Variation computation method ("l2", "cosine", "abs")
        enabled: Whether to apply V2Drop
        min_tokens: Minimum number of vision tokens to keep
    
    Returns:
        filtered_tokens: Tokens after dropping [B, T_new, D]
        keep_mask: Boolean mask indicating kept tokens [B, T]
    
    Example:
        >>> tokens_before = torch.randn(1, 100, 768)
        >>> tokens_after = torch.randn(1, 100, 768)
        >>> filtered, mask = apply_v2drop(
        ...     tokens_before, tokens_after, 
        ...     num_vision_tokens=50, drop_ratio=0.5
        ... )
        >>> # filtered.shape = (1, 75, 768) if 25 vision tokens were dropped
    """
    if not enabled:
        keep_mask = torch.ones(
            tokens_after.shape[0], tokens_after.shape[1],
            dtype=torch.bool, device=tokens_after.device
        )
        return tokens_after, keep_mask
    
    if num_vision_tokens <= min_tokens:
        # Can't drop any tokens
        keep_mask = torch.ones(
            tokens_after.shape[0], tokens_after.shape[1],
            dtype=torch.bool, device=tokens_after.device
        )
        return tokens_after, keep_mask
    
    # Compute variation scores
    variation_scores = compute_token_variation(tokens_before, tokens_after, method=method)
    
    # Drop tokens with minimal variation
    filtered_tokens, keep_mask = drop_tokens_by_variation(
        tokens_after,
        variation_scores,
        num_vision_tokens=num_vision_tokens,
        drop_ratio=drop_ratio,
        min_tokens=min_tokens,
    )
    
    return filtered_tokens, keep_mask

