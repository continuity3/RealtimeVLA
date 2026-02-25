"""Token Merging (ToMe) implementation for JAX/Flax.

ToMe is a technique to reduce the number of tokens in vision transformers
by merging similar tokens, which can significantly speed up inference
while maintaining performance.
"""

import jax
import jax.numpy as jnp
from typing import Optional


def bipartite_soft_matching(
    tokens: jnp.ndarray,
    r: int,
    metric: str = "cosine",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Bipartite soft matching algorithm for token merging.
    Optimized version using JAX vectorized operations.
    
    Args:
        tokens: Input tokens of shape (batch, num_tokens, dim)
        r: Number of tokens to remove (merge)
        metric: Similarity metric to use ("cosine" or "euclidean")
    
    Returns:
        merged_tokens: Merged tokens of shape (batch, num_tokens - r, dim)
        merge_weights: Merge weights for tracking
    """
    batch_size, num_tokens, dim = tokens.shape
    
    if r <= 0 or num_tokens <= r:
        return tokens, jnp.ones((batch_size, num_tokens, 1))
    
    # Normalize tokens for cosine similarity
    if metric == "cosine":
        tokens_norm = tokens / (jnp.linalg.norm(tokens, axis=-1, keepdims=True) + 1e-8)
    else:
        tokens_norm = tokens
    
    # Split tokens into two sets
    num_keep = num_tokens - r
    tokens_a = tokens_norm[:, :num_keep]  # (batch, num_keep, dim)
    tokens_b = tokens_norm[:, num_keep:]  # (batch, r, dim)
    
    # Compute similarity matrix
    if metric == "cosine":
        # (batch, num_keep, r)
        similarity = jnp.einsum("bnd,bmd->bnm", tokens_a, tokens_b)
    else:
        # Euclidean distance (negative for similarity)
        # (batch, num_keep, r)
        diff = tokens_a[:, :, None, :] - tokens_b[:, None, :, :]
        similarity = -jnp.linalg.norm(diff, axis=-1)
    
    # Find best matches: for each token in A, find best match in B
    # Use argmax along the last dimension
    best_matches = jnp.argmax(similarity, axis=-1)  # (batch, num_keep)
    best_sims = jnp.take_along_axis(
        similarity, 
        best_matches[:, :, None], 
        axis=-1
    ).squeeze(-1)  # (batch, num_keep)
    
    # Compute merge weights (sigmoid of similarity)
    merge_weights_a = 0.5 + 0.5 * jax.nn.sigmoid(best_sims)  # (batch, num_keep)
    merge_weights_b = 1.0 - merge_weights_a
    
    # Merge tokens: weighted average
    tokens_a_orig = tokens[:, :num_keep]  # (batch, num_keep, dim)
    tokens_b_orig = tokens[:, num_keep:]  # (batch, r, dim)
    
    # Gather matched tokens from B
    batch_indices = jnp.arange(batch_size)[:, None]  # (batch, 1)
    matched_tokens_b = tokens_b_orig[batch_indices, best_matches]  # (batch, num_keep, dim)
    
    # Merge
    merged_tokens_a = (
        merge_weights_a[:, :, None] * tokens_a_orig +
        merge_weights_b[:, :, None] * matched_tokens_b
    )  # (batch, num_keep, dim)
    
    # Find unmatched tokens in B (tokens that weren't selected)
    # For simplicity, we'll keep all tokens from A (merged) and skip unmatched B tokens
    # This is a simplified version - full implementation would track all unmatched tokens
    merged_tokens = merged_tokens_a
    
    # Create weights array
    merge_weights = merge_weights_a[:, :, None]  # (batch, num_keep, 1)
    
    return merged_tokens, merge_weights


def merge_tokens(
    tokens: jnp.ndarray,
    r: int,
    metric: str = "cosine",
) -> jnp.ndarray:
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
    tokens: jnp.ndarray,
    ratio: float = 0.5,
    metric: str = "cosine",
    enabled: bool = True,
) -> jnp.ndarray:
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
        >>> tokens = jnp.random.normal(key, (1, 100, 768))
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

