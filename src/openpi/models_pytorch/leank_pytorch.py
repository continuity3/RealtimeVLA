"""LeanK: Learnable K Cache Channel Pruning for Efficient Decoding

LeanK is a learnable method for pruning K cache channels (head_dim dimension)
to reduce memory and computation during decoding.

Usage:
    In pi0_config.py, set:
        leank_enabled: bool = True
        leank_pruning_ratio: float = 0.5  # Keep 50% of channels
        leank_method: str = "magnitude"  # "magnitude", "variance", or "learnable"
        leank_topk: bool = True  # Use top-k selection

Note:
    - Channel pruning changes head_dim, which may require attention mechanism adjustment
    - For inference, use "magnitude" or "variance" methods (no training needed)
    - "learnable" method requires training a scorer module
    - Currently works with DynamicCache and list/tuple cache formats

Reference: Based on channel pruning techniques for KV cache optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np


class ChannelScorer(nn.Module):
    """
    Learnable scorer for selecting important channels in K cache.
    This module learns to score each channel dimension based on its importance.
    """
    def __init__(self, head_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.head_dim = head_dim
        # Use a small MLP to score channels
        self.scorer = nn.Sequential(
            nn.Linear(head_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        Score channels in K cache.
        
        Args:
            k: Key states [batch, num_heads, seq_len, head_dim]
        
        Returns:
            channel_scores: [batch, num_heads, head_dim] - importance scores for each channel
        """
        batch_size, num_heads, seq_len, head_dim = k.shape
        
        # Aggregate across sequence dimension (mean pooling)
        k_aggregated = k.mean(dim=2)  # [B, H, head_dim]
        
        # Score each channel
        # Reshape for scorer: [B*H, head_dim] -> [B*H, 1]
        k_flat = k_aggregated.reshape(-1, head_dim)  # [B*H, head_dim]
        scores_flat = self.scorer(k_flat)  # [B*H, 1]
        scores = scores_flat.reshape(batch_size, num_heads, 1)  # [B, H, 1]
        
        # Expand to match head_dim for per-channel scoring
        # For simplicity, we use the same score for all channels initially
        # In practice, you might want to score each channel independently
        channel_scores = scores.expand(-1, -1, head_dim)  # [B, H, head_dim]
        
        return channel_scores


class PerChannelScorer(nn.Module):
    """
    More sophisticated scorer that scores each channel independently.
    """
    def __init__(self, head_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.head_dim = head_dim
        # Score each channel independently
        self.scorer = nn.Sequential(
            nn.Linear(head_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, head_dim),  # Output one score per channel
            nn.Sigmoid(),  # Normalize to [0, 1]
        )
    
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        Score each channel in K cache independently.
        
        Args:
            k: Key states [batch, num_heads, seq_len, head_dim]
        
        Returns:
            channel_scores: [batch, num_heads, head_dim] - importance scores for each channel
        """
        batch_size, num_heads, seq_len, head_dim = k.shape
        
        # Aggregate across sequence dimension (mean pooling)
        k_aggregated = k.mean(dim=2)  # [B, H, head_dim]
        
        # Score each channel independently
        k_flat = k_aggregated.reshape(-1, head_dim)  # [B*H, head_dim]
        scores_flat = self.scorer(k_flat)  # [B*H, head_dim]
        channel_scores = scores_flat.reshape(batch_size, num_heads, head_dim)  # [B, H, head_dim]
        
        return channel_scores


def compute_channel_importance(
    k: torch.Tensor,
    method: str = "learnable",
    scorer: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Compute importance scores for each channel in K cache.
    
    Args:
        k: Key states [batch, num_heads, seq_len, head_dim]
        method: Method for computing importance ("learnable", "magnitude", "variance")
        scorer: Learnable scorer module (required if method="learnable")
    
    Returns:
        channel_scores: [batch, num_heads, head_dim] - importance scores
    """
    batch_size, num_heads, seq_len, head_dim = k.shape
    
    if method == "learnable":
        if scorer is None:
            raise ValueError("Scorer module required for learnable method")
        channel_scores = scorer(k)
    elif method == "magnitude":
        # Use L2 norm of each channel across sequence
        channel_scores = torch.norm(k, p=2, dim=2)  # [B, H, head_dim]
        # Normalize to [0, 1]
        channel_scores = channel_scores / (channel_scores.max(dim=-1, keepdim=True)[0] + 1e-8)
    elif method == "variance":
        # Use variance of each channel across sequence
        channel_scores = torch.var(k, dim=2)  # [B, H, head_dim]
        # Normalize to [0, 1]
        channel_scores = channel_scores / (channel_scores.max(dim=-1, keepdim=True)[0] + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return channel_scores


def prune_k_cache_channels(
    k: torch.Tensor,
    v: torch.Tensor,
    pruning_ratio: float = 0.5,
    method: str = "learnable",
    scorer: Optional[nn.Module] = None,
    topk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prune channels in K cache based on importance scores.
    V cache is kept intact.
    
    Args:
        k: Key states [batch, num_heads, seq_len, head_dim]
        v: Value states [batch, num_heads, seq_len, head_dim] (kept intact)
        pruning_ratio: Ratio of channels to keep (0.0 to 1.0). 0.5 means keep 50% of channels
        method: Method for computing importance ("learnable", "magnitude", "variance")
        scorer: Learnable scorer module (required if method="learnable")
        topk: If True, use top-k selection; if False, use threshold-based selection
    
    Returns:
        pruned_k: Pruned key states [batch, num_heads, seq_len, new_head_dim]
        v: Value states (unchanged) [batch, num_heads, seq_len, head_dim]
        channel_mask: Boolean mask indicating selected channels [batch, num_heads, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = k.shape
    
    # Calculate number of channels to keep
    num_keep = max(1, int(head_dim * pruning_ratio))
    num_keep = min(num_keep, head_dim)
    
    if num_keep >= head_dim:
        # No pruning needed
        channel_mask = torch.ones(
            (batch_size, num_heads, head_dim),
            dtype=torch.bool,
            device=k.device
        )
        return k, v, channel_mask
    
    # Compute channel importance scores
    channel_scores = compute_channel_importance(k, method=method, scorer=scorer)
    
    # Select channels based on scores
    if topk:
        # Top-k selection: keep channels with highest scores
        _, selected_indices = torch.topk(
            channel_scores,
            k=num_keep,
            dim=-1,
            sorted=False
        )  # [B, H, num_keep]
        
        # Create mask
        channel_mask = torch.zeros(
            (batch_size, num_heads, head_dim),
            dtype=torch.bool,
            device=k.device
        )
        for b in range(batch_size):
            for h in range(num_heads):
                channel_mask[b, h, selected_indices[b, h, :]] = True
        
        # Gather selected channels
        pruned_k = torch.zeros(
            (batch_size, num_heads, seq_len, num_keep),
            dtype=k.dtype,
            device=k.device
        )
        for b in range(batch_size):
            for h in range(num_heads):
                indices = selected_indices[b, h, :]  # [num_keep]
                pruned_k[b, h, :, :] = k[b, h, :, indices]
    else:
        # Threshold-based selection: keep channels with scores above threshold
        threshold = channel_scores.quantile(1.0 - pruning_ratio, dim=-1, keepdim=True)
        channel_mask = channel_scores >= threshold  # [B, H, head_dim]
        
        # Count actual number of kept channels per head
        num_keep_per_head = channel_mask.sum(dim=-1)  # [B, H]
        max_keep = num_keep_per_head.max().item()
        
        # Gather selected channels (pad to max_keep for batching)
        pruned_k = torch.zeros(
            (batch_size, num_heads, seq_len, max_keep),
            dtype=k.dtype,
            device=k.device
        )
        for b in range(batch_size):
            for h in range(num_heads):
                mask = channel_mask[b, h, :]  # [head_dim]
                selected_k = k[b, h, :, mask]  # [seq_len, num_selected]
                pruned_k[b, h, :, :selected_k.shape[1]] = selected_k
        
        # Adjust num_keep to max_keep for consistency
        num_keep = max_keep
    
    return pruned_k, v, channel_mask


def apply_leank_to_cache(
    past_key_values,
    pruning_ratio: float = 0.5,
    method: str = "magnitude",  # Default to magnitude for inference without training
    scorer: Optional[nn.Module] = None,
    topk: bool = True,
    enabled: bool = True,
):
    """
    Apply LeanK channel pruning to a transformers Cache object.
    Only prunes K cache channels, keeps V cache intact.
    
    Args:
        past_key_values: Transformers Cache object (e.g., DynamicCache)
        pruning_ratio: Ratio of channels to keep in K cache
        method: Method for computing importance ("learnable", "magnitude", "variance")
        scorer: Learnable scorer module (required if method="learnable")
        topk: If True, use top-k selection; if False, use threshold-based
        enabled: Whether to apply LeanK
    
    Returns:
        Pruned past_key_values (same type as input)
    """
    if not enabled or past_key_values is None:
        return past_key_values
    
    # Handle different cache types
    if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
        # DynamicCache or similar
        # Note: Channel pruning changes head_dim, which may break compatibility
        # with the model's attention mechanism. This is a simplified implementation.
        # In practice, you may need to adjust the attention computation accordingly.
        
        compressed_cache = type(past_key_values)()
        compressed_cache.key_cache = []
        compressed_cache.value_cache = []
        
        num_layers = len(past_key_values.key_cache)
        
        for layer_idx in range(num_layers):
            k = past_key_values.key_cache[layer_idx]  # [B, num_kv_heads, seq_len, head_dim]
            v = past_key_values.value_cache[layer_idx]  # [B, num_kv_heads, seq_len, head_dim]
            
            # Ensure we have the right shape: [B, H, seq_len, D]
            if k.dim() == 4:
                pruned_k, v_kept, channel_mask = prune_k_cache_channels(
                    k, v,
                    pruning_ratio=pruning_ratio,
                    method=method,
                    scorer=scorer,
                    topk=topk,
                )
                
                # Store pruned K and original V
                # Note: This changes head_dim, which may require attention mechanism adjustment
                while len(compressed_cache.key_cache) <= layer_idx:
                    compressed_cache.key_cache.append(None)
                    compressed_cache.value_cache.append(None)
                compressed_cache.key_cache[layer_idx] = pruned_k
                compressed_cache.value_cache[layer_idx] = v_kept
            else:
                # Unexpected shape, skip pruning for this layer
                compressed_cache.key_cache.append(k)
                compressed_cache.value_cache.append(v)
        
        return compressed_cache
    
    elif isinstance(past_key_values, (list, tuple)):
        # List of tuples: [(k0, v0), (k1, v1), ...]
        compressed_cache = []
        
        for layer_idx, (k, v) in enumerate(past_key_values):
            # k, v: [B, H, seq_len, D]
            pruned_k, v_kept, channel_mask = prune_k_cache_channels(
                k, v,
                pruning_ratio=pruning_ratio,
                method=method,
                scorer=scorer,
                topk=topk,
            )
            compressed_cache.append((pruned_k, v_kept))
        
        return compressed_cache
    
    else:
        # Unknown cache type, return as is
        return past_key_values

