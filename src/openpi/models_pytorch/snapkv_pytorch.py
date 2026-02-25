"""SnapKV: LLM Knows What You are Looking for Before Generation

SnapKV is a fine-tuning-free method for compressing KV caches in LLMs.
It identifies important KV positions for each attention head using an observation window
and compresses the KV cache before generation.

Reference: https://arxiv.org/abs/2404.14469
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from collections import defaultdict
import numpy as np


def compute_attention_features(
    k: torch.Tensor,
    v: torch.Tensor,
    observation_window: int = 32,
) -> torch.Tensor:
    """
    Compute attention features from the observation window.
    
    Args:
        k: Key states [batch, num_heads, seq_len, head_dim]
        v: Value states [batch, num_heads, seq_len, head_dim]
        observation_window: Number of tokens at the end to use as observation window
    
    Returns:
        attention_features: [batch, num_heads, seq_len] - attention importance scores
    """
    batch_size, num_heads, seq_len, head_dim = k.shape
    
    if seq_len <= observation_window:
        # If sequence is shorter than observation window, use all tokens
        obs_k = k
        obs_v = v
    else:
        # Use last observation_window tokens as observation window
        obs_k = k[:, :, -observation_window:, :]  # [B, H, obs_win, D]
        obs_v = v[:, :, -observation_window:, :]
    
    # Compute attention scores: how much each token in the full sequence
    # is attended to by tokens in the observation window
    # This is done by computing similarity between full sequence and observation window
    
    # Normalize for cosine similarity
    k_norm = F.normalize(k, p=2, dim=-1)  # [B, H, seq_len, D]
    obs_k_norm = F.normalize(obs_k, p=2, dim=-1)  # [B, H, obs_win, D]
    
    # Compute attention-like scores: for each position in full sequence,
    # compute max similarity with any token in observation window
    # [B, H, seq_len, obs_win]
    similarities = torch.einsum('bhnd,bhmd->bhnm', k_norm, obs_k_norm)
    
    # For each position, get max attention score across observation window
    attention_features = similarities.max(dim=-1)[0]  # [B, H, seq_len]
    
    return attention_features


def cluster_kv_positions(
    attention_features: torch.Tensor,
    num_clusters: int,
    method: str = "kmeans",
) -> torch.Tensor:
    """
    Cluster KV positions based on attention features to identify important positions.
    
    Args:
        attention_features: [batch, num_heads, seq_len] - attention importance scores
        num_clusters: Number of clusters (important positions to keep)
        method: Clustering method ("kmeans" or "topk")
    
    Returns:
        selected_indices: [batch, num_heads, num_clusters] - indices of selected KV positions
    """
    batch_size, num_heads, seq_len = attention_features.shape
    
    if method == "topk":
        # Simple top-k selection based on attention features
        _, selected_indices = torch.topk(
            attention_features,
            k=min(num_clusters, seq_len),
            dim=-1,
            sorted=False
        )  # [B, H, num_clusters]
        return selected_indices
    
    elif method == "kmeans":
        # K-means clustering on attention features
        # For simplicity, we use a greedy approach: select positions with highest
        # attention features, but ensure diversity across the sequence
        selected_indices = torch.zeros(
            (batch_size, num_heads, num_clusters),
            dtype=torch.long,
            device=attention_features.device
        )
        
        for b in range(batch_size):
            for h in range(num_heads):
                features = attention_features[b, h, :]  # [seq_len]
                
                # Greedy selection: iteratively select positions
                # that maximize attention while maintaining diversity
                selected = []
                remaining = set(range(seq_len))
                
                # First, select the position with highest attention
                if len(remaining) > 0:
                    first_idx = features.argmax().item()
                    selected.append(first_idx)
                    remaining.remove(first_idx)
                
                # Then, iteratively select positions that are both
                # high in attention and diverse from already selected
                for _ in range(min(num_clusters - 1, len(remaining))):
                    if not remaining:
                        break
                    
                    best_score = -float('inf')
                    best_idx = None
                    
                    for idx in remaining:
                        # Score combines attention value and diversity
                        attention_score = features[idx].item()
                        
                        # Diversity: distance from already selected positions
                        if selected:
                            min_dist = min(abs(idx - s) for s in selected)
                            diversity = min_dist / seq_len  # normalize
                        else:
                            diversity = 1.0
                        
                        # Combined score (can be tuned)
                        score = attention_score + 0.1 * diversity
                        
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                    
                    if best_idx is not None:
                        selected.append(best_idx)
                        remaining.remove(best_idx)
                
                # Fill remaining slots with top-k if needed
                while len(selected) < num_clusters and remaining:
                    remaining_list = list(remaining)
                    remaining_features = features[remaining_list]
                    next_idx = remaining_list[remaining_features.argmax().item()]
                    selected.append(next_idx)
                    remaining.remove(next_idx)
                
                # Pad with last position if needed
                while len(selected) < num_clusters:
                    selected.append(seq_len - 1)
                
                selected_indices[b, h, :] = torch.tensor(
                    selected[:num_clusters],
                    device=attention_features.device
                )
        
        return selected_indices
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def compress_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    compression_ratio: float = 0.5,
    observation_window: int = 32,
    clustering_method: str = "topk",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compress KV cache using SnapKV method.
    
    Args:
        k: Key states [batch, num_heads, seq_len, head_dim]
        v: Value states [batch, num_heads, seq_len, head_dim]
        compression_ratio: Ratio of KV positions to keep (0.0 to 1.0)
        observation_window: Number of tokens at the end to use as observation window
        clustering_method: Method for selecting KV positions ("topk" or "kmeans")
    
    Returns:
        compressed_k: Compressed key states [batch, num_heads, new_seq_len, head_dim]
        compressed_v: Compressed value states [batch, num_heads, new_seq_len, head_dim]
        selected_indices: Indices of selected positions [batch, num_heads, new_seq_len]
    """
    batch_size, num_heads, seq_len, head_dim = k.shape
    
    # Calculate number of positions to keep
    num_keep = max(1, int(seq_len * compression_ratio))
    num_keep = min(num_keep, seq_len)
    
    if num_keep >= seq_len:
        # No compression needed
        indices = torch.arange(seq_len, device=k.device)[None, None, :].expand(
            batch_size, num_heads, -1
        )
        return k, v, indices
    
    # Compute attention features from observation window
    attention_features = compute_attention_features(
        k, v, observation_window=observation_window
    )
    
    # Cluster to select important positions
    selected_indices = cluster_kv_positions(
        attention_features,
        num_clusters=num_keep,
        method=clustering_method,
    )  # [B, H, num_keep]
    
    # Gather selected KV positions
    # Use advanced indexing to select positions for each head
    batch_size, num_heads, seq_len, head_dim = k.shape
    compressed_k = torch.zeros(
        (batch_size, num_heads, num_keep, head_dim),
        dtype=k.dtype,
        device=k.device
    )
    compressed_v = torch.zeros(
        (batch_size, num_heads, num_keep, head_dim),
        dtype=v.dtype,
        device=v.device
    )
    
    for b in range(batch_size):
        for h in range(num_heads):
            indices = selected_indices[b, h, :]  # [num_keep]
            compressed_k[b, h, :, :] = k[b, h, indices, :]
            compressed_v[b, h, :, :] = v[b, h, indices, :]
    
    return compressed_k, compressed_v, selected_indices


def apply_snapkv_to_cache(
    past_key_values,
    compression_ratio: float = 0.5,
    observation_window: int = 32,
    clustering_method: str = "topk",
    enabled: bool = True,
):
    """
    Apply SnapKV compression to a transformers Cache object.
    
    Args:
        past_key_values: Transformers Cache object (e.g., DynamicCache)
        compression_ratio: Ratio of KV positions to keep
        observation_window: Number of tokens at the end to use as observation window
        clustering_method: Method for selecting KV positions
        enabled: Whether to apply SnapKV
    
    Returns:
        Compressed past_key_values (same type as input)
    """
    if not enabled or past_key_values is None:
        return past_key_values
    
    # Handle different cache types
    if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
        # DynamicCache or similar
        # Create a new cache and directly set the compressed values
        compressed_cache = type(past_key_values)()
        compressed_cache.key_cache = []
        compressed_cache.value_cache = []
        
        num_layers = len(past_key_values.key_cache)
        
        for layer_idx in range(num_layers):
            k = past_key_values.key_cache[layer_idx]  # [B, num_kv_heads, seq_len, head_dim]
            v = past_key_values.value_cache[layer_idx]  # [B, num_kv_heads, seq_len, head_dim]
            
            # Ensure we have the right shape: [B, H, seq_len, D]
            if k.dim() == 4:
                compressed_k, compressed_v, _ = compress_kv_cache(
                    k, v,
                    compression_ratio=compression_ratio,
                    observation_window=observation_window,
                    clustering_method=clustering_method,
                )
            else:
                # Unexpected shape, skip compression for this layer
                compressed_k, compressed_v = k, v
            
            # Directly set the cache values (don't use update which would concatenate)
            while len(compressed_cache.key_cache) <= layer_idx:
                compressed_cache.key_cache.append(None)
                compressed_cache.value_cache.append(None)
            compressed_cache.key_cache[layer_idx] = compressed_k
            compressed_cache.value_cache[layer_idx] = compressed_v
        
        return compressed_cache
    
    elif isinstance(past_key_values, (list, tuple)):
        # List of tuples: [(k0, v0), (k1, v1), ...]
        compressed_cache = []
        
        for layer_idx, (k, v) in enumerate(past_key_values):
            # k, v: [B, H, seq_len, D]
            compressed_k, compressed_v, _ = compress_kv_cache(
                k, v,
                compression_ratio=compression_ratio,
                observation_window=observation_window,
                clustering_method=clustering_method,
            )
            compressed_cache.append((compressed_k, compressed_v))
        
        return compressed_cache
    
    else:
        # Unknown cache type, return as is
        return past_key_values

