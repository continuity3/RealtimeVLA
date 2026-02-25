"""SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference

Based on: https://github.com/Gumpest/SparseVLMs
Paper: SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference
        SparseVLM+: Visual Token Sparsification with Improved Text-Visual Attention Pattern

SparseVLM sparsifies visual tokens adaptively based on the question prompt,
unlike text-agnostic methods. It uses text-visual attention to select relevant visual patches.

Reference: https://arxiv.org/pdf/2410.04417
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_text_visual_attention(
    vision_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    method: str = "cross_attention",
) -> torch.Tensor:
    """
    Compute attention scores between text and vision tokens.
    
    Args:
        vision_tokens: Vision token embeddings [B, num_vision, D]
        text_tokens: Text token embeddings [B, num_text, D]
        method: Method for computing attention ("cross_attention", "cosine", "dot_product")
    
    Returns:
        attention_scores: [B, num_vision] - importance scores for each vision token
    """
    B, num_vision, D = vision_tokens.shape
    _, num_text, _ = text_tokens.shape
    
    if method == "cross_attention":
        # Compute cross-attention: how much each vision token attends to text tokens
        # Normalize for cosine similarity
        vision_norm = F.normalize(vision_tokens, p=2, dim=-1)  # [B, num_vision, D]
        text_norm = F.normalize(text_tokens, p=2, dim=-1)  # [B, num_text, D]
        
        # Compute similarity matrix: [B, num_vision, num_text]
        similarity = torch.bmm(vision_norm, text_norm.transpose(1, 2))
        
        # Aggregate across text tokens: max or mean
        attention_scores = similarity.max(dim=-1)[0]  # [B, num_vision] - max attention
        # Alternative: attention_scores = similarity.mean(dim=-1)  # mean attention
        
    elif method == "cosine":
        # Average cosine similarity with all text tokens
        vision_norm = F.normalize(vision_tokens, p=2, dim=-1)  # [B, num_vision, D]
        text_norm = F.normalize(text_tokens, p=2, dim=-1)  # [B, num_text, D]
        
        # Compute similarity: [B, num_vision, num_text]
        similarity = torch.bmm(vision_norm, text_norm.transpose(1, 2))
        
        # Average across text tokens
        attention_scores = similarity.mean(dim=-1)  # [B, num_vision]
        
    elif method == "dot_product":
        # Simple dot product with average text token
        text_avg = text_tokens.mean(dim=1, keepdim=True)  # [B, 1, D]
        attention_scores = (vision_tokens * text_avg).sum(dim=-1)  # [B, num_vision]
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return attention_scores


def sparsify_vision_tokens(
    vision_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    num_retain: int = 192,
    method: str = "cross_attention",
    topk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sparsify vision tokens based on text-visual attention.
    
    Args:
        vision_tokens: Vision token embeddings [B, num_vision, D]
        text_tokens: Text token embeddings [B, num_text, D]
        num_retain: Number of vision tokens to retain (192, 128, 96, or 64)
        method: Method for computing text-visual attention
        topk: If True, use top-k selection; if False, use threshold-based
    
    Returns:
        sparsified_vision: Sparsified vision tokens [B, num_retain, D]
        keep_mask: Boolean mask indicating retained tokens [B, num_vision]
    """
    B, num_vision, D = vision_tokens.shape
    
    # Ensure num_retain is valid
    num_retain = min(num_retain, num_vision)
    num_retain = max(1, num_retain)
    
    if num_retain >= num_vision:
        # No sparsification needed
        keep_mask = torch.ones(B, num_vision, dtype=torch.bool, device=vision_tokens.device)
        return vision_tokens, keep_mask
    
    # Compute text-visual attention scores
    attention_scores = compute_text_visual_attention(
        vision_tokens, text_tokens, method=method
    )  # [B, num_vision]
    
    # Select tokens based on attention scores
    if topk:
        # Top-k selection: keep tokens with highest attention scores
        _, selected_indices = torch.topk(
            attention_scores,
            k=num_retain,
            dim=-1,
            sorted=False
        )  # [B, num_retain]
        
        # Create keep mask
        keep_mask = torch.zeros(B, num_vision, dtype=torch.bool, device=vision_tokens.device)
        batch_indices = torch.arange(B, device=vision_tokens.device)[:, None]  # [B, 1]
        keep_mask[batch_indices, selected_indices] = True
        
        # Gather selected tokens
        sparsified_vision = torch.zeros(
            (B, num_retain, D),
            dtype=vision_tokens.dtype,
            device=vision_tokens.device
        )
        for b in range(B):
            indices = selected_indices[b, :]  # [num_retain]
            sparsified_vision[b, :, :] = vision_tokens[b, indices, :]
    else:
        # Threshold-based selection
        threshold = attention_scores.quantile(1.0 - num_retain / num_vision, dim=-1, keepdim=True)
        keep_mask = attention_scores >= threshold  # [B, num_vision]
        
        # Count actual number of kept tokens per batch
        num_keep_per_batch = keep_mask.sum(dim=-1)  # [B]
        max_keep = num_keep_per_batch.max().item()
        max_keep = min(max_keep, num_vision)
        
        # Gather selected tokens (pad to max_keep for batching)
        sparsified_vision = torch.zeros(
            (B, max_keep, D),
            dtype=vision_tokens.dtype,
            device=vision_tokens.device
        )
        for b in range(B):
            mask = keep_mask[b, :]  # [num_vision]
            selected = vision_tokens[b, mask, :]  # [num_selected, D]
            sparsified_vision[b, :selected.shape[0], :] = selected
        
        # Adjust num_retain to max_keep for consistency
        num_retain = max_keep
    
    return sparsified_vision, keep_mask


def apply_sparsevlm(
    vision_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    num_retain: int = 192,
    method: str = "cross_attention",
    version: str = "1.5",  # "1.5" or "2.0" (SparseVLM+)
    enabled: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply SparseVLM sparsification to vision tokens.
    
    Args:
        vision_tokens: Vision token embeddings [B, num_vision, D]
        text_tokens: Text token embeddings [B, num_text, D]
        num_retain: Number of vision tokens to retain (192, 128, 96, or 64)
        method: Method for computing text-visual attention
        version: SparseVLM version ("1.5" or "2.0" for SparseVLM+)
        enabled: Whether to apply SparseVLM
    
    Returns:
        sparsified_vision: Sparsified vision tokens [B, num_retain, D]
        keep_mask: Boolean mask indicating retained tokens [B, num_vision]
    
    Note:
        - SparseVLM+ (v2.0) uses improved text-visual attention pattern
        - For v2.0, you may want to use a different attention computation method
    """
    if not enabled:
        keep_mask = torch.ones(
            vision_tokens.shape[0], vision_tokens.shape[1],
            dtype=torch.bool, device=vision_tokens.device
        )
        return vision_tokens, keep_mask
    
    # Adjust method for SparseVLM+ if needed
    if version == "2.0":
        # SparseVLM+ uses improved attention pattern
        # You can customize this based on the paper's description
        if method == "cross_attention":
            method = "cross_attention"  # Keep same, or use improved variant
        # Add any SparseVLM+ specific modifications here
    
    return sparsify_vision_tokens(
        vision_tokens, text_tokens,
        num_retain=num_retain,
        method=method,
        topk=True,
    )

