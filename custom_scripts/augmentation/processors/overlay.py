"""Overlay processors for combining modalities."""

import numpy as np

def mask_edge_overlay(mask: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Overlay edges onto segmentation mask via pixel addition clipped to 255.
    Args:
        mask: Segmentation mask (H, W, 3), uint8
        edges: Edge image (H, W, 3), uint8
    Returns:
        Combined image (H, W, 3), uint8
    """
    return np.clip(mask.astype(np.uint16) + edges.astype(np.uint16), 0, 255).astype(np.uint8)


def original_mask_edge_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """
    Overlay mask and edges onto original RGB via pixel addition clipped to 255.
    Args:
        original: Original RGB frame (H, W, 3), uint8
        mask: Segmentation mask (H, W, 3), uint8
        edges: Edge image (H, W, 3), uint8
    Returns:
        Combined image (H, W, 3), uint8
    """
    return np.clip(original.astype(np.uint16) + mask.astype(np.uint16) + edges.astype(np.uint16),
                   0, 255).astype(np.uint8) # superimposed pixel values capped to 255.
