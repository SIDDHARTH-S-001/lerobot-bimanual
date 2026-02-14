"""Depth estimation model wrapper."""

import logging

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

log = logging.getLogger(__name__)

def _get_device(requested: str) -> str: 
    return "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"

class DepthEstimator:
    """Depth estimation using HuggingFace pipeline (DA_v2 or ZoeDepth)."""
    def __init__(self, model_id: str, colormap: str = "INFERNO", device: str = "cuda"):
        actual_device = _get_device(device)
        log.info("Depth model will run on: %s", actual_device)
        self.pipe = pipeline(task="depth-estimation", model=model_id, device=actual_device)
        self.colormap = getattr(cv2, f"COLORMAP_{colormap.upper()}", cv2.COLORMAP_INFERNO)

    def estimate(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth and return colorized RGB visualization.
        Args:
            rgb_frame: RGB image (H, W, 3), uint8
        Returns:
            Depth map (H, W, 3), uint8, RGB colorized
        """
        pil_img = Image.fromarray(rgb_frame)
        outputs = self.pipe(pil_img)
        depth_np = np.array(outputs["depth"])

        # Normalize to 0-255
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)

        # Apply colormap (produces BGR) then convert to RGB
        depth_colored = cv2.applyColorMap(depth_uint8, self.colormap)
        depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

        return depth_rgb
