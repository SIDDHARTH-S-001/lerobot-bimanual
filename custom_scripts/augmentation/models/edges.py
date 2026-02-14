"""Canny edge detection wrapper."""

import cv2
import numpy as np

class CannyDetector:
    """Canny edge detector producing white edges on black background."""
    def __init__(self, threshold1: int = 100, threshold2: int = 200, aperture_size: int = 3):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size

    def detect(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Detect edges in an RGB frame.
        Args:
            rgb_frame: RGB image (H, W, 3), uint8
        Returns:
            Edge image (H, W, 3), uint8. White edges on black background.
        """
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.threshold1, self.threshold2, apertureSize=self.aperture_size)
        # Convert single-channel to 3-channel RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return edges_rgb
