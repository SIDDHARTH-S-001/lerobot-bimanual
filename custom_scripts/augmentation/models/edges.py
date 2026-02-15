"""Canny edge detection wrapper."""

import cv2
import numpy as np

class CannyDetector:
    """Canny edge detector producing white edges on black background."""
    def __init__(
        self,
        threshold1: int = 100,
        threshold2: int = 200,
        aperture_size: int = 3,
        sigma: float = 1.0,
        kernel_size: tuple[int, int] = (3, 3),
    ):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.sigma = sigma
        self.kernel_size = tuple(kernel_size)

    def detect(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Detect edges in an RGB frame.
        Args:
            rgb_frame: RGB image (H, W, 3), uint8
        Returns:
            Edge image (H, W, 3), uint8. White edges on black background.
        """
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian Blur pre-processing to reduce noise
        if self.sigma > 0:
            gray = cv2.GaussianBlur(gray, self.kernel_size, self.sigma)
            
        edges = cv2.Canny(gray, self.threshold1, self.threshold2, apertureSize=self.aperture_size)
        # Convert single-channel to 3-channel RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return edges_rgb
