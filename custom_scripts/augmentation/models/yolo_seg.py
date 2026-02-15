"""YOLO segmentation model wrapper."""

from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def _cuda_works() -> bool:
    """Check if CUDA kernels actually execute (guards against arch mismatch)."""
    try:
        _ = torch.zeros(1, device="cuda") + 1
        return True
    except RuntimeError:
        return False


class YOLOSegmenter:
    """YOLO-based instance segmentation producing color-coded masks."""
    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 512,
        retina_masks: bool = False,
        max_det: int = 8,
        device: str = "cuda",
    ):
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.retina_masks = retina_masks
        self.max_det = max_det

        self.use_gpu = device == "cuda" and torch.cuda.is_available() and _cuda_works()
        self.device = 0 if self.use_gpu else "cpu"
        self.half = self.use_gpu

    def segment(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Generate color-coded segmentation mask.
        Args:
            rgb_frame: RGB image (H, W, 3), uint8
        Returns:
            Mask (H, W, 3), uint8. Black background, colored objects.
        """
        # Convert RGB to BGR as YOLO/OpenCV expects BGR
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        results = self.model.predict(
            bgr_frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            verbose=False,
            retina_masks=self.retina_masks,
            max_det=self.max_det,
        )[0]

        h, w = rgb_frame.shape[:2]
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        if results.masks is None:
            return mask

        colors = self._get_colors(len(results.names))

        for seg_mask, cls_id in zip(
            results.masks.data.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
        ):
            seg_resized = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            color = colors[int(cls_id)]
            mask[seg_resized > 0.5] = color

        return mask

    @staticmethod
    def _get_colors(num_classes: int) -> dict:
        """Generate distinct colors via HSV color space."""
        colors = {}
        for idx in range(num_classes):
            hue = int(180 * idx / num_classes)
            hsv = np.uint8([[[hue, 255, 255]]])
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
            colors[idx] = tuple(rgb.tolist())
        return colors
