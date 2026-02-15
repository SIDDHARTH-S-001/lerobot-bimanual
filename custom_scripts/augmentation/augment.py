"""
Dataset Augmentation Pipeline - Main Entry Point

Augments a LeRobot dataset with visual modalities (segmentation, edges, depth,
overlays) using a flexible, class-based architecture.

Usage:
    python augment.py --config config.yaml
    python augment.py --config config.yaml --episodes 0 1 2 --dry-run
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import av # pyav
import cv2
import numpy as np
import torch
import yaml

# Ensure project root is in path for LeRobot imports
PROJECT_ROOT = Path(__file__).resolve().parents[2] # original lerobot_bimanual repo dir path
sys.path.insert(0, str(PROJECT_ROOT / "src")) # ~/lerobot_bimanual/src

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from models.depth import DepthEstimator
from models.edges import CannyDetector
from models.yolo_seg import YOLOSegmenter
from processors.overlay import mask_edge_overlay, original_mask_edge_overlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class AugmentationPipeline:
    def __init__(self, config_path: str):
        """Initialize pipeline from config file."""
        self.config_path = config_path
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self._validate_config()

        # Config extraction
        self.dataset_cfg = self.cfg["dataset"]
        self.modalities_cfg = self.cfg["modalities"]
        self.proc_cfg = self.cfg["processing"]
        self.cameras = self.cfg["cameras"]  # Dynamic list from config

        # Determine output name
        self.output_repo_id, self.output_path = self._determine_output_name()
        
        # Determine device
        self.device = self.proc_cfg.get("device", "cuda")
        self.use_gpu = self.device == "cuda" and torch.cuda.is_available() # Models handle internal checks
        
        self.models_loaded = False
        self.segmenters = {}
        self.canny_detectors = {}
        self.depth_model = None

    def _validate_config(self):
        """Basic validation of paths and requirements."""
        src_path = Path(self.cfg["dataset"]["source_path"])
        if not src_path.exists():
            raise FileNotFoundError(f"Source dataset not found: {src_path}")

        # Check model paths
        for cam_key in ["top_camera", "front_camera"]:
            if cam_key in self.cfg["segmentation"]:
                model_path = Path(self.cfg["segmentation"][cam_key]["model_path"])
                if not model_path.exists():
                    raise FileNotFoundError(f"YOLO model not found: {model_path}")

    def _determine_output_name(self) -> tuple[str, Path]:
        """
        Dynamically construct output dataset name based on enabled modalities.
        Logic: {original_name}_{separate_mods}_{overlay_mods}
        """
        original_name = self.dataset_cfg["original_name"]
        root_dir = Path(self.dataset_cfg["root_dir"])

        m_seg = self.modalities_cfg.get("seg", False)
        m_edges = self.modalities_cfg.get("edges", False)
        m_depth = self.modalities_cfg.get("depth", False)
        m_overlays = self.modalities_cfg.get("overlays", False)

        # 1. Fully augmented check
        if m_seg and m_edges and m_depth and m_overlays:
            suffix = "fully_augmented"
        else:
            # 2. Separate modalities string (m, d, e)
            sep_str = ""
            if m_seg:
                sep_str += "m"
            if m_depth:
                sep_str += "d"
            if m_edges:
                sep_str += "e"
            
            # 3. Overlay modalities string
            overlay_str = ""
            if m_overlays:
                overlay_str = "me_overlay"
            
            # Combine parts
            parts = []
            if sep_str:
                parts.append(sep_str)
            if overlay_str:
                parts.append(overlay_str)
            
            suffix = "_".join(parts)

        # Final construction
        if suffix:
            repo_name = f"{original_name}_{suffix}"
        else:
            repo_name = f"{original_name}_copy" # Fallback if nothing enabled

        output_path = root_dir / repo_name
        
        # Construct repo ID (assuming user/repo format from original)
        # We'll use the original source repo ID's user prefix if available
        src_repo_id = self.dataset_cfg["source_repo_id"]
        if "/" in src_repo_id:
            user_prefix = src_repo_id.split("/")[0]
            repo_id = f"{user_prefix}/{repo_name}"
        else:
            repo_id = repo_name

        log.info(f"Dynamic Output Name: {repo_name}")
        log.info(f"Target Repo ID: {repo_id}")
        log.info(f"Target Path: {output_path}")

        return repo_id, output_path

    def _init_models(self):
        """Initialize all required models."""
        log.info("Initializing models...")

        # YOLO Segmenters (per camera)
        if self.modalities_cfg.get("seg", False) or self.modalities_cfg.get("overlays", False):
            for cam in self.cameras:
                # Map observation key (e.g. observation.images.top) to config key (top_camera)
                # Simple logic: check if 'top' or 'front' in name
                if "top" in cam:
                    cfg_key = "top_camera"
                elif "front" in cam:
                    cfg_key = "front_camera"
                else:
                    log.warning(f"Unknown camera {cam}, skipping segmentation init")
                    continue
                
                seg_cfg = self.cfg["segmentation"][cfg_key]
                self.segmenters[cam] = YOLOSegmenter(
                    model_path=seg_cfg["model_path"],
                    conf=seg_cfg["conf"],
                    iou=seg_cfg["iou"],
                    imgsz=seg_cfg["imgsz"],
                    retina_masks=seg_cfg["retina_masks"],
                    max_det=seg_cfg["max_det"],
                    device=self.device,
                )
                log.info(f"  YOLO loaded for {cam}")

        # Depth Model
        if self.modalities_cfg.get("depth", False):
            d_cfg = self.cfg["depth"]
            if d_cfg["type"] == "da_v2":
                model_cfg = d_cfg["da_v2"]
                self.depth_suffix = "depth_da_v2"
            else:
                model_cfg = d_cfg["zoedepth"]
                self.depth_suffix = "depth_zoedepth"
            
            self.depth_model = DepthEstimator(
                model_id=model_cfg["model_id"],
                colormap=model_cfg["colormap"],
                device=self.device,
            )
            log.info(f"  Depth model loaded: {model_cfg['model_id']}")
        else:
            self.depth_suffix = "depth" # Placeholder

        # Canny Detectors (per camera)
        if self.modalities_cfg.get("edges", False) or self.modalities_cfg.get("overlays", False):
            for cam in self.cameras:
                if "top" in cam:
                    cfg_key = "top_camera"
                elif "front" in cam:
                    cfg_key = "front_camera"
                else:
                    continue # Should not happen based on config

                canny_cfg = self.cfg["canny"][cfg_key]
                self.canny_detectors[cam] = CannyDetector(
                    threshold1=canny_cfg["threshold1"],
                    threshold2=canny_cfg["threshold2"],
                    aperture_size=canny_cfg["aperture_size"],
                    sigma=canny_cfg.get("sigma", 1.0),
                    kernel_size=canny_cfg.get("kernel_size", [3, 3]),
                )
                log.info(f"  Canny initialized for {cam}: {canny_cfg}")
        
        self.models_loaded = True

    def _build_features(self, src_features: dict) -> dict:
        """Construct the features dictionary for the new dataset."""
        default_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        features = {}
        
        # Copy non-default features
        for key, ft in src_features.items():
            if key in default_keys:
                continue
            features[key] = ft.copy()
            if ft["dtype"] == "video" and "info" in features[key]:
                del features[key]["info"]

        # Add new modalities
        suffixes = []
        if self.modalities_cfg.get("seg", False):
            suffixes.append("_seg")
        if self.modalities_cfg.get("edges", False):
            suffixes.append("_edges")
        if self.modalities_cfg.get("depth", False):
            suffixes.append(f"_{self.depth_suffix}")
        if self.modalities_cfg.get("overlays", False):
            suffixes.append("_mask_edge_overlay")
            suffixes.append("_original_mask_edge_overlay")

        for cam in self.cameras:
            for s in suffixes:
                new_key = f"{cam}{s}"
                features[new_key] = {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"],
                }
        
        return features

    def get_episode_info(self, src: LeRobotDataset, ep_idx: int) -> tuple[str, int, int]:
        """
        Combined helper to get task string and data indices for an episode.
        Returns: (task_string, from_index, to_index)
        """
        # Get task string
        # We need to find the task index from the dataset item
        # Since map is heavy, we'll access the first frame of the episode via indices logic
        ep = src.meta.episodes[ep_idx]
        from_idx = ep["dataset_from_index"]
        to_idx = ep["dataset_to_index"]
        
        # Read the item to get task_index
        # Note: src.hf_dataset access might be slow if random access, but here we do it once per ep
        item = src.hf_dataset[from_idx]
        task_idx = item["task_index"]
        if hasattr(task_idx, "item"):
            task_idx = task_idx.item()
        
        task_str = src.meta.tasks.iloc[task_idx].name
        
        return task_str, from_idx, to_idx

    def read_frames_pyav(self, src: LeRobotDataset, ep_idx: int, camera: str) -> np.ndarray:
        """Read frames using PyAV directly (bypassing torchvision Conflict)."""
        ep = src.meta.episodes[ep_idx]
        episode_length = ep["length"]
        from_ts = ep[f"videos/{camera}/from_timestamp"]
        tolerance_s = self.proc_cfg.get("tolerance_s", 0.04)

        video_path = src.root / src.meta.get_video_file_path(ep_idx, camera)
        
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        
        target_pts = int(from_ts / stream.time_base)
        container.seek(target_pts, stream=stream)
        
        frames = []
        for frame in container.decode(stream):
            if frame.time < (from_ts - tolerance_s):
                continue
            
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
            
            if len(frames) >= episode_length:
                break
        
        container.close()
        
        if len(frames) < episode_length:
            log.warning(f"Frame mismatch for ep {ep_idx} {camera}: got {len(frames)}, expected {episode_length}")
            # Pad with last frame if needed? Or raise?
            # User's strict requirement suggests we should care, but validation might fail if we don't return enough.
            # We'll fail fast for now.
            raise RuntimeError(f"Read {len(frames)}/{episode_length} frames for {video_path}")
            
        return np.stack(frames) # (T, H, W, 3)

    def process_camera_frames(self, frames_rgb: np.ndarray, camera: str) -> dict[str, list[np.ndarray]]:
        """Process all frames for a single camera view."""
        # Outputs storage
        outs = {}
        # Init lists for enabled modalities
        keys = []
        if self.modalities_cfg.get("seg", False):
            keys.append("seg")
        if self.modalities_cfg.get("edges", False):
            keys.append("edges")
        if self.modalities_cfg.get("depth", False):
            keys.append(self.depth_suffix)
        if self.modalities_cfg.get("overlays", False):
            keys.append("mask_edge_overlay")
            keys.append("original_mask_edge_overlay")
            
        if not keys:
            return {} # Nothing to do

        for k in keys:
            outs[k] = []

        # Models
        segmenter = self.segmenters.get(camera)
        canny = self.canny_detectors.get(camera)
        
        num_frames = len(frames_rgb)
        
        for i in range(num_frames):
            frame = frames_rgb[i]
            
            # 1. Segmentation
            mask = None
            if segmenter:
                mask = segmenter.segment(frame)
                if self.modalities_cfg.get("seg", False):
                    outs["seg"].append(mask)

            # 2. Edges
            edges = None
            if canny:
                edges = canny.detect(frame)
                if self.modalities_cfg.get("edges", False):
                    outs["edges"].append(edges)
            
            # 3. Depth
            if self.modalities_cfg.get("depth", False):
                depth = self.depth_model.estimate(frame)
                outs[self.depth_suffix].append(depth)

            # 4. Overlays
            if self.modalities_cfg.get("overlays", False):
                # Ensure mask/edges exist (they should if logic is correct)
                # If separate seg/edges disabled but overlays enabled, we ran models above but didn't save separate output.
                # So mask/edges variables hold the data needed here.
                if mask is None: # Should have run segmenter
                   if not segmenter: raise RuntimeError("Overlays enabled but no segmenter found")
                   mask = segmenter.segment(frame)
                if edges is None:
                    if not canny: raise RuntimeError("Overlays enabled but no canny found")
                    edges = canny.detect(frame)
                
                outs["mask_edge_overlay"].append(mask_edge_overlay(mask, edges))
                outs["original_mask_edge_overlay"].append(original_mask_edge_overlay(frame, mask, edges))

            # Periodic clear cache
            if self.use_gpu and i > 0 and i % self.proc_cfg.get("cuda_clear_interval", 50) == 0:
                torch.cuda.empty_cache()

            if (i+1) % 100 == 0:
                log.info(f"    Frame {i+1}/{num_frames}")

        return outs

    def run(self, episodes: list[int] | None = None, dry_run: bool = False):
        """Run the full augmentation pipeline."""
        src_id = self.dataset_cfg["source_repo_id"]
        src_path = self.dataset_cfg["source_path"]
        
        log.info(f"Loading source: {src_id}")
        src = LeRobotDataset(repo_id=src_id, root=src_path, video_backend="pyav")
        
        # Determine features
        # We need depth_suffix determined before building features, so init models first if not dry run
        # Or just determine suffix from config without loading.
        # We did suffix setup in _init_models, let's peek config here or just init models.
        if self.modalities_cfg.get("depth", False):
             self.depth_suffix = "depth_da_v2" if self.cfg["depth"]["type"] == "da_v2" else "depth_zoedepth"
        else:
            self.depth_suffix = ""

        aug_features = self._build_features(src.meta.features)
        
        log.info(f"Features: {list(aug_features.keys())}")
        if dry_run:
            log.info("Dry run logic complete. Exiting.")
            return

        # Init models
        self._init_models()

        # Clean output
        if self.output_path.exists():
            import shutil
            log.warning(f"Removing existing output: {self.output_path}")
            shutil.rmtree(self.output_path)

        # Create destination
        log.info(f"Creating dataset at {self.output_path}")
        dst = LeRobotDataset.create(
            repo_id=self.output_repo_id,
            fps=src.fps,
            features=aug_features,
            root=str(self.output_path),
            robot_type=src.meta.robot_type,
            image_writer_threads=self.proc_cfg.get("image_writer_threads", 4),
        )

        ep_list = episodes if episodes is not None else range(src.meta.total_episodes)
        total = len(ep_list)
        
        log.info(f"Processing {total} episodes...")
        
        pipeline_start = time.time()
        
        for i, ep_idx in enumerate(ep_list):
            ep_start = time.time()
            task, from_idx, to_idx = self.get_episode_info(src, ep_idx)
            
            log.info(f"=== Episode {i+1}/{total} (idx={ep_idx}) Task: {task} ===")
            
            # Read source frames
            cam_frames = {}
            for cam in self.cameras:
                log.info(f"  Reading {cam}...")
                cam_frames[cam] = self.read_frames_pyav(src, ep_idx, cam)
            
            # Process
            cam_processed = {}
            for cam in self.cameras:
                log.info(f"  Processing {cam}...")
                cam_processed[cam] = self.process_camera_frames(cam_frames[cam], cam)
            
            # Write frames
            ep_len_actual = len(cam_frames[self.cameras[0]])
            log.info(f"  Writing {ep_len_actual} frames...")
            
            for f_i in range(ep_len_actual):
                frame_data = {"task": task}
                
                # Copy scalar features
                src_item = src.hf_dataset[from_idx + f_i]
                for k, v in src.meta.features.items():
                    if k not in aug_features: continue # Skip if not in new features (e.g. old info)
                    if v["dtype"] in ("video", "image"): continue
                    val = src_item[k]
                    if hasattr(val, "numpy"): val = val.numpy()
                    frame_data[k] = val
                
                # Add video frames (original + augmented)
                for cam in self.cameras:
                    frame_data[cam] = cam_frames[cam][f_i]
                    
                    # Add modalities
                    mods = cam_processed[cam]
                    # Map back to features dict keys
                    if self.modalities_cfg.get("seg", False):
                        frame_data[f"{cam}_seg"] = mods["seg"][f_i]
                    if self.modalities_cfg.get("edges", False):
                        frame_data[f"{cam}_edges"] = mods["edges"][f_i]
                    if self.modalities_cfg.get("depth", False):
                        frame_data[f"{cam}_{self.depth_suffix}"] = mods[self.depth_suffix][f_i]
                    if self.modalities_cfg.get("overlays", False):
                        frame_data[f"{cam}_mask_edge_overlay"] = mods["mask_edge_overlay"][f_i]
                        frame_data[f"{cam}_original_mask_edge_overlay"] = mods["original_mask_edge_overlay"][f_i]
                        
                dst.add_frame(frame_data)
            
            dst.save_episode()
            log.info(f"  Episode done in {time.time()-ep_start:.1f}s")
            
        dst.finalize()
        log.info(f"Pipeline complete. Saved to: {self.output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pipeline = AugmentationPipeline(args.config)
    pipeline.run(episodes=args.episodes, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
