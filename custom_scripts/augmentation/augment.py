"""
Dataset Augmentation Pipeline - Main Entry Point

Augments a LeRobot dataset with visual modalities (segmentation, edges, depth,
overlays) using LeRobotDataset.create() + add_frame() + save_episode() API.

Usage:
    python augment.py --config config.yaml
    python augment.py --config config.yaml --episodes 0 1 2  # test subset
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# Ensure project root is in path for LeRobot imports
PROJECT_ROOT = Path(__file__).resolve().parents[2] # original lerobot_bimanual repo dir path
sys.path.insert(0, str(PROJECT_ROOT / "src")) # ~/lerobot_bimanual/src

import av # pyav
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load and validate the YAML configuration."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Exclusive depth model check
    da_v2 = cfg["modalities"]["depth_da_v2"]
    zoedepth = cfg["modalities"]["depth_zoedepth"]
    if da_v2 and zoedepth:
        raise ValueError(
            "FATAL: Both depth models enabled. "
            "Only ONE depth model can be active at a time. "
            "Set either depth_da_v2 or depth_zoedepth to false."
        )
    if not da_v2 and not zoedepth:
        raise ValueError("No depth model enabled. At least one must be active.")

    # Validate YOLO model paths
    for cam_key in ["top_camera", "front_camera"]:
        model_path = Path(cfg["segmentation"][cam_key]["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

    # Validate source dataset path
    src_path = Path(cfg["dataset"]["source_path"])
    if not src_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_path}")

    return cfg


def get_depth_suffix(cfg: dict) -> str:
    """Return the depth modality suffix based on config."""
    if cfg["modalities"]["depth_da_v2"]:
        return "depth_da_v2"
    return "depth_zoedepth"


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def build_augmented_features(src_features: dict, cameras: list[str], depth_suffix: str) -> dict:
    """
    Build the features dict for the augmented dataset.

    Includes all source features (excluding DEFAULT_FEATURES which are auto-added)
    plus 10 new video features (5 per camera).
    """
    # DEFAULT_FEATURES that LeRobotDatasetMetadata.create() adds automatically
    default_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

    features = {}
    for key, ft in src_features.items():
        if key in default_keys:
            continue
        features[key] = ft.copy()
        # Strip video info metadata from original video features
        # (it will be regenerated during encoding with the correct codec info)
        if ft["dtype"] == "video" and "info" in features[key]:
            del features[key]["info"]

    # Add new video features
    modality_suffixes = [
        "_seg",
        "_edges",
        f"_{depth_suffix}",
        "_mask_edge_overlay",
        "_original_mask_edge_overlay",
    ]
    for camera in cameras:
        for suffix in modality_suffixes:
            new_key = f"{camera}{suffix}"
            features[new_key] = {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            }

    return features


# ---------------------------------------------------------------------------
# Video reading
# ---------------------------------------------------------------------------


def read_episode_frames(
    src: LeRobotDataset,
    ep_idx: int,
    camera: str,
    tolerance_s: float,
) -> np.ndarray:
    """
    Read all frames for one episode from one camera using direct pyav.
    This avoids torchvision.io.VideoReader conflicts with cv2.

    Returns:
        frames: (T, H, W, 3) uint8 RGB numpy array
    """
    ep = src.meta.episodes[ep_idx]
    episode_length = ep["length"]
    from_ts = ep[f"videos/{camera}/from_timestamp"]
    
    video_path = src.root / src.meta.get_video_file_path(ep_idx, camera)
    
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    
    # Seek to start timestamp
    # Convert seconds to stream time base
    target_pts = int(from_ts / stream.time_base)
    container.seek(target_pts, stream=stream)
    
    frames = []
    decoded_count = 0
    
    # Iterate and collect frames
    # Note: seek might land before the target, so we need to discard until we reach it
    # But since we just want sequential frames for the episode, catching the first one close enough is key.
    # Given the tolerance logic in LeRobot, we rely on the stream PTS.
    
    # Simple logic: decode until we have enough frames. 
    # Since episodes are sequential, we might need to skip frames if seek landed too early.
    
    # Actually, let's implement strict timestamp checking similar to decode_video_frames but sequential
    # But `from_ts` is the EXACT timestamp of the first frame of the episode in this file.
    
    for frame in container.decode(stream):
        # Skip if before start time (with slight tolerance)
        if frame.time < (from_ts - tolerance_s):
            continue
            
        # Convert to RGB numpy
        img = frame.to_ndarray(format="rgb24")
        frames.append(img)
        
        if len(frames) >= episode_length:
            break
            
    container.close()
    
    frames_np = np.stack(frames)
    
    # Validation
    if len(frames_np) != episode_length:
        log.warning(
            "Expected %d frames, got %d for ep_idx=%d camera=%s", 
            episode_length, len(frames_np), ep_idx, camera
        )
        # Pad or trim if strictly necessary? 
        # For now, let's raise if mismatch is large, but usually it matches.
        if len(frames_np) < episode_length:
            raise RuntimeError(f"Could not read enough frames from {video_path}")
            
    return frames_np


# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------


def process_frames(
    frames_rgb: np.ndarray,
    segmenter: YOLOSegmenter,
    canny: CannyDetector,
    depth_model: DepthEstimator,
    depth_suffix: str,
    device: str,
    cuda_clear_interval: int,
) -> dict[str, list[np.ndarray]]:
    """
    Process all frames for one camera to generate 5 modalities.

    Returns:
        dict mapping modality suffix to list of (H, W, 3) uint8 frames
    """
    modalities = {
        "seg": [],
        "edges": [],
        depth_suffix: [],
        "mask_edge_overlay": [],
        "original_mask_edge_overlay": [],
    }

    use_gpu = device == "cuda" and torch.cuda.is_available()
    num_frames = len(frames_rgb)

    for idx in range(num_frames):
        frame = frames_rgb[idx]

        # Generate base modalities
        mask = segmenter.segment(frame)
        edges = canny.detect(frame)
        depth = depth_model.estimate(frame)

        # Verify sizes match original
        assert mask.shape == frame.shape, f"Mask size mismatch at frame {idx}"
        assert edges.shape == frame.shape, f"Edges size mismatch at frame {idx}"
        assert depth.shape == frame.shape, f"Depth size mismatch at frame {idx}"

        # Compute overlays
        me_overlay = mask_edge_overlay(mask, edges)
        ome_overlay = original_mask_edge_overlay(frame, mask, edges)

        # Store
        modalities["seg"].append(mask)
        modalities["edges"].append(edges)
        modalities[depth_suffix].append(depth)
        modalities["mask_edge_overlay"].append(me_overlay)
        modalities["original_mask_edge_overlay"].append(ome_overlay)

        # Clear CUDA cache periodically
        if use_gpu and idx > 0 and idx % cuda_clear_interval == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Progress log every 100 frames
        if (idx + 1) % 100 == 0 or (idx + 1) == num_frames:
            log.info("    Frame %d/%d", idx + 1, num_frames)

    return modalities


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def get_episode_task(src: LeRobotDataset, ep_idx: int) -> str:
    """Get the task string for an episode from the source dataset."""
    # Find the first frame of this episode in the hf_dataset
    for i in range(len(src.hf_dataset)):
        item = src.hf_dataset[i]
        ep = item["episode_index"]
        if hasattr(ep, "item"):
            ep = ep.item()
        if ep == ep_idx:
            task_idx = item["task_index"]
            if hasattr(task_idx, "item"):
                task_idx = task_idx.item()
            return src.meta.tasks.iloc[task_idx].name
    raise ValueError(f"No frames found for episode {ep_idx}")


def get_episode_data_slice(src: LeRobotDataset, ep_idx: int) -> tuple[int, int]:
    """Get the (from_index, to_index) for an episode in the hf_dataset."""
    ep = src.meta.episodes[ep_idx]
    from_idx = ep["dataset_from_index"]
    to_idx = ep["dataset_to_index"]
    return from_idx, to_idx


def run_pipeline(cfg: dict, episode_indices: list[int] | None = None):
    """Run the full augmentation pipeline."""
    depth_suffix = get_depth_suffix(cfg)
    cameras = cfg["cameras"]
    device = cfg["processing"]["device"]
    tolerance_s = cfg["processing"]["tolerance_s"]
    cuda_clear_interval = cfg["processing"]["cuda_clear_interval"]

    # ---------------------------------------------------------------
    # Step 1: Load source dataset
    # ---------------------------------------------------------------
    log.info("Loading source dataset: %s", cfg["dataset"]["source_repo_id"])
    src = LeRobotDataset(
        repo_id=cfg["dataset"]["source_repo_id"],
        root=cfg["dataset"]["source_path"],
        video_backend="pyav",
    )
    log.info(
        "Source: %d episodes, %d frames, features: %s",
        src.meta.total_episodes,
        src.meta.total_frames,
        list(src.meta.features.keys()),
    )

    # ---------------------------------------------------------------
    # Step 2: Build augmented features dict
    # ---------------------------------------------------------------
    aug_features = build_augmented_features(src.meta.features, cameras, depth_suffix)
    log.info("Augmented features (%d total):", len(aug_features))
    for k, v in aug_features.items():
        log.info("  %s: dtype=%s, shape=%s", k, v["dtype"], v.get("shape"))

    # ---------------------------------------------------------------
    # Step 3: Create destination dataset
    # ---------------------------------------------------------------
    output_path = Path(cfg["dataset"]["output_path"])
    if output_path.exists():
        import shutil
        log.info("Removing existing output directory: %s", output_path)
        shutil.rmtree(output_path)

    log.info("Creating augmented dataset: %s", cfg["dataset"]["output_repo_id"])
    dst = LeRobotDataset.create(
        repo_id=cfg["dataset"]["output_repo_id"],
        fps=src.fps,
        features=aug_features,
        root=str(output_path),
        robot_type=src.meta.robot_type,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=cfg["processing"]["image_writer_threads"],
    )
    log.info("Destination dataset created at: %s", dst.root)

    # ---------------------------------------------------------------
    # Step 4: Initialize models
    # ---------------------------------------------------------------
    log.info("Initializing models...")

    # Camera -> YOLO segmenter mapping
    segmenters = {}
    for camera in cameras:
        cam_key = "top_camera" if "top" in camera else "front_camera"
        seg_cfg = cfg["segmentation"][cam_key]
        segmenters[camera] = YOLOSegmenter(
            model_path=seg_cfg["model_path"],
            conf=seg_cfg["conf"],
            iou=seg_cfg["iou"],
            imgsz=seg_cfg["imgsz"],
            retina_masks=seg_cfg["retina_masks"],
            max_det=seg_cfg["max_det"],
            device=device,
        )
        log.info("  YOLO segmenter for %s loaded", camera)

    # Depth model (exclusive)
    if cfg["modalities"]["depth_da_v2"]:
        depth_cfg = cfg["depth"]["da_v2"]
    else:
        depth_cfg = cfg["depth"]["zoedepth"]
    depth_model = DepthEstimator(
        model_id=depth_cfg["model_id"],
        colormap=depth_cfg["colormap"],
        device=device,
    )
    log.info("  Depth model loaded: %s", depth_cfg["model_id"])

    # Canny
    canny = CannyDetector(**cfg["canny"])
    log.info("  Canny detector initialized")

    # ---------------------------------------------------------------
    # Step 5: Determine episodes to process
    # ---------------------------------------------------------------
    if episode_indices is not None:
        episodes = episode_indices
    else:
        episodes = list(range(src.meta.total_episodes))

    total_episodes = len(episodes)
    log.info("Processing %d episodes: %s", total_episodes, episodes[:10])

    # ---------------------------------------------------------------
    # Step 6: Process each episode
    # ---------------------------------------------------------------
    pipeline_start = time.time()
    total_frames_processed = 0

    for ep_num, ep_idx in enumerate(episodes):
        ep_start = time.time()
        ep = src.meta.episodes[ep_idx]
        episode_length = ep["length"]

        log.info(
            "=== Episode %d/%d (index=%d, length=%d) ===",
            ep_num + 1, total_episodes, ep_idx, episode_length,
        )

        # Get task string for this episode
        task_str = get_episode_task(src, ep_idx)
        log.info("  Task: %s", task_str)

        # Get source data indices for this episode
        from_idx, to_idx = get_episode_data_slice(src, ep_idx)

        # Read original video frames per camera
        all_camera_frames = {}
        for camera in cameras:
            log.info("  Reading %s frames...", camera)
            all_camera_frames[camera] = read_episode_frames(
                src, ep_idx, camera, tolerance_s
            )
            log.info(
                "    Read %d frames, shape=%s",
                len(all_camera_frames[camera]),
                all_camera_frames[camera].shape,
            )

        # Process frames per camera to generate new modalities
        all_camera_modalities = {}
        for camera in cameras:
            log.info("  Processing %s modalities...", camera)
            all_camera_modalities[camera] = process_frames(
                frames_rgb=all_camera_frames[camera],
                segmenter=segmenters[camera],
                canny=canny,
                depth_model=depth_model,
                depth_suffix=depth_suffix,
                device=device,
                cuda_clear_interval=cuda_clear_interval,
            )

        # Add each frame to the destination dataset
        log.info("  Adding %d frames to dataset...", episode_length)
        for frame_idx in range(episode_length):
            # Build frame dict with all features
            frame = {"task": task_str}

            # Non-video features from source parquet data
            src_item = src.hf_dataset[from_idx + frame_idx]
            for key, ft in src.meta.features.items():
                if ft["dtype"] in ("video", "image"):
                    continue
                if key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
                    continue
                val = src_item[key]
                if hasattr(val, "numpy"):
                    val = val.numpy()
                frame[key] = val

            # Original video frames (as numpy RGB uint8)
            for camera in cameras:
                frame[camera] = all_camera_frames[camera][frame_idx]

            # New modality frames
            for camera in cameras:
                mods = all_camera_modalities[camera]
                frame[f"{camera}_seg"] = mods["seg"][frame_idx]
                frame[f"{camera}_edges"] = mods["edges"][frame_idx]
                frame[f"{camera}_{depth_suffix}"] = mods[depth_suffix][frame_idx]
                frame[f"{camera}_mask_edge_overlay"] = mods["mask_edge_overlay"][frame_idx]
                frame[f"{camera}_original_mask_edge_overlay"] = mods["original_mask_edge_overlay"][frame_idx]

            dst.add_frame(frame)

        # Save episode (writes parquet, encodes videos, updates metadata)
        log.info("  Saving episode (encoding %d video streams)...", len(dst.meta.video_keys))
        dst.save_episode()

        ep_elapsed = time.time() - ep_start
        total_frames_processed += episode_length
        log.info(
            "  Episode %d done in %.1fs (%.1f frames/sec)",
            ep_idx, ep_elapsed, episode_length / ep_elapsed,
        )

        # Clear GPU memory between episodes
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Step 7: Finalize
    # ---------------------------------------------------------------
    log.info("Finalizing dataset...")
    dst.finalize()

    total_elapsed = time.time() - pipeline_start
    log.info(
        "=== PIPELINE COMPLETE ===\n"
        "  Episodes: %d\n"
        "  Total frames: %d\n"
        "  Time: %.1f seconds (%.1f min)\n"
        "  Output: %s",
        total_episodes,
        total_frames_processed,
        total_elapsed,
        total_elapsed / 60,
        dst.root,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Augment LeRobot dataset with visual modalities")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Optional: specific episode indices to process (for testing)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_pipeline(cfg, args.episodes)


if __name__ == "__main__":
    main()
