#!/usr/bin/env python
"""
Comprehensive dataset analysis across all parquet files.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path("/home/kinisi/Documents/main/lerobot-bimanual/custom_scripts/analysis/data/pranavnaik98/bimanual_so101_stacking_100")

print("="*80)
print("COMPREHENSIVE DATASET ANALYSIS")
print("="*80)

# Analyze all data parquet files
data_dir = DATASET_PATH / "data" / "chunk-000"
data_files = sorted(data_dir.glob("*.parquet"))

print(f"\n📁 Data Files Found: {len(data_files)}")
total_frames = 0
for file in data_files:
    df = pd.read_parquet(file)
    total_frames += len(df)
    print(f"  - {file.name}: {len(df)} frames")

print(f"\n📊 Total Frames: {total_frames}")

# Analyze all episode parquet files
episodes_dir = DATASET_PATH / "meta" / "episodes" / "chunk-000"
episodes_files = sorted(episodes_dir.glob("*.parquet"))

print(f"\n📁 Episode Files Found: {len(episodes_files)}")
all_episodes = []
for file in episodes_files:
    df = pd.read_parquet(file)
    all_episodes.append(df)
    print(f"  - {file.name}: {len(df)} episodes")

# Concatenate all episodes
episodes_df = pd.concat(all_episodes, ignore_index=True)
print(f"\n📊 Total Episodes: {len(episodes_df)}")

# Show episode distribution
print(f"\n📋 Episode Distribution:")
print(f"  - Episode indices: {episodes_df['episode_index'].min()} to {episodes_df['episode_index'].max()}")
print(f"  - Frames per episode (min): {episodes_df['length'].min()}")
print(f"  - Frames per episode (max): {episodes_df['length'].max()}")
print(f"  - Frames per episode (mean): {episodes_df['length'].mean():.1f}")

# Verify frame count
expected_frames = episodes_df['length'].sum()
print(f"\n✅ Verification:")
print(f"  - Sum of episode lengths: {expected_frames}")
print(f"  - Actual frames in data files: {total_frames}")
print(f"  - Match: {expected_frames == total_frames}")

# Show unique tasks
print(f"\n📝 Tasks:")
unique_tasks = set()
for tasks in episodes_df['tasks']:
    for task in tasks:
        unique_tasks.add(task)
print(f"  - {list(unique_tasks)}")

# Check video metadata
print(f"\n🎥 Video Info:")
video_cols_top = [col for col in episodes_df.columns if col.startswith('videos/observation.images.top/')]
video_cols_front = [col for col in episodes_df.columns if col.startswith('videos/observation.images.front/')]
print(f"  - Top camera columns: {len(video_cols_top)}")
print(f"  - Front camera columns: {len(video_cols_front)}")

# Show first and last few episodes
print(f"\n📄 First 3 episodes:")
print(episodes_df[['episode_index', 'length', 'dataset_from_index', 'dataset_to_index']].head(3).to_string())

print(f"\n📄 Last 3 episodes:")
print(episodes_df[['episode_index', 'length', 'dataset_from_index', 'dataset_to_index']].tail(3).to_string())
