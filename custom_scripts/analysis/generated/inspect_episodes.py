#!/usr/bin/env python
"""
Simple dataset structure inspection - episodes metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path("/home/kinisi/Documents/main/lerobot-bimanual/custom_scripts/analysis/data/pranavnaik98/bimanual_so101_stacking_100")

print("="*80)
print("INSPECTING EPISODES PARQUET FILE")
print("="*80)

episodes_file = DATASET_PATH / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
print(f"\nFile: {episodes_file}")
df = pd.read_parquet(episodes_file)

print(f"\nShape: {df.shape}")
print(f"Total columns: {len(df.columns)}")

# Group columns by prefix
column_groups = {}
for col in df.columns:
    prefix = col.split('/')[0]
    if prefix not in column_groups:
        column_groups[prefix] = []
    column_groups[prefix].append(col)

print("\nColumn groups:")
for prefix, cols in sorted(column_groups.items()):
    print(f"  {prefix}: {len(cols)} columns")

print("\n" + "="*80)
print("BASE EPISODE METADATA")
print("="*80)

base_cols = [col for col in df.columns if '/' not in col]
print(f"\nBase columns: {base_cols}")
print("\nValues:")
print(df[base_cols].to_string(index=True))

print("\n" + "="*80)
print("DATA CHUNK/FILE INFO")
print("="*80)

data_cols = [col for col in df.columns if col.startswith('data/')]
print(f"\nData columns: {data_cols}")
print("\nValues:")
print(df[data_cols].to_string(index=True))

print("\n" + "="*80)
print("VIDEO METADATA (observation.images.top)")
print("="*80)

video_top_cols = [col for col in df.columns if col.startswith('videos/observation.images.top/')]
print(f"\nVideo top columns: {video_top_cols}")
print("\nValues:")
print(df[video_top_cols].to_string(index=True))

print("\n" + "="*80)
print("VIDEO METADATA (observation.images.front)")
print("="*80)

video_front_cols = [col for col in df.columns if col.startswith('videos/observation.images.front/')]
print(f"\nVideo front columns: {video_front_cols}")
print("\nValues:")
print(df[video_front_cols].to_string(index=True))

print("\n" + "="*80)
print("STATS METADATA (action) - Sample")
print("="*80)

action_stats_cols = [col for col in df.columns if col.startswith('stats/action/')]
print(f"\nAction stats columns ({len(action_stats_cols)} total):")
print(action_stats_cols[:5], "... and more")
print("\nFirst row values for action stats:")
for col in action_stats_cols[:10]:
    val = df[col].iloc[0]
    if isinstance(val, np.ndarray):
        print(f"  {col}: shape={val.shape}, values={val[:3]}...")
    else:
        print(f"  {col}: {val}")
