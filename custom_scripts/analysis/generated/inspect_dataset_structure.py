#!/usr/bin/env python
"""
Comprehensive dataset structure analysis for LeRobot datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def inspect_data_parquet(dataset_path: Path):
    """Inspect the data/chunk-000 parquet files (frame-level data)."""
    print("\n" + "="*80)
    print("DATA PARQUET FILES (Frame-level data)")
    print("="*80)
    
    data_dir = dataset_path / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    
    for pq_file in parquet_files:
        print(f"\n📄 File: {pq_file.name}")
        df = pd.read_parquet(pq_file)
        
        print(f"   Shape: {df.shape} (rows × columns)")
        print(f"   Columns: {list(df.columns)}")
        print(f"\n   First 3 rows:")
        print(df.head(3).to_string(index=True))
        
        # Show data types and sample values
        print(f"\n   Column Details:")
        for col in df.columns:
            dtype = df[col].dtype
            sample_val = df[col].iloc[0]
            print(f"      {col}: dtype={dtype}, sample={sample_val}")
        
        # Check if action or observation.state are arrays
        if 'action' in df.columns:
            print(f"\n   Action shape per row: {df['action'].iloc[0].shape if hasattr(df['action'].iloc[0], 'shape') else 'N/A'}")
            print(f"   Action sample values (first row): {df['action'].iloc[0]}")
        
        if 'observation.state' in df.columns:
            print(f"\n   Observation.state shape per row: {df['observation.state'].iloc[0].shape if hasattr(df['observation.state'].iloc[0], 'shape') else 'N/A'}")
            print(f"   Observation.state sample values (first row): {df['observation.state'].iloc[0]}")
        
        print("\n" + "-"*80)

def inspect_episodes_parquet(dataset_path: Path):
    """Inspect the meta/episodes/chunk-000 parquet files (episode metadata)."""
    print("\n" + "="*80)
    print("EPISODES PARQUET FILES (Episode-level metadata)")
    print("="*80)
    
    episodes_dir = dataset_path / "meta" / "episodes" / "chunk-000"
    parquet_files = sorted(episodes_dir.glob("*.parquet"))
    
    for pq_file in parquet_files:
        print(f"\n📄 File: {pq_file.name}")
        df = pd.read_parquet(pq_file)
        
        print(f"   Shape: {df.shape} (rows × columns)")
        print(f"   Columns ({len(df.columns)} total):")
        
        # Group columns by category
        column_groups = {}
        for col in df.columns:
            if col.startswith('stats/'):
                category = 'stats'
            elif col.startswith('videos/'):
                category = 'videos'
            elif col.startswith('data/'):
                category = 'data'
            elif col.startswith('meta/'):
                category = 'meta'
            else:
                category = 'base'
            
            if category not in column_groups:
                column_groups[category] = []
            column_groups[category].append(col)
        
        for category, cols in sorted(column_groups.items()):
            print(f"\n   [{category.upper()}] columns ({len(cols)}):")
            for col in cols[:5]:  # Show first 5
                print(f"      - {col}")
            if len(cols) > 5:
                print(f"      ... and {len(cols) - 5} more")
        
        print(f"\n   Full DataFrame:")
        # Show all rows since there are only 8 episodes
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.to_string(index=True))
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        
        print("\n" + "-"*80)

def inspect_with_lerobot_api(dataset_path: Path):
    """Use LeRobot API to show dataset structure."""
    print("\n" + "="*80)
    print("LEROBOT DATASET API INSPECTION")
    print("="*80)
    
    repo_id = "pranavnaik98/bimanual_so101_stacking_100"
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path.parent.parent)
    
    print(f"\n📊 Dataset: {dataset.repo_id}")
    print(f"   Total episodes: {dataset.meta.total_episodes}")
    print(f"   Total frames: {dataset.meta.total_frames}")
    print(f"   FPS: {dataset.meta.fps}")
    print(f"   Robot type: {dataset.meta.robot_type}")
    
    print(f"\n📐 Features:")
    for key, value in dataset.meta.features.items():
        print(f"   - {key}:")
        for k, v in value.items():
            print(f"      {k}: {v}")
    
    print(f"\n🎬 Sample from first episode:")
    sample = dataset[0]
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n📈 Episode info (first episode):")
    ep0 = dataset.meta.episodes[0]
    for key, value in ep0.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    DATASET_PATH = Path("/home/kinisi/Documents/main/lerobot-bimanual/custom_scripts/analysis/data/pranavnaik98/bimanual_so101_stacking_100")
    
    print("Starting comprehensive dataset structure analysis...")
    print(f"Dataset path: {DATASET_PATH}")
    
    inspect_data_parquet(DATASET_PATH)
    inspect_episodes_parquet(DATASET_PATH)
    inspect_with_lerobot_api(DATASET_PATH)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
