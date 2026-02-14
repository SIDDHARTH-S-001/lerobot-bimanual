#!/usr/bin/env python
"""
Simple dataset structure inspection - data parquet only.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path("/home/kinisi/Documents/main/lerobot-bimanual/custom_scripts/analysis/data/pranavnaik98/bimanual_so101_stacking_100")

print("="*80)
print("INSPECTING DATA PARQUET FILE")
print("="*80)

data_file = DATASET_PATH / "data" / "chunk-000" / "file-000.parquet"
print(f"\nFile: {data_file}")
df = pd.read_parquet(data_file)

print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n" + "="*80)
print("COLUMN DETAILS")
print("="*80)

for col in df.columns:
    print(f"\n{col}:")
    print(f"  dtype: {df[col].dtype}")
    print(f"  First value: {df[col].iloc[0]}")
    
    if isinstance(df[col].iloc[0], np.ndarray):
        print(f"  Shape of value: {df[col].iloc[0].shape}")
        print(f"  First few elements: {df[col].iloc[0][:min(5, len(df[col].iloc[0]))]}")

print("\n" + "="*80)
print("FIRST 5 ROWS (summary)")
print("="*80)
print(df.head().to_string())
