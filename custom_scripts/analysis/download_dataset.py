#!/usr/bin/env python

from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def download_dataset(repo_id: str, root_dir: str | Path):
    """
    Downloads a LeRobot dataset from Hugging Face Hub to a local directory.
    
    Args:
        repo_id (str): The Hugging Face repository ID (e.g., 'lerobot/pusht').
        root_dir (str | Path): The local root directory to save the dataset.
    """
    root_dir = Path(root_dir)
    print(f"Downloading '{repo_id}' to '{root_dir}'...")
    
    # Instantiating LeRobotDataset automatically downloads the data if not present.
    dataset = LeRobotDataset(repo_id=repo_id, root=root_dir)
    
    print(f"Successfully downloaded '{repo_id}'.")
    print(f"Total episodes: {dataset.meta.total_episodes}")
    print(f"Total frames: {dataset.meta.total_frames}")
    print(f"Data stored in: {dataset.root}")

if __name__ == "__main__":
    # Configuration
    REPO_ID = "pranavnaik98/bimanual_so101_stacking_100"  # Change this to your desired dataset
    ROOT_DIR = f"./data/{REPO_ID}"  # Change this to your desired local path
    
    download_dataset(REPO_ID, ROOT_DIR)
