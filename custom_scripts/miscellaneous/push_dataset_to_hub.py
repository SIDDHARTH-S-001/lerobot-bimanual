import argparse
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Push a LeRobot dataset to the hub.")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID for the dataset (e.g., 'kinisi/kr1_push_box').",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory for the repository (e.g., 'data/kinisi/kr1_push_box').",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Tolerance in seconds (default: 0.1).",
    )

    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root, tolerance_s=args.tolerance)
    dataset.push_to_hub(private=True)


if __name__ == "__main__":
    main()