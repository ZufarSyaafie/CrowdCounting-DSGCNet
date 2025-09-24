#!/usr/bin/env python3
"""
Prepare HOLOGY dataset for DSGC-Net training:
1. Create train.txt and test.txt list files
2. Generate density maps from annotation .txt files
"""

import os
import glob
import numpy as np
from pathlib import Path
from PIL import Image
import scipy.spatial
from scipy.ndimage import gaussian_filter
import argparse


def gaussian_filter_density(img, points):
    """Generate density map using adaptive Gaussian kernel based on k-nearest neighbors"""
    img_shape = [img.shape[0], img.shape[1]]
    print(f"Shape: {img_shape}, Points: {len(points)}")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)

    if gt_count == 0:
        return density

    leafsize = 2048
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    distances, _ = tree.query(points, k=min(4, gt_count))

    print("Generating density...")
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.0
        else:
            continue

        if gt_count > 1:
            # Use k-nearest neighbor distances for adaptive sigma
            sigma = (distances[i][1:].sum()) * 0.1 if len(distances[i]) > 1 else 5.0
        else:
            sigma = np.average(np.array(img_shape)) / 2.0 / 2.0

        density += gaussian_filter(pt2d, sigma, mode="constant")
    print("Done.")
    return density


def create_list_files(data_root):
    """Create train.txt and test.txt files for HOLOGY dataset"""
    data_path = Path(data_root)
    train_dir = data_path / "train"
    test_dir = data_path / "test"

    # Create train.txt
    if train_dir.exists():
        train_list = []
        for img_file in sorted(train_dir.glob("*.jpg")):
            txt_file = img_file.with_suffix(".txt")
            if txt_file.exists():
                rel_img = f"train/{img_file.name}"
                rel_txt = f"train/{txt_file.name}"
                train_list.append(f"{rel_img} {rel_txt}")

        with open(data_path / "train.txt", "w") as f:
            f.write("\n".join(train_list) + "\n")
        print(f"Created train.txt with {len(train_list)} entries")

    # Create test.txt (for validation during training)
    if test_dir.exists():
        test_list = []
        for img_file in sorted(test_dir.glob("*.jpg")):
            txt_file = img_file.with_suffix(".txt")
            if txt_file.exists():
                rel_img = f"test/{img_file.name}"
                rel_txt = f"test/{txt_file.name}"
                test_list.append(f"{rel_img} {rel_txt}")

        with open(data_path / "test.txt", "w") as f:
            f.write("\n".join(test_list) + "\n")
        print(f"Created test.txt with {len(test_list)} entries")


def generate_density_maps(data_root):
    """Generate density maps for train and test sets"""
    data_path = Path(data_root)

    for split in ["train", "test"]:
        img_dir = data_path / split
        density_dir = data_path / "gt_density_maps" / split

        if not img_dir.exists():
            continue

        density_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(img_dir.glob("*.jpg"))
        for img_file in img_files:
            txt_file = img_file.with_suffix(".txt")
            npy_file = density_dir / img_file.with_suffix(".npy").name

            if not txt_file.exists():
                print(f"Warning: No annotation file for {img_file}")
                continue

            if npy_file.exists():
                print(f"Skipping existing {npy_file}")
                continue

            # Load image
            img = np.array(Image.open(img_file))

            # Load points from txt file
            points = []
            with open(txt_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        x, y = map(float, line.split())
                        points.append([x, y])

            points = np.array(points)
            print(f"Processing {img_file.name}: {len(points)} points")

            # Generate density map
            density = gaussian_filter_density(img, points)

            # Save density map
            np.save(npy_file, density)
            print(f"Saved density map: {npy_file}")


def main():
    parser = argparse.ArgumentParser("Prepare HOLOGY dataset for DSGC-Net training")
    parser.add_argument(
        "--data_root", required=True, help="Path to HOLOGY dataset root"
    )
    parser.add_argument(
        "--skip_lists", action="store_true", help="Skip creating list files"
    )
    parser.add_argument(
        "--skip_density", action="store_true", help="Skip generating density maps"
    )

    args = parser.parse_args()

    if not args.skip_lists:
        create_list_files(args.data_root)

    if not args.skip_density:
        generate_density_maps(args.data_root)

    print("Data preparation complete!")


if __name__ == "__main__":
    main()
