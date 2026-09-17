#!/usr/bin/env python3
"""
Batch process FireSentry videos with SAM2 segmentation
"""

import os
import argparse
from pathlib import Path
import subprocess
from tqdm import tqdm

def process_region(region_dir, output_base_dir, checkpoint, prompt_method, keep_frames):
    """Process all videos in a region"""
    region_name = Path(region_dir).name
    infrared_dir = Path(region_dir) / "Infrared Videos"

    if not infrared_dir.exists():
        print(f"Warning: No Infrared Videos directory found in {region_name}")
        return

    # Get all video files
    video_files = sorted(list(infrared_dir.glob("*.mp4")))

    if not video_files:
        print(f"Warning: No video files found in {infrared_dir}")
        return

    print(f"\nProcessing {len(video_files)} videos from {region_name}")
    print("="*60)

    # Create output directory for this region
    region_output_dir = Path(output_base_dir) / region_name
    region_output_dir.mkdir(exist_ok=True, parents=True)

    # Process each video
    for video_file in tqdm(video_files, desc=f"Processing {region_name}"):
        video_name = video_file.stem
        video_output_dir = region_output_dir / video_name

        # Skip if already processed
        if (video_output_dir / "segmentation_mask.mp4").exists():
            print(f"Skipping {video_name} (already processed)")
            continue

        # Build command
        cmd = [
            "python3",
            "firesentry_sam2_segmentation.py",
            "--video", str(video_file),
            "--output", str(video_output_dir),
            "--checkpoint", checkpoint,
            "--prompt-method", prompt_method
        ]

        if keep_frames:
            cmd.append("--keep-frames")

        # Run segmentation
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video_name}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Batch process FireSentry videos")
    parser.add_argument("--dataset-dir", type=str, required=True,
                       help="Path to FireSentry-Benchmark-Dataset directory")
    parser.add_argument("--regions", type=str, nargs="+", default=["Region A"],
                       help="Regions to process (e.g., 'Region A' 'Region B')")
    parser.add_argument("--output", type=str, default="firesentry_segmentation_results",
                       help="Output base directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2.1_hiera_large.pt",
                       help="SAM2 checkpoint path")
    parser.add_argument("--prompt-method", type=str, default="threshold",
                       choices=["center", "grid", "threshold"],
                       help="Method for generating initial prompts")
    parser.add_argument("--keep-frames", action="store_true",
                       help="Keep extracted frames after processing")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of videos to process per region")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("FireSentry Batch Video Segmentation")
    print("="*60)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Regions to process: {args.regions}")
    print(f"Prompt method: {args.prompt_method}")
    print("="*60)

    # Process each region
    for region in args.regions:
        region_dir = dataset_dir / region

        if not region_dir.exists():
            print(f"Warning: Region directory {region_dir} does not exist")
            continue

        process_region(region_dir, output_dir, args.checkpoint, args.prompt_method, args.keep_frames)

    print("\n" + "="*60)
    print("Batch processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
