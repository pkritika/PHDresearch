#!/usr/bin/env python3
"""
Calculate Accuracy, IoU, and Commission Error for each region from existing results
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd

def load_mask_video_frames(video_path):
    """Load all frames from a mask video"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    cap.release()
    return np.array(frames)

def load_mask_images(mask_dir):
    """Load mask images from directory"""
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    masks = []

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masks.append(mask)

    return np.array(masks)

def compute_metrics(pred_masks, gt_masks):
    """
    Compute segmentation metrics:
    - Accuracy: (TP + TN) / (TP + TN + FP + FN)
    - IoU: TP / (TP + FP + FN)
    - Commission Error: FP / (TP + FP)
    - Omission Error: FN / (TP + FN)
    """
    # Flatten masks to 1D arrays
    pred_flat = (pred_masks > 127).astype(np.uint8).flatten()
    gt_flat = (gt_masks > 127).astype(np.uint8).flatten()

    # Calculate confusion matrix components
    TP = np.sum((pred_flat == 1) & (gt_flat == 1))  # True Positives
    TN = np.sum((pred_flat == 0) & (gt_flat == 0))  # True Negatives
    FP = np.sum((pred_flat == 1) & (gt_flat == 0))  # False Positives
    FN = np.sum((pred_flat == 0) & (gt_flat == 1))  # False Negatives

    # Accuracy
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0

    # IoU
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    # Commission Error (False Positive Rate among predictions)
    commission_error = FP / (TP + FP) if (TP + FP) > 0 else 0

    # Omission Error (False Negative Rate among ground truth)
    omission_error = FN / (TP + FN) if (TP + FN) > 0 else 0

    return {
        'accuracy': accuracy,
        'iou': iou,
        'commission_error': commission_error,
        'omission_error': omission_error,
        'TP': int(TP),
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN)
    }

def evaluate_video(pred_masks_dir, gt_video_path, video_name):
    """Evaluate a single video"""
    if not os.path.exists(pred_masks_dir):
        print(f"Warning: Mask directory not found: {pred_masks_dir}")
        return None

    if not os.path.exists(gt_video_path):
        print(f"Warning: Ground truth video not found: {gt_video_path}")
        return None

    # Load masks
    try:
        pred_masks = load_mask_images(pred_masks_dir)
        gt_masks = load_mask_video_frames(gt_video_path)

        # Ensure same number of frames
        min_frames = min(len(pred_masks), len(gt_masks))
        pred_masks = pred_masks[:min_frames]
        gt_masks = gt_masks[:min_frames]

        # Compute metrics
        metrics = compute_metrics(pred_masks, gt_masks)
        metrics['video_name'] = video_name

        return metrics
    except Exception as e:
        print(f"Error processing {video_name}: {e}")
        return None

def evaluate_region(output_dir, dataset_dir, region):
    """Evaluate all videos in a region"""
    region_output_dir = Path(output_dir) / region
    gt_video_dir = Path(dataset_dir) / region / "Fire Mask Videos"

    if not region_output_dir.exists():
        print(f"Output directory not found: {region_output_dir}")
        return None

    if not gt_video_dir.exists():
        print(f"Ground truth directory not found: {gt_video_dir}")
        return None

    # Get all video directories
    video_dirs = sorted([d for d in os.listdir(region_output_dir) if os.path.isdir(region_output_dir / d)])

    results = []
    for video_dir in tqdm(video_dirs, desc=f"Evaluating {region}"):
        video_name = video_dir
        pred_masks_dir = region_output_dir / video_dir / "masks"
        gt_video_path = gt_video_dir / f"{video_name}.mp4"

        metrics = evaluate_video(pred_masks_dir, gt_video_path, video_name)
        if metrics:
            results.append(metrics)

    return results

def main():
    # Hardcoded paths based on your structure
    output_dir = "result/output"
    dataset_dir = "FireSentry-Benchmark-Dataset"
    regions = ['Region A', 'Region B', 'Region C', 'Region D', 'Region E']
    output_file = 'region_metrics_summary.csv'

    all_results = []

    for region in regions:
        print(f"\n{'='*60}")
        print(f"Evaluating {region}")
        print(f"{'='*60}")

        results = evaluate_region(output_dir, dataset_dir, region)
        if results:
            # Add region column
            for r in results:
                r['region'] = region
            all_results.extend(results)

            # Calculate regional averages
            df = pd.DataFrame(results)
            print(f"\n{region} Summary:")
            print(f"  Videos processed: {len(results)}")
            print(f"  Accuracy:         {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
            print(f"  IoU:              {df['iou'].mean():.4f} ± {df['iou'].std():.4f}")
            print(f"  Commission Error: {df['commission_error'].mean():.4f} ± {df['commission_error'].std():.4f}")
            print(f"  Omission Error:   {df['omission_error'].mean():.4f} ± {df['omission_error'].std():.4f}")

    # Save all results
    if all_results:
        df_all = pd.DataFrame(all_results)
        # Reorder columns
        cols = ['region', 'video_name', 'accuracy', 'iou', 'commission_error', 'omission_error', 'TP', 'TN', 'FP', 'FN']
        df_all = df_all[cols]
        df_all.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to {output_file}")
        print(f"{'='*60}")

        # Overall summary by region
        print("\n" + "="*60)
        print("OVERALL SUMMARY BY REGION")
        print("="*60)
        summary = df_all.groupby('region')[['accuracy', 'iou', 'commission_error', 'omission_error']].agg(['mean', 'std'])
        print(summary)

        # Save summary
        summary.to_csv('region_metrics_summary_stats.csv')
        print(f"\nSummary statistics saved to region_metrics_summary_stats.csv")

if __name__ == '__main__':
    main()
