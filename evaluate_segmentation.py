#!/usr/bin/env python3
"""
Evaluate SAM2 segmentation results against FireSentry ground truth masks
Computes metrics: AUPRC, F1-score, IoU, MSE
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, f1_score

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
    Compute segmentation metrics
    - AUPRC: Area Under Precision-Recall Curve
    - F1: F1 Score
    - IoU: Intersection over Union
    - MSE: Mean Squared Error
    """
    # Flatten masks for metric computation
    pred_flat = pred_masks.flatten() / 255.0  # Normalize to [0, 1]
    gt_flat = (gt_masks.flatten() > 127).astype(float)  # Binary threshold

    # AUPRC
    if len(np.unique(gt_flat)) > 1:  # Need both classes for AUPRC
        precision, recall, _ = precision_recall_curve(gt_flat, pred_flat)
        auprc = auc(recall, precision)
    else:
        auprc = np.nan

    # Convert predictions to binary for other metrics
    pred_binary = (pred_flat > 0.5).astype(float)

    # F1 Score
    f1 = f1_score(gt_flat, pred_binary, zero_division=0)

    # IoU (Intersection over Union)
    intersection = np.logical_and(pred_binary, gt_flat).sum()
    union = np.logical_or(pred_binary, gt_flat).sum()
    iou = intersection / union if union > 0 else 0

    # MSE
    mse = np.mean((pred_flat - gt_flat) ** 2)

    return {
        'AUPRC': auprc,
        'F1': f1,
        'IoU': iou,
        'MSE': mse
    }

def evaluate_single_video(pred_dir, gt_video_path):
    """Evaluate a single video's segmentation"""
    # Load predicted masks
    pred_mask_dir = os.path.join(pred_dir, "masks")
    if not os.path.exists(pred_mask_dir):
        print(f"Error: Predicted mask directory not found: {pred_mask_dir}")
        return None

    pred_masks = load_mask_images(pred_mask_dir)

    # Load ground truth masks
    if not os.path.exists(gt_video_path):
        print(f"Warning: Ground truth video not found: {gt_video_path}")
        return None

    gt_masks = load_mask_video_frames(gt_video_path)

    # Ensure same number of frames
    min_frames = min(len(pred_masks), len(gt_masks))
    if len(pred_masks) != len(gt_masks):
        print(f"Warning: Frame count mismatch. Using first {min_frames} frames.")
        pred_masks = pred_masks[:min_frames]
        gt_masks = gt_masks[:min_frames]

    # Ensure same dimensions
    if pred_masks.shape[1:] != gt_masks.shape[1:]:
        print(f"Resizing ground truth masks to match predictions: {pred_masks.shape[1:]}")
        resized_gt = []
        for gt in gt_masks:
            resized = cv2.resize(gt, (pred_masks.shape[2], pred_masks.shape[1]))
            resized_gt.append(resized)
        gt_masks = np.array(resized_gt)

    # Compute metrics
    metrics = compute_metrics(pred_masks, gt_masks)

    return metrics

def evaluate_region(pred_base_dir, gt_base_dir, region_name):
    """Evaluate all videos in a region"""
    pred_region_dir = Path(pred_base_dir) / region_name
    gt_region_dir = Path(gt_base_dir) / region_name / "Fire Mask Videos"

    if not pred_region_dir.exists():
        print(f"Error: Predicted region directory not found: {pred_region_dir}")
        return None

    if not gt_region_dir.exists():
        print(f"Error: Ground truth region directory not found: {gt_region_dir}")
        return None

    # Get all predicted video directories
    pred_video_dirs = sorted([d for d in pred_region_dir.iterdir() if d.is_dir()])

    all_metrics = []

    print(f"\nEvaluating {len(pred_video_dirs)} videos from {region_name}")
    print("="*60)

    for pred_video_dir in tqdm(pred_video_dirs, desc=f"Evaluating {region_name}"):
        video_name = pred_video_dir.name
        gt_video_path = gt_region_dir / f"{video_name}.mp4"

        metrics = evaluate_single_video(str(pred_video_dir), str(gt_video_path))

        if metrics is not None:
            metrics['video_name'] = video_name
            all_metrics.append(metrics)

    return all_metrics

def print_metrics_summary(all_metrics, region_name):
    """Print summary statistics"""
    if not all_metrics:
        print("No metrics to summarize")
        return

    print(f"\n{'='*60}")
    print(f"Evaluation Results for {region_name}")
    print(f"{'='*60}")
    print(f"Videos evaluated: {len(all_metrics)}")
    print()

    # Compute mean metrics (excluding NaN values)
    metrics_keys = ['AUPRC', 'F1', 'IoU', 'MSE']
    mean_metrics = {}

    for key in metrics_keys:
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        if values:
            mean_metrics[key] = np.mean(values)
            std_metrics = np.std(values)
            print(f"{key:10s}: {mean_metrics[key]:.4f} ± {std_metrics:.4f}")
        else:
            print(f"{key:10s}: N/A")

    print(f"{'='*60}\n")

    # Print per-video results
    print("Per-video results:")
    print(f"{'Video':<15} {'AUPRC':>8} {'F1':>8} {'IoU':>8} {'MSE':>8}")
    print("-" * 60)
    for m in all_metrics:
        auprc = f"{m['AUPRC']:.4f}" if not np.isnan(m['AUPRC']) else "N/A"
        print(f"{m['video_name']:<15} {auprc:>8} {m['F1']:>8.4f} {m['IoU']:>8.4f} {m['MSE']:>8.4f}")

    return mean_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM2 segmentation results")
    parser.add_argument("--pred-dir", type=str, required=True,
                       help="Directory containing predicted segmentation results")
    parser.add_argument("--gt-dir", type=str, required=True,
                       help="Directory containing FireSentry ground truth (dataset root)")
    parser.add_argument("--region", type=str, default="Region A",
                       help="Region to evaluate")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for metrics (CSV format)")

    args = parser.parse_args()

    # Evaluate region
    all_metrics = evaluate_region(args.pred_dir, args.gt_dir, args.region)

    if all_metrics:
        # Print summary
        mean_metrics = print_metrics_summary(all_metrics, args.region)

        # Save to CSV if requested
        if args.output:
            import csv
            with open(args.output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['video_name', 'AUPRC', 'F1', 'IoU', 'MSE'])
                writer.writeheader()
                writer.writerows(all_metrics)
            print(f"Metrics saved to: {args.output}")

if __name__ == "__main__":
    main()
