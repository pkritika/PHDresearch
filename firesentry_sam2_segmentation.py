#!/usr/bin/env python3
"""
FireSentry SAM2 Video Segmentation
This script performs fire segmentation on FireSentry infrared videos using SAM2.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor

def setup_sam2(checkpoint_path, model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"):
    """Initialize SAM2 video predictor"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
    return predictor, device

def extract_frames(video_path, output_dir):
    """Extract frames from video for SAM2 processing"""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Extracting {total_frames} frames from video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # SAM2 expects frame filenames to be just numbers (e.g., 00000.jpg, 00001.jpg)
        frame_path = os.path.join(output_dir, f"{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    return frame_count, fps

def get_initial_prompts(first_frame, method="center"):
    """
    Generate initial prompts for fire detection
    Methods:
    - center: Click at center of frame (assumes fire is central)
    - grid: Multiple points in a grid pattern
    - threshold: Use brightness thresholding to find hot spots
    """
    h, w = first_frame.shape[:2]

    if method == "center":
        # Single point at center
        points = np.array([[w//2, h//2]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)  # 1 = foreground

    elif method == "grid":
        # Grid of points in central region
        grid_points = []
        for i in range(3):
            for j in range(3):
                x = int(w * (0.25 + i * 0.25))
                y = int(h * (0.25 + j * 0.25))
                grid_points.append([x, y])
        points = np.array(grid_points, dtype=np.float32)
        labels = np.array([1] * len(grid_points), dtype=np.int32)

    elif method == "threshold":
        # Find bright regions (hot spots in infrared)
        if len(first_frame.shape) == 3:
            gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = first_frame

        # Threshold to find brightest regions
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get centers of largest contours
            points_list = []
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points_list.append([cx, cy])

            if points_list:
                points = np.array(points_list, dtype=np.float32)
                labels = np.array([1] * len(points_list), dtype=np.int32)
            else:
                # Fallback to center
                points = np.array([[w//2, h//2]], dtype=np.float32)
                labels = np.array([1], dtype=np.int32)
        else:
            # Fallback to center
            points = np.array([[w//2, h//2]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)

    return points, labels

def segment_video(predictor, frames_dir, output_dir, prompt_method="threshold"):
    """Perform video segmentation with SAM2"""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize predictor with the frames directory
    inference_state = predictor.init_state(video_path=frames_dir)

    # Load first frame to get prompts
    first_frame_path = os.path.join(frames_dir, "00000.jpg")
    first_frame = cv2.imread(first_frame_path)

    # Get initial prompts
    points, labels = get_initial_prompts(first_frame, method=prompt_method)

    print(f"Using {len(points)} prompt points for segmentation")

    # Add prompts to the first frame (frame_idx=0, obj_id=1)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=labels,
    )

    # Propagate through the video
    print("Propagating segmentation through video...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments

def save_segmentation_results(video_segments, frames_dir, output_dir, original_video_path):
    """Save segmentation masks as images and video"""
    mask_output_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_output_dir, exist_ok=True)

    # Get original video properties
    cap = cv2.VideoCapture(str(original_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Create video writer for overlay
    overlay_video_path = os.path.join(output_dir, "segmentation_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(overlay_video_path, fourcc, fps, (width, height))

    # Create video writer for mask only
    mask_video_path = os.path.join(output_dir, "segmentation_mask.mp4")
    mask_video = cv2.VideoWriter(mask_video_path, fourcc, fps, (width, height), isColor=False)

    print("Saving segmentation results...")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    for frame_idx, frame_file in enumerate(tqdm(frame_files)):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)

        if frame_idx in video_segments:
            # Get mask for this frame
            masks = video_segments[frame_idx]
            if 1 in masks:  # obj_id = 1
                mask = masks[1][0]  # Get first mask

                # Convert boolean mask to uint8
                mask_uint8 = (mask * 255).astype(np.uint8)

                # Save mask as image
                mask_path = os.path.join(mask_output_dir, f"mask_{frame_idx:05d}.png")
                cv2.imwrite(mask_path, mask_uint8)

                # Create overlay
                overlay = frame.copy()
                overlay[mask] = overlay[mask] * 0.5 + np.array([0, 0, 255]) * 0.5  # Red overlay

                # Write to videos
                out_video.write(overlay.astype(np.uint8))
                mask_video.write(mask_uint8)
            else:
                # No mask for this frame
                out_video.write(frame)
                mask_video.write(np.zeros((height, width), dtype=np.uint8))
        else:
            # No mask for this frame
            out_video.write(frame)
            mask_video.write(np.zeros((height, width), dtype=np.uint8))

    out_video.release()
    mask_video.release()

    print(f"Overlay video saved to: {overlay_video_path}")
    print(f"Mask video saved to: {mask_video_path}")
    print(f"Individual masks saved to: {mask_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="FireSentry SAM2 Video Segmentation")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2.1_hiera_large.pt",
                       help="SAM2 checkpoint path")
    parser.add_argument("--prompt-method", type=str, default="threshold",
                       choices=["center", "grid", "threshold"],
                       help="Method for generating initial prompts")
    parser.add_argument("--keep-frames", action="store_true",
                       help="Keep extracted frames after processing")

    args = parser.parse_args()

    # Setup
    video_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*60)
    print("FireSentry SAM2 Video Segmentation")
    print("="*60)
    print(f"Input video: {video_path}")
    print(f"Output directory: {output_dir}")

    # Extract frames
    frames_dir = output_dir / "frames"
    frame_count, fps = extract_frames(video_path, frames_dir)
    print(f"Extracted {frame_count} frames at {fps} FPS")

    # Initialize SAM2
    print("\nInitializing SAM2...")
    predictor, device = setup_sam2(args.checkpoint)

    # Segment video
    print("\nPerforming segmentation...")
    video_segments = segment_video(predictor, str(frames_dir), str(output_dir), args.prompt_method)
    print(f"Segmented {len(video_segments)} frames")

    # Save results
    print("\nSaving results...")
    save_segmentation_results(video_segments, str(frames_dir), str(output_dir), video_path)

    # Cleanup frames if requested
    if not args.keep_frames:
        print("\nCleaning up extracted frames...")
        import shutil
        shutil.rmtree(frames_dir)

    print("\n" + "="*60)
    print("Segmentation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
