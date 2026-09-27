#!/usr/bin/env python3
"""
Batch frame extraction: saves EVERY frame of every video in a folder,
one output folder per video.

Usage:
    python extract_frames.py --input videos/ --output frames/

Output:
    frames/
    ├── video01/
    │   ├── 00000.jpg
    │   ├── 00001.jpg
    │   └── ...
    ├── video02/
    │   └── ...
"""

import argparse
from pathlib import Path

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".webm"}


def extract_frames(video_path: Path, out_dir: Path, fmt: str, jpg_quality: int) -> int:
    """Save every frame of one video into out_dir. Returns the number of frames saved."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Could not open {video_path.name}, skipping")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {fps:.2f} | frames reported by metadata: {expected}")

    params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality] if fmt == "jpg" else []

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Zero-padded numeric names (00000.jpg) are also what SAM2 expects
        cv2.imwrite(str(out_dir / f"{count:05d}.{fmt}"), frame, params)
        count += 1
        if count % 500 == 0:
            print(f"    ...{count} frames")

    cap.release()
    return count


def main():
    parser = argparse.ArgumentParser(description="Extract every frame from all videos in a folder")
    parser.add_argument("--input", required=True, help="Folder containing the videos")
    parser.add_argument("--output", default="frames", help="Where to create the per-video folders")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg",
                        help="Image format (png is lossless but much larger)")
    parser.add_argument("--quality", type=int, default=95, help="JPG quality 0-100")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos whose output folder already has frames")
    args = parser.parse_args()

    in_dir, out_root = Path(args.input), Path(args.output)
    videos = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)

    if not videos:
        print(f"No videos found in {in_dir}")
        return

    print(f"Found {len(videos)} videos in {in_dir}\n")
    summary = []

    for i, video in enumerate(videos, 1):
        out_dir = out_root / video.stem  # folder named after the video file
        print(f"[{i}/{len(videos)}] {video.name} -> {out_dir}")

        if args.skip_existing and out_dir.exists() and any(out_dir.iterdir()):
            print("  Already extracted, skipping\n")
            summary.append((video.name, "skipped"))
            continue

        n = extract_frames(video, out_dir, args.format, args.quality)
        print(f"  Saved {n} frames\n")
        summary.append((video.name, n))

    print("=" * 50)
    print("Summary")
    print("=" * 50)
    for name, n in summary:
        print(f"{name:40s} {n}")


if __name__ == "__main__":
    main()