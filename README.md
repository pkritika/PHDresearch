# Wildfire Segmentation using SAM2

This project applies Meta's Segment Anything Model 2 (SAM2) to segment wildfires in infrared videos from the FireSentry benchmark dataset.

## About

Wildfire detection and tracking is important for understanding fire behavior and monitoring fire spread. This implementation uses SAM2's video segmentation capabilities to automatically identify fire regions in thermal infrared footage captured by drones.

## Dataset

Uses the **FireSentry-Benchmark-Dataset** (KDD 2026):
- Infrared thermal videos of real wildfires
- Ground truth fire masks for evaluation
- 5 geographic regions
- Environmental and vegetation data

Download dataset:
```bash
git clone https://github.com/Munan222/FireSentry-Benchmark-Dataset.git
```

## Setup

```bash
# Clone repository
git clone https://github.com/pkritika/PHDresearch.git
cd PHDresearch

# Create environment
python3 -m venv firediff_env
source firediff_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download SAM2 model (~856MB)
mkdir -p checkpoints
curl -L -o checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Usage

Process a single video:
```bash
python3 firesentry_sam2_segmentation.py \
  --video "FireSentry-Benchmark-Dataset/Region A/Infrared Videos/video_001.mp4" \
  --output results \
  --prompt-method threshold
```

Process multiple videos:
```bash
python3 batch_segment_firesentry.py \
  --dataset-dir FireSentry-Benchmark-Dataset \
  --regions "Region A" \
  --output results \
  --prompt-method threshold
```

Evaluate results:
```bash
python3 evaluate_segmentation.py \
  --pred-dir results \
  --gt-dir FireSentry-Benchmark-Dataset \
  --region "Region A"
```

## How It Works

1. Extracts frames from infrared video
2. Identifies bright regions (fire) using threshold detection
3. Uses SAM2 to segment and track fire across frames
4. Generates fire masks as output

## Requirements

- Python 3.8+
- PyTorch 2.5+
- GPU recommended (works on CPU but slower)

## References

- [SAM2 Paper](https://arxiv.org/abs/2408.00714)
- [FireSentry Dataset](https://arxiv.org/abs/2512.03369)
