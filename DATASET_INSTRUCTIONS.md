# Dataset Download Instructions

The FireSentry-Benchmark-Dataset (~532MB) is too large to include in this GitHub repository.

## 📥 How to Download the Dataset

### Option 1: Clone from Official Repository (Recommended)
```bash
# In your project directory
git clone https://github.com/Munan222/FireSentry-Benchmark-Dataset.git
```

### Option 2: Manual Download
1. Visit: https://github.com/Munan222/FireSentry-Benchmark-Dataset
2. Click "Code" → "Download ZIP"
3. Extract to your project directory

## 📁 Expected Structure

After downloading, your directory should look like:

```
FireDiff/
├── firesentry_sam2_segmentation.py
├── batch_segment_firesentry.py
├── evaluate_segmentation.py
├── requirements.txt
└── FireSentry-Benchmark-Dataset/    ← Downloaded dataset
    ├── Region A/
    ├── Region B/
    ├── Region C/
    ├── Region D/
    └── Region E/
```

## ✅ Verify Dataset

```bash
# Check if dataset exists
ls FireSentry-Benchmark-Dataset/Region\ A/

# Should show:
# - Infrared Videos/
# - Fire Mask Videos/
# - Visible Light/
# - Environmental Info/
```

## 📊 Dataset Size

- **Total**: ~536 MB
- **Region A**: ~71 MB
- **All 5 regions**: 70+ videos each
- **Format**: MP4 videos, JPEG images, CSV data
