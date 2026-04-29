# COMP/ENGN 4528 Mini Project
## Single-Image Semantic Segmentation with Augmentation Robustness Analysis

---

## Overview

This project trains a U-Net from scratch to perform semantic segmentation on a single image of the John Curtin School of Medical Research at ANU. Pseudo-labels are generated automatically using CLIPSeg (no manual annotation). Two augmentation strategies (V1 mild, V2 wide) are trained and compared under five types of image distortion.

**Four classes:** sky, building, vegetation, ground

---

## Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies

**On Windows (with NVIDIA GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**On Mac:**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

---

## Files

```
project/
  segmentation_new.ipynb   # Main notebook (all code)
  image_data.jpg           # The single input image
  mask.png                 # Hand-painted ground truth mask (for evaluation only)
  README.md                # This file
  requirements.txt         # Python dependencies
```

---

## How to Run

1. Place `image_data.jpg` and `mask.png` in the same folder as the notebook
2. Open `segmentation_new.ipynb` in VS Code or Jupyter
3. Run all cells from top to bottom in order

The notebook will:
- Generate pseudo-labels using CLIPSeg
- Create a block-based train/test spatial split
- Train V1 (mild augmentation) for 50 epochs
- Train V2 (wide augmentation) for 50 epochs
- Evaluate both models on the held-out test region
- Run robustness experiments across 5 distortion types x 5 severities
- Generate and save all figures

**Expected training time:**
- V1: ~22 minutes (Mac CPU/MPS) or ~5 minutes (NVIDIA GPU)
- V2: ~22 minutes (Mac CPU/MPS) or ~5 minutes (NVIDIA GPU)

---

## Key Results

| Model | Pixel Accuracy | Mean IoU |
|-------|---------------|----------|
| V1 (mild augmentation) | 96.69% | 78.35% |
| V2 (wide augmentation) | 96.91% | 71.35% |

V2 outperforms V1 significantly under severe distortions (up to +64.64% IoU under extreme brightness reduction) but loses performance on vegetation due to colour augmentation destroying colour cues.

---

## Dependencies

Main libraries used:
- torch, torchvision
- transformers (CLIPSeg)
- opencv-python
- numpy, matplotlib
- scikit-learn
- seaborn
- Pillow