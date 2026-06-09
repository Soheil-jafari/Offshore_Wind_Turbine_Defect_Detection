# Wind Turbine Blade Defect Detection

A demo project for detecting surface defects on wind turbine blades using
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics), combined with a
small autoencoder-based anomaly detector.

## Overview

The pipeline has two stages:

1. **Supervised detection.** Fine-tune a COCO-pretrained YOLO model on the
   **WTBD** (Wind Turbine Blade Defect) dataset to localize and classify visible
   blade defects (e.g. cracks, erosion, surface damage).

2. **Anomaly detection.** Train a small convolutional **autoencoder** on crops of
   *healthy* (defect-free) blade regions. At inference time, a high
   reconstruction error flags anomalous crops — catching defect types that were
   under-represented or unseen during supervised training.

## Project structure

```
.
├── data/         # WTBD dataset (git-ignored; download/prepare separately)
├── notebooks/    # Colab training notebooks (train_yolo.ipynb)
├── src/          # Helper scripts (data prep, utilities)
├── results/      # Saved metrics, plots, and evaluation artifacts
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

For training, open `notebooks/train_yolo.ipynb` in Google Colab (T4 GPU runtime
recommended). See [Training](#training) below.

## Dataset

The WTBD dataset is **not** included in this repository (the `data/` directory is
git-ignored). Get it from Figshare — DOI
[10.6084/m9.figshare.30210175](https://doi.org/10.6084/m9.figshare.30210175) —
and extract it under `data/` (PASCAL VOC format: `JPEGImages/` + `Annotations/`).

`src/prepare_data.py` converts the VOC annotations to YOLO format, builds the
train/val/test splits, and writes `data/data.yaml`. It defaults to the
`data/WT blade defect dataset` folder, auto-detects `JPEGImages/` +
`Annotations/` (ignoring the dataset's extra scripts/figures), and reads the
class order from the dataset's `class_definitions.txt` — so no arguments are
needed for the common case:

```bash
# Seeded 70/15/15 resample by image (the default; controlled by --seed, default 42):
python src/prepare_data.py

# Official split — reproduces the dataset's train_val_test_split.txt;
# use this for comparability with published results:
python src/prepare_data.py --split-file "data/WT blade defect dataset/train_val_test_split.txt"
```

Override any default with `--raw-dir`, `--images-dir`, `--annotations-dir`,
`--out-dir`, or `--names` (see `--help`).

Both produce the standard Ultralytics layout (6 classes, 1065 images):

```
data/
├── images/{train,val,test}/   # copied image files
├── labels/{train,val,test}/   # YOLO .txt labels (class xc yc w h, normalised)
└── data.yaml                  # Ultralytics dataset config (paths + class names)
```

## Training

Stage 1 (supervised detection) runs in `notebooks/train_yolo.ipynb` on Google
Colab with a **T4 GPU**. The notebook:

1. Checks the GPU (`nvidia-smi`) and installs Ultralytics.
2. Mounts Google Drive and locates the prepared dataset (see note below).
3. Loads COCO-pretrained `yolov8s.pt`.
4. Runs validation **before** fine-tuning to record a baseline (≈0 mAP — COCO has
   no blade classes — establishing the floor).
5. Fine-tunes for ~50 epochs with early stopping and blade-appropriate
   augmentation (flips, small rotation, scale/translate jitter, HSV
   brightness/contrast, mosaic on), each choice explained in the notebook.
6. Saves the best weights and metrics to Drive.
7. Reports per-class precision, recall, mAP@0.5 and mAP@0.5:0.95.

**Getting the dataset onto Drive:** `prepare_data.py` writes the dataset under
local `data/` (git-ignored), so upload that output — `images/`, `labels/`, and
`data.yaml` — to a Drive folder (default `MyDrive/wtbd_yolo/`; editable in the
notebook). The notebook rewrites the `data.yaml` `path:` line to the Drive
location automatically.

## Anomaly detection (stage 2)

A second, **unsupervised** detector complements the supervised model. A small
convolutional autoencoder (`src/anomaly.py`, notebook
`notebooks/train_anomaly.ipynb`) is trained on crops of **healthy** blade
surface only — it learns what *normal* looks like and uses **no defect labels**
in training. At test time, crops that reconstruct poorly (high error) are flagged
as anomalous, which catches defect types that were rare or unlabelled in the
detection data.

```bash
python src/anomaly.py --data data/data.yaml
```

Outputs to `results/anomaly/`:

- `autoencoder.pt` — trained weights.
- `metrics.json` — crop counts, reconstruction-error statistics, ROC-AUC, AUPRC,
  and an operating point (the healthy 95th-percentile error threshold).
- `error_distributions.png` — healthy vs defective reconstruction-error histograms.
- `roc_pr_curves.png` — ROC and precision-recall curves for the separation.
- `heatmaps.png` — per-pixel error heatmaps localising defects on example crops.

Healthy crops are windows with no overlap with any defect box; defective crops
are centred on labelled defects and used for evaluation only. Run
`python src/anomaly.py --help` for crop size, epochs, and other options.

## Evaluation

Detection quality on the held-out **test** split is reported by
`src/evaluate.py` (and mirrored in the notebook's Step 9). For a safety-critical
inspection task it deliberately **does not report accuracy** — under heavy class
imbalance a "no defect everywhere" model scores high accuracy while catching
nothing (the accuracy paradox). Instead it emphasises **per-class recall** and
**AUPRC**, and also reports precision and F1.

```bash
python src/evaluate.py --weights path/to/yolov8s_wtbd_best.pt --data data/data.yaml
```

Outputs to `results/`:

- `metrics.json` — per-class recall / precision / F1 / AUPRC (+ the weakest
  class by recall, i.e. the defect most often missed).
- `pr_curve.png` — per-class precision-recall curves with AUPRC.
- `confusion_matrix.png` — detection confusion matrix (incl. background).
- `examples/example_*.png` — 8-10 images with predicted vs ground-truth boxes.

Run `python src/evaluate.py --help` for thresholds and other options.
