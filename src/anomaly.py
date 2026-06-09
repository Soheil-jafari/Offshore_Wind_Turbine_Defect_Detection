#!/usr/bin/env python3
"""Unsupervised anomaly detection on wind-turbine-blade crops.

A small convolutional autoencoder is trained to reconstruct crops of *healthy*
blade surface only. Defect labels are never used during training: the model
learns what normal blade texture looks like, and anything it cannot reconstruct
well (high reconstruction error) is flagged as anomalous.

Pipeline:
    1. Extract healthy crops (regions with no defect box) for training, and a
       held-out mix of healthy + defective crops for evaluation.
    2. Train the autoencoder on the healthy crops.
    3. Score every evaluation crop by its mean reconstruction error.
    4. Compare the error distributions of healthy vs defective crops, render a
       few reconstruction-error heatmaps, and measure ROC-AUC / AUPRC for the
       healthy-vs-defect separation.

Outputs (under --out-dir, default ``results/anomaly``):
    autoencoder.pt          trained weights
    metrics.json            counts, error stats, ROC-AUC, AUPRC, operating point
    error_distributions.png healthy vs defect reconstruction-error histograms
    roc_pr_curves.png       ROC and precision-recall curves
    heatmaps.png            input / reconstruction / error heatmap examples

Usage:
    python src/anomaly.py --data data/data.yaml
    python src/anomaly.py --data data/data.yaml --epochs 30 --crop 128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT_DIR / "data" / "data.yaml"
DEFAULT_CROPS = ROOT_DIR / "data" / "anomaly_crops"
DEFAULT_OUT = ROOT_DIR / "results" / "anomaly"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})


# --- Dataset plumbing -------------------------------------------------------
def parse_data_yaml(data_yaml: Path, split: str) -> Tuple[Path, Path]:
    """Return (images_dir, labels_dir) for a split from an Ultralytics yaml."""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_rel = cfg[split]
    images_dir = (root / split_rel).resolve()
    labels_dir = (root / split_rel.replace("images", "labels", 1)).resolve()
    return images_dir, labels_dir


def list_images(images_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(images_dir.rglob(f"*{ext}"))
    return sorted(files, key=lambda p: p.stem)


def load_boxes(label_path: Path, width: int, height: int) -> np.ndarray:
    """Read YOLO labels into pixel xyxy boxes (class column dropped)."""
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, w, h = (float(v) for v in parts[:5])
        rows.append([(xc - w / 2) * width, (yc - h / 2) * height,
                     (xc + w / 2) * width, (yc + h / 2) * height])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


# --- Crop extraction --------------------------------------------------------
def _window_overlaps(win: np.ndarray, boxes: np.ndarray, max_frac: float) -> bool:
    """True if `win` intersects any box by more than max_frac of the window area."""
    if len(boxes) == 0:
        return False
    ix1 = np.maximum(win[0], boxes[:, 0])
    iy1 = np.maximum(win[1], boxes[:, 1])
    ix2 = np.minimum(win[2], boxes[:, 2])
    iy2 = np.minimum(win[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    win_area = (win[2] - win[0]) * (win[3] - win[1])
    return bool(np.any(inter / max(win_area, 1e-9) > max_frac))


def extract_healthy_crops(
    images: List[Path], labels_dir: Path, out_dir: Path, crop: int,
    per_image: int, max_crops: int, rng: np.random.Generator,
) -> int:
    """Save square crops that do not overlap any defect box (normal surface)."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for img_path in images:
        if saved >= max_crops:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = load_boxes(labels_dir / f"{img_path.stem}.txt", w, h)
        if min(h, w) < crop:
            scale = crop / min(h, w)
            img = cv2.resize(img, (max(int(round(w * scale)), crop), max(int(round(h * scale)), crop)))
            boxes = boxes * scale
            h, w = img.shape[:2]
        for _ in range(per_image):
            if saved >= max_crops:
                break
            for _try in range(25):
                x = int(rng.integers(0, w - crop + 1))
                y = int(rng.integers(0, h - crop + 1))
                win = np.array([x, y, x + crop, y + crop], dtype=np.float32)
                if not _window_overlaps(win, boxes, max_frac=0.0):
                    cv2.imwrite(str(out_dir / f"{img_path.stem}_{saved:05d}.jpg"),
                                img[y:y + crop, x:x + crop])
                    saved += 1
                    break
    return saved


def extract_defect_crops(
    images: List[Path], labels_dir: Path, out_dir: Path, crop: int,
    max_crops: int, margin: float = 0.25,
) -> int:
    """Save square crops centred on each defect box, resized to `crop`."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for img_path in images:
        if saved >= max_crops:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = load_boxes(labels_dir / f"{img_path.stem}.txt", w, h)
        for box in boxes:
            if saved >= max_crops:
                break
            bw, bh = box[2] - box[0], box[3] - box[1]
            side = max(bw, bh) * (1 + margin)
            side = float(min(max(side, crop * 0.5), min(h, w)))
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            x1 = int(np.clip(cx - side / 2, 0, w - side))
            y1 = int(np.clip(cy - side / 2, 0, h - side))
            patch = img[y1:y1 + int(side), x1:x1 + int(side)]
            if patch.size == 0:
                continue
            patch = cv2.resize(patch, (crop, crop))
            cv2.imwrite(str(out_dir / f"{img_path.stem}_{saved:05d}.jpg"), patch)
            saved += 1
    return saved


class CropDataset(Dataset):
    """Loads crop images as float tensors in [0, 1], CHW."""

    def __init__(self, crop_dir: Path, crop: int):
        self.paths = list_images(crop_dir)
        self.crop = crop

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        import cv2

        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.crop, self.crop):
            img = cv2.resize(img, (self.crop, self.crop))
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return t


# --- Model ------------------------------------------------------------------
class ConvAutoencoder(nn.Module):
    """Four-stage convolutional autoencoder (input side must be divisible by 16)."""

    def __init__(self, channels: int = 3, base: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 4

        def down(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        def up(cin, cout, last=False):
            layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1)]
            layers += [nn.Sigmoid()] if last else [nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.encoder = nn.Sequential(down(channels, c1), down(c1, c2), down(c2, c3), down(c3, c4))
        self.decoder = nn.Sequential(up(c4, c3), up(c3, c2), up(c2, c1), up(c1, channels, last=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# --- Training & scoring -----------------------------------------------------
def train_autoencoder(
    model: ConvAutoencoder, loader: DataLoader, epochs: int, lr: float, device: str,
) -> List[float]:
    """Optimise pixel-wise MSE reconstruction on the healthy crops."""
    model.to(device).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history: List[float] = []
    for epoch in range(1, epochs + 1):
        running, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimiser.step()
            running += loss.item() * batch.size(0)
            n += batch.size(0)
        epoch_loss = running / max(n, 1)
        history.append(epoch_loss)
        print(f"  epoch {epoch:>3}/{epochs}  mse={epoch_loss:.5f}")
    return history


@torch.no_grad()
def reconstruction_errors(
    model: ConvAutoencoder, dataset: CropDataset, device: str, batch_size: int = 64,
) -> np.ndarray:
    """Mean squared reconstruction error per crop."""
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    errs: List[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        recon = model(batch)
        per_image = ((batch - recon) ** 2).mean(dim=[1, 2, 3])
        errs.append(per_image.cpu().numpy())
    return np.concatenate(errs) if errs else np.zeros((0,), dtype=np.float32)


@torch.no_grad()
def error_map(model: ConvAutoencoder, crop_tensor: torch.Tensor, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (reconstruction HWC, per-pixel error map HW) for one crop."""
    model.to(device).eval()
    x = crop_tensor.unsqueeze(0).to(device)
    recon = model(x)
    err = ((x - recon) ** 2).mean(dim=1)[0].cpu().numpy()
    recon_img = recon[0].cpu().numpy().transpose(1, 2, 0)
    return recon_img, err


# --- Plots ------------------------------------------------------------------
def plot_error_distributions(
    healthy: np.ndarray, defect: np.ndarray, threshold: float, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    lo = float(min(healthy.min(), defect.min()))
    hi = float(max(healthy.max(), defect.max()))
    bins = np.linspace(lo, hi, 40)
    ax.hist(healthy, bins=bins, density=True, alpha=0.6, color="#2c7fb8", label=f"healthy (n={len(healthy)})")
    ax.hist(defect, bins=bins, density=True, alpha=0.6, color="#d95f0e", label=f"defective (n={len(defect)})")
    ax.axvline(threshold, color="black", linestyle="--", lw=2,
               label=f"threshold (healthy 95th pct = {threshold:.4f})")
    ax.set_xlabel("Reconstruction error (mean squared error)")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction error: healthy vs defective crops")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_pr(labels: np.ndarray, scores: np.ndarray, auc: float, ap: float, out_path: Path) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    prec, rec, _ = precision_recall_curve(labels, scores)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].plot(fpr, tpr, lw=3, color="#2c7fb8")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1.5)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC  (AUC = {auc:.3f})")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(rec, prec, lw=3, color="#d95f0e")
    axes[1].set_xlabel("Recall (defects caught)")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall  (AUPRC = {ap:.3f})")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Healthy-vs-defect separation by reconstruction error", fontsize=18, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)


def plot_heatmaps(
    model: ConvAutoencoder, dataset: CropDataset, errors: np.ndarray,
    device: str, out_path: Path, n: int = 4,
) -> None:
    """Show input, reconstruction, and error heatmap for the top-error defect crops."""
    order = np.argsort(-errors)[:n]
    fig, axes = plt.subplots(len(order), 3, figsize=(12, 4 * len(order)))
    if len(order) == 1:
        axes = axes[None, :]
    for row, idx in enumerate(order):
        x = dataset[idx]
        recon, err = error_map(model, x, device)
        inp = x.numpy().transpose(1, 2, 0)
        axes[row, 0].imshow(np.clip(inp, 0, 1))
        axes[row, 0].set_title("Input" if row == 0 else "")
        axes[row, 1].imshow(np.clip(recon, 0, 1))
        axes[row, 1].set_title("Reconstruction" if row == 0 else "")
        axes[row, 2].imshow(np.clip(inp, 0, 1))
        hm = axes[row, 2].imshow(err, cmap="jet", alpha=0.55)
        axes[row, 2].set_title("Error heatmap (defect highlighted)" if row == 0 else "")
        fig.colorbar(hm, ax=axes[row, 2], fraction=0.046, pad=0.04)
        for col in range(3):
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"err={errors[idx]:.4f}", rotation=0, labelpad=40, fontsize=12)
    fig.suptitle("Reconstruction-error heatmaps on defective crops", fontsize=18, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)


# --- Orchestration ----------------------------------------------------------
def run(
    data_yaml: Path, crops_dir: Path, out_dir: Path, crop: int, epochs: int,
    batch: int, lr: float, base: int, per_image: int, max_train: int,
    max_eval: int, seed: int, force: bool,
) -> dict:
    assert crop % 16 == 0, "crop size must be divisible by 16"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_imgs_dir, train_lbl_dir = parse_data_yaml(data_yaml, "train")
    test_imgs_dir, test_lbl_dir = parse_data_yaml(data_yaml, "test")
    train_images = list_images(train_imgs_dir)
    test_images = list_images(test_imgs_dir)

    d_train_healthy = crops_dir / "train_healthy"
    d_test_healthy = crops_dir / "test_healthy"
    d_test_defect = crops_dir / "test_defect"

    def needs(p: Path) -> bool:
        return force or not p.exists() or not list_images(p)

    print(f"Device: {device} | crop {crop}px")
    if needs(d_train_healthy):
        n = extract_healthy_crops(train_images, train_lbl_dir, d_train_healthy, crop, per_image, max_train, rng)
        print(f"Extracted {n} healthy training crops -> {d_train_healthy}")
    if needs(d_test_healthy):
        n = extract_healthy_crops(test_images, test_lbl_dir, d_test_healthy, crop, per_image, max_eval, rng)
        print(f"Extracted {n} healthy eval crops -> {d_test_healthy}")
    if needs(d_test_defect):
        n = extract_defect_crops(test_images, test_lbl_dir, d_test_defect, crop, max_eval)
        print(f"Extracted {n} defective eval crops -> {d_test_defect}")

    train_set = CropDataset(d_train_healthy, crop)
    if len(train_set) == 0:
        raise RuntimeError("no healthy training crops were extracted")
    loader = DataLoader(train_set, batch_size=batch, shuffle=True, drop_last=len(train_set) >= batch)

    print(f"Training autoencoder on {len(train_set)} healthy crops for {epochs} epochs:")
    model = ConvAutoencoder(channels=3, base=base)
    history = train_autoencoder(model, loader, epochs, lr, device)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "autoencoder.pt")

    healthy_set = CropDataset(d_test_healthy, crop)
    defect_set = CropDataset(d_test_defect, crop)
    err_healthy = reconstruction_errors(model, healthy_set, device, batch)
    err_defect = reconstruction_errors(model, defect_set, device, batch)

    labels = np.concatenate([np.zeros(len(err_healthy)), np.ones(len(err_defect))])
    scores = np.concatenate([err_healthy, err_defect])
    from sklearn.metrics import average_precision_score, roc_auc_score

    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    threshold = float(np.percentile(err_healthy, 95))
    detect_rate = float((err_defect > threshold).mean())
    false_pos_rate = float((err_healthy > threshold).mean())

    plot_error_distributions(err_healthy, err_defect, threshold, out_dir / "error_distributions.png")
    plot_roc_pr(labels, scores, auc, ap, out_dir / "roc_pr_curves.png")
    plot_heatmaps(model, defect_set, err_defect, device, out_dir / "heatmaps.png", n=4)

    metrics = {
        "crop_size": crop,
        "epochs": epochs,
        "final_train_mse": round(history[-1], 6) if history else None,
        "counts": {
            "train_healthy": len(train_set),
            "test_healthy": len(healthy_set),
            "test_defect": len(defect_set),
        },
        "reconstruction_error": {
            "healthy_mean": round(float(err_healthy.mean()), 6),
            "healthy_std": round(float(err_healthy.std()), 6),
            "defect_mean": round(float(err_defect.mean()), 6),
            "defect_std": round(float(err_defect.std()), 6),
        },
        "separation": {"roc_auc": round(auc, 4), "auprc": round(ap, 4)},
        "operating_point": {
            "threshold_healthy_p95": round(threshold, 6),
            "defect_detection_rate": round(detect_rate, 4),
            "healthy_false_positive_rate": round(false_pos_rate, 4),
        },
        "note": ("Trained on healthy crops only; defect labels were not used in "
                 "training. Higher reconstruction error indicates deviation from "
                 "normal blade surface."),
        "figures": ["error_distributions.png", "roc_pr_curves.png", "heatmaps.png"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_report(metrics)
    return metrics


def print_report(m: dict) -> None:
    print("\n" + "=" * 56)
    print("Anomaly detection — healthy-vs-defect by reconstruction error")
    print("=" * 56)
    c, e, s, op = m["counts"], m["reconstruction_error"], m["separation"], m["operating_point"]
    print(f"crops: {c['train_healthy']} train healthy | "
          f"{c['test_healthy']} eval healthy | {c['test_defect']} eval defect")
    print(f"mean error  healthy={e['healthy_mean']:.5f}   defect={e['defect_mean']:.5f}")
    print(f"ROC-AUC = {s['roc_auc']:.3f}    AUPRC = {s['auprc']:.3f}")
    print(f"at healthy-95th-pct threshold: defects flagged={op['defect_detection_rate']:.2%}, "
          f"healthy false alarms={op['healthy_false_positive_rate']:.2%}")
    print("=" * 56)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="anomaly.py",
        description="Train a convolutional autoencoder on healthy blade crops and "
                    "score healthy-vs-defect separation by reconstruction error.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Ultralytics data.yaml.")
    p.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS, help="Where extracted crops are stored.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Where weights/metrics/plots are written.")
    p.add_argument("--crop", type=int, default=128, help="Crop size in pixels (divisible by 16).")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    p.add_argument("--batch", type=int, default=64, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    p.add_argument("--base", type=int, default=16, help="Base channel width of the autoencoder.")
    p.add_argument("--per-image", type=int, default=5, help="Healthy crops sampled per image.")
    p.add_argument("--max-train", type=int, default=4000, help="Cap on healthy training crops.")
    p.add_argument("--max-eval", type=int, default=600, help="Cap on each evaluation crop set.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--force", action="store_true", help="Re-extract crops even if present.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data.exists():
        print(f"error: data.yaml not found: {args.data}", file=sys.stderr)
        return 1
    run(
        data_yaml=args.data, crops_dir=args.crops_dir, out_dir=args.out_dir, crop=args.crop,
        epochs=args.epochs, batch=args.batch, lr=args.lr, base=args.base, per_image=args.per_image,
        max_train=args.max_train, max_eval=args.max_eval, seed=args.seed, force=args.force,
    )
    print(f"\nWrote results to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
