#!/usr/bin/env python3
"""Evaluate a fine-tuned YOLO blade-defect model on the TEST split.

Why this script exists separately from Ultralytics' built-in ``val``: for a
*safety-critical* inspection task the headline number is not mAP or accuracy —
it is **per-class recall** (how many real defects of each type we catch) and the
**area under the precision-recall curve (AUPRC)** per class. This script reports
those front-and-centre and produces clean, presentation-ready figures.

It computes everything from raw predictions with explicit IoU matching, so the
numbers and plots do not depend on Ultralytics' internal metric objects.

Outputs (under --out-dir, default ``results/``):
    metrics.json            all numbers (per-class P/R/F1/AUPRC + support)
    pr_curve.png            per-class precision-recall curves with AUPRC
    confusion_matrix.png    detection confusion matrix (incl. background)
    examples/example_*.png  8-10 images: ground truth vs predictions

Usage:
    python src/evaluate.py --weights path/to/best.pt
    python src/evaluate.py --weights best.pt --data data/data.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless; safe for CLI and servers
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT_DIR / "data" / "data.yaml"
DEFAULT_OUT = ROOT_DIR / "results"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Presentation-ready defaults: large fonts, tight layout.
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.titleweight": "bold",
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})


# --- Dataset plumbing -------------------------------------------------------
def parse_data_yaml(data_yaml: Path, split: str) -> Tuple[Path, Path, List[str]]:
    """Return (images_dir, labels_dir, class_names) for the requested split."""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    split_rel = cfg.get(split)
    if split_rel is None:
        raise KeyError(f"split '{split}' not found in {data_yaml} (has: {list(cfg)})")
    images_dir = (root / split_rel).resolve()
    # YOLO convention: labels mirror images, with 'images' -> 'labels'.
    labels_dir = (root / split_rel.replace("images", "labels", 1)).resolve()

    names_field = cfg["names"]
    if isinstance(names_field, dict):
        class_names = [names_field[i] for i in sorted(names_field)]
    else:
        class_names = list(names_field)
    return images_dir, labels_dir, class_names


def list_images(images_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(images_dir.rglob(f"*{ext}"))
    return sorted(files, key=lambda p: p.stem)


def load_gt_boxes(label_path: Path, width: int, height: int) -> np.ndarray:
    """Load YOLO labels -> array [[cls, x1, y1, x2, y2], ...] in pixel coords."""
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, xc, yc, w, h = (float(v) for v in parts[:5])
        x1 = (xc - w / 2) * width
        y1 = (yc - h / 2) * height
        x2 = (xc + w / 2) * width
        y2 = (yc + h / 2) * height
        rows.append([cls, x1, y1, x2, y2])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


# --- Geometry ---------------------------------------------------------------
def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes -> [len(a), len(b)]."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


# --- Prediction gathering ---------------------------------------------------
class ImageResult:
    """Per-image predictions and ground truth, in pixel xyxy."""

    __slots__ = ("path", "width", "height", "gt", "pred_boxes", "pred_cls", "pred_conf")

    def __init__(self, path, width, height, gt, pred_boxes, pred_cls, pred_conf):
        self.path = path
        self.width = width
        self.height = height
        self.gt = gt
        self.pred_boxes = pred_boxes
        self.pred_cls = pred_cls
        self.pred_conf = pred_conf


def gather_results(
    model,
    images: List[Path],
    labels_dir: Path,
    imgsz: int,
    nms_iou: float,
    min_conf: float,
) -> List[ImageResult]:
    """Run the model on every image and pair predictions with ground truth.

    Predictions are gathered at a very low confidence (min_conf) so the full
    precision-recall curve can be reconstructed; thresholding happens later.
    """
    results: List[ImageResult] = []
    for img_path in images:
        pred = model.predict(
            source=str(img_path), imgsz=imgsz, conf=min_conf, iou=nms_iou, verbose=False
        )[0]
        h, w = pred.orig_shape
        boxes = pred.boxes
        if boxes is not None and len(boxes):
            pb = boxes.xyxy.cpu().numpy().astype(np.float32)
            pc = boxes.cls.cpu().numpy().astype(int)
            pcf = boxes.conf.cpu().numpy().astype(np.float32)
        else:
            pb = np.zeros((0, 4), np.float32)
            pc = np.zeros((0,), int)
            pcf = np.zeros((0,), np.float32)
        gt = load_gt_boxes(labels_dir / f"{img_path.stem}.txt", w, h)
        results.append(ImageResult(img_path, w, h, gt, pb, pc, pcf))
    return results


# --- Metrics ----------------------------------------------------------------
def match_class_scores(
    results: List[ImageResult], num_classes: int, match_iou: float
) -> Tuple[Dict[int, List[Tuple[float, int]]], np.ndarray]:
    """Per class, label each prediction as TP(1)/FP(0) by greedy IoU matching.

    Matching is class-aware and per image: highest-confidence predictions claim
    the best unmatched ground-truth box of the same class (IoU >= match_iou).
    Returns (class -> [(conf, tp), ...], total_gt_per_class).
    """
    scores: Dict[int, List[Tuple[float, int]]] = {c: [] for c in range(num_classes)}
    total_gt = np.zeros(num_classes, dtype=int)

    for r in results:
        for c in range(num_classes):
            gt_c = r.gt[r.gt[:, 0] == c][:, 1:5] if len(r.gt) else np.zeros((0, 4), np.float32)
            total_gt[c] += len(gt_c)
            sel = r.pred_cls == c
            pb, pcf = r.pred_boxes[sel], r.pred_conf[sel]
            if len(pb) == 0:
                continue
            order = np.argsort(-pcf)
            pb, pcf = pb[order], pcf[order]
            matched = np.zeros(len(gt_c), dtype=bool)
            ious = iou_matrix(pb, gt_c)
            for i in range(len(pb)):
                tp = 0
                if len(gt_c):
                    j = int(np.argmax(ious[i]))
                    if ious[i, j] >= match_iou and not matched[j]:
                        matched[j] = True
                        tp = 1
                scores[c].append((float(pcf[i]), tp))
    return scores, total_gt


def precision_recall_curve(
    confs: np.ndarray, tps: np.ndarray, total_gt: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (recall, precision_envelope, auprc) for one class.

    Precision is replaced by its monotone upper envelope so the plotted curve is
    clean; AUPRC is the area under that same envelope (all-point interpolation),
    so the reported number matches the figure.
    """
    if total_gt == 0:
        return np.array([0.0]), np.array([0.0]), float("nan")
    if len(confs) == 0:
        return np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.0

    order = np.argsort(-confs)
    tp_cum = np.cumsum(tps[order])
    fp_cum = np.cumsum(1 - tps[order])
    recall = tp_cum / total_gt
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)

    # Monotone (non-increasing) precision envelope.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    auprc = float(np.sum((recall - recall_prev) * envelope))
    return recall, envelope, auprc


def operating_point_metrics(
    scores: Dict[int, List[Tuple[float, int]]], total_gt: np.ndarray, conf_op: float
) -> Dict[int, Dict[str, float]]:
    """Per-class precision/recall/F1 at a fixed confidence threshold."""
    out: Dict[int, Dict[str, float]] = {}
    for c, lst in scores.items():
        if lst:
            arr = np.array(lst, dtype=np.float32)
            keep = arr[:, 0] >= conf_op
            tp = int(arr[keep, 1].sum())
            fp = int(keep.sum() - tp)
        else:
            tp = fp = 0
        fn = int(total_gt[c]) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[c] = {"precision": precision, "recall": recall, "f1": f1,
                  "tp": tp, "fp": fp, "fn": fn, "support": int(total_gt[c])}
    return out


def confusion_matrix(
    results: List[ImageResult], num_classes: int, conf_op: float, match_iou: float
) -> np.ndarray:
    """Detection confusion matrix of shape (nc+1, nc+1).

    Rows = ground truth, cols = prediction; the extra index is 'background'
    (a missed GT, or a false-positive prediction). Matching is class-agnostic
    and greedy by IoU at the operating threshold.
    """
    bg = num_classes
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    for r in results:
        keep = r.pred_conf >= conf_op
        pb, pc = r.pred_boxes[keep], r.pred_cls[keep]
        gt_boxes = r.gt[:, 1:5] if len(r.gt) else np.zeros((0, 4), np.float32)
        gt_cls = r.gt[:, 0].astype(int) if len(r.gt) else np.zeros((0,), int)

        ious = iou_matrix(pb, gt_boxes)
        gt_taken = np.zeros(len(gt_boxes), dtype=bool)
        pred_order = np.argsort(-r.pred_conf[keep]) if len(pb) else np.array([], int)
        pred_matched = np.zeros(len(pb), dtype=bool)
        for i in pred_order:
            if len(gt_boxes) == 0:
                continue
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= match_iou and not gt_taken[j]:
                gt_taken[j] = True
                pred_matched[i] = True
                cm[gt_cls[j], pc[i]] += 1
        # Unmatched ground truth -> predicted background (a miss).
        for j in range(len(gt_boxes)):
            if not gt_taken[j]:
                cm[gt_cls[j], bg] += 1
        # Unmatched predictions -> background predicted as a class (false alarm).
        for i in range(len(pb)):
            if not pred_matched[i]:
                cm[bg, pc[i]] += 1
    return cm


# --- Plotting ---------------------------------------------------------------
def plot_pr_curves(
    per_class_curve: Dict[int, Tuple[np.ndarray, np.ndarray, float]],
    op_metrics: Dict[int, Dict[str, float]],
    class_names: List[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("tab10")
    for c, name in enumerate(class_names):
        recall, precision, auprc = per_class_curve[c]
        if np.isnan(auprc):
            continue
        color = cmap(c % 10)
        ax.plot(recall, precision, lw=2.5, color=color,
                label=f"{name}  (AUPRC={auprc:.3f}, R={op_metrics[c]['recall']:.2f})")
        # Mark the operating point.
        ax.scatter([op_metrics[c]["recall"]], [op_metrics[c]["precision"]],
                   color=color, s=70, edgecolor="black", zorder=5)
    ax.set_xlabel("Recall  (fraction of real defects found)")
    ax.set_ylabel("Precision")
    ax.set_title("Per-class Precision-Recall (TEST split)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.9, title="dot = operating point")
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    labels = class_names + ["background"]
    # Row-normalise to read recall down the diagonal; annotate raw counts.
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = cm / np.clip(row_sums, 1, None)

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row-normalised (recall)", rotation=270, labelpad=20)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Detection confusion matrix (TEST split)")

    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > thresh else "black", fontsize=12)
    fig.savefig(out_path)
    plt.close(fig)


def save_examples(
    results: List[ImageResult],
    class_names: List[str],
    conf_op: float,
    out_dir: Path,
    n_examples: int,
) -> List[str]:
    """Save side-by-side ground-truth vs prediction images."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    chosen = [r for r in results if len(r.gt)][:n_examples]
    saved: List[str] = []

    def draw(ax, img, boxes, classes, confs, title):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
        for k in range(len(boxes)):
            x1, y1, x2, y2 = boxes[k]
            c = int(classes[k])
            color = cmap(c % 10)
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor=color, linewidth=2.2))
            label = class_names[c] if confs is None else f"{class_names[c]} {confs[k]:.2f}"
            ax.text(x1, max(y1 - 4, 0), label, color="white", fontsize=10,
                    bbox=dict(facecolor=color, edgecolor="none", pad=1, alpha=0.85))

    for idx, r in enumerate(chosen, 1):
        img = cv2.cvtColor(cv2.imread(str(r.path)), cv2.COLOR_BGR2RGB)
        keep = r.pred_conf >= conf_op
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        draw(axes[0], img, r.gt[:, 1:5], r.gt[:, 0], None, "Ground truth")
        draw(axes[1], img, r.pred_boxes[keep], r.pred_cls[keep], r.pred_conf[keep],
             f"Predicted (conf ≥ {conf_op:g})")
        fig.suptitle(r.path.name, fontsize=14)
        out_path = out_dir / f"example_{idx:02d}.png"
        fig.savefig(out_path)
        plt.close(fig)
        saved.append(out_path.name)
    return saved


# --- Orchestration ----------------------------------------------------------
def evaluate(
    weights: Path,
    data_yaml: Path,
    split: str,
    out_dir: Path,
    conf_op: float,
    match_iou: float,
    nms_iou: float,
    imgsz: int,
    min_conf: float,
    n_examples: int,
) -> dict:
    from ultralytics import YOLO

    images_dir, labels_dir, class_names = parse_data_yaml(data_yaml, split)
    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"no images found in {images_dir}")
    nc = len(class_names)
    print(f"Evaluating {weights.name} on {len(images)} '{split}' images, {nc} classes.")

    model = YOLO(str(weights))
    results = gather_results(model, images, labels_dir, imgsz, nms_iou, min_conf)

    scores, total_gt = match_class_scores(results, nc, match_iou)
    op = operating_point_metrics(scores, total_gt, conf_op)
    curves = {}
    for c in range(nc):
        arr = np.array(scores[c], dtype=np.float32) if scores[c] else np.zeros((0, 2), np.float32)
        confs = arr[:, 0] if len(arr) else np.zeros((0,), np.float32)
        tps = arr[:, 1] if len(arr) else np.zeros((0,), np.float32)
        curves[c] = precision_recall_curve(confs, tps, int(total_gt[c]))

    cm = confusion_matrix(results, nc, conf_op, match_iou)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_pr_curves(curves, op, class_names, out_dir / "pr_curve.png")
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
    examples = save_examples(results, class_names, conf_op, out_dir / "examples", n_examples)

    # Assemble metrics.json (recall-first).
    per_class = {}
    for c, name in enumerate(class_names):
        per_class[name] = {
            "recall": round(op[c]["recall"], 4),
            "precision": round(op[c]["precision"], 4),
            "f1": round(op[c]["f1"], 4),
            "auprc": round(curves[c][2], 4) if not np.isnan(curves[c][2]) else None,
            "support": op[c]["support"],
            "tp": op[c]["tp"], "fp": op[c]["fp"], "fn": op[c]["fn"],
        }
    present = [c for c in range(nc) if total_gt[c] > 0]
    macro = {
        "recall": round(float(np.mean([op[c]["recall"] for c in present])), 4),
        "precision": round(float(np.mean([op[c]["precision"] for c in present])), 4),
        "f1": round(float(np.mean([op[c]["f1"] for c in present])), 4),
        "mAUPRC": round(float(np.mean([curves[c][2] for c in present])), 4),
    }
    worst = min(present, key=lambda c: op[c]["recall"])
    metrics = {
        "split": split,
        "weights": weights.name,
        "num_images": len(images),
        "conf_threshold": conf_op,
        "match_iou": match_iou,
        "headline": {
            "macro_recall": macro["recall"],
            "mAUPRC": macro["mAUPRC"],
            "worst_class_recall": {class_names[worst]: round(op[worst]["recall"], 4)},
        },
        "per_class": per_class,
        "macro": macro,
        "note": ("Accuracy is intentionally omitted: defect pixels/boxes are a "
                 "small minority, so accuracy is misleading (accuracy paradox). "
                 "Recall and AUPRC are the safety-relevant metrics."),
        "figures": {
            "pr_curve": "pr_curve.png",
            "confusion_matrix": "confusion_matrix.png",
            "examples": [f"examples/{e}" for e in examples],
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_report(metrics, class_names)
    return metrics


def print_report(metrics: dict, class_names: List[str]) -> None:
    pc = metrics["per_class"]
    name_w = max(len("class"), *(len(n) for n in class_names))
    print("\n" + "=" * (name_w + 44))
    print(f"TEST metrics @ conf={metrics['conf_threshold']}  "
          f"(recall-sorted; accuracy deliberately not reported)")
    print("=" * (name_w + 44))
    print(f"{'class':<{name_w}}  {'recall':>8}{'prec':>8}{'F1':>8}{'AUPRC':>8}{'support':>9}")
    print("-" * (name_w + 44))
    for name, m in sorted(pc.items(), key=lambda kv: kv[1]["recall"]):
        auprc = f"{m['auprc']:.3f}" if m["auprc"] is not None else "  -  "
        print(f"{name:<{name_w}}  {m['recall']:>8.3f}{m['precision']:>8.3f}"
              f"{m['f1']:>8.3f}{auprc:>8}{m['support']:>9}")
    print("-" * (name_w + 44))
    ma = metrics["macro"]
    print(f"{'macro':<{name_w}}  {ma['recall']:>8.3f}{ma['precision']:>8.3f}"
          f"{ma['f1']:>8.3f}{ma['mAUPRC']:>8.3f}")
    worst = metrics["headline"]["worst_class_recall"]
    wname, wval = next(iter(worst.items()))
    print(f"\n[safety] weakest class by recall: {wname} = {wval:.3f}  "
          f"<-- the defect type most often MISSED")
    print("=" * (name_w + 44))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Evaluate a fine-tuned YOLO blade-defect model on the test split, "
                    "emphasising per-class recall and AUPRC (not accuracy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", type=Path, required=True, help="Path to fine-tuned .pt weights.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Ultralytics data.yaml.")
    p.add_argument("--split", default="test", help="Dataset split to evaluate.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Where to write metrics/plots.")
    p.add_argument("--conf", type=float, default=0.25, dest="conf_op",
                   help="Confidence threshold for the operating-point P/R/F1 and confusion matrix.")
    p.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for a correct detection.")
    p.add_argument("--nms-iou", type=float, default=0.6, help="NMS IoU used during prediction.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    p.add_argument("--min-conf", type=float, default=0.001,
                   help="Low conf for gathering predictions (defines the PR curve).")
    p.add_argument("--examples", type=int, default=10, help="Number of GT-vs-pred example images.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.weights.exists():
        print(f"error: weights not found: {args.weights}", file=sys.stderr)
        return 1
    if not args.data.exists():
        print(f"error: data.yaml not found: {args.data}", file=sys.stderr)
        return 1
    evaluate(
        weights=args.weights, data_yaml=args.data, split=args.split, out_dir=args.out_dir,
        conf_op=args.conf_op, match_iou=args.match_iou, nms_iou=args.nms_iou,
        imgsz=args.imgsz, min_conf=args.min_conf, n_examples=args.examples,
    )
    print(f"\nWrote results to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
