#!/usr/bin/env python3
"""Prepare the WTBD wind-turbine-blade-defect dataset for Ultralytics YOLO.

This script does not download anything; it operates on a dataset already
extracted on disk, converting PASCAL VOC annotations to YOLO format and building
the train/val/test splits plus a ``data.yaml``.

================================================================================
Where to get the data
================================================================================
WTBD (Wind Turbine Blade Defect) dataset, Figshare:

    DOI: 10.6084/m9.figshare.30210175
    URL: https://doi.org/10.6084/m9.figshare.30210175

The dataset is distributed as an archive on Figshare.

================================================================================
Expected folder layout (under --raw-dir, default ``data/WT blade defect dataset``)
================================================================================
The dataset is PASCAL VOC: one image plus one XML annotation per image, paired
by filename stem (e.g. ``blade_0001.jpg`` <-> ``blade_0001.xml``).

Images and annotations live in sibling folders. The script auto-detects the
common VOC names; any of these layouts are recognised:

    data/raw/
    ├── images/            (or JPEGImages/ , or imgs/)
    │   ├── blade_0001.jpg
    │   └── ...
    └── annotations/       (or Annotations/ , or labels_voc/ , or xml/)
        ├── blade_0001.xml
        └── ...

A flat layout also works (images and .xml mixed in one folder):

    data/raw/
    ├── blade_0001.jpg
    ├── blade_0001.xml
    └── ...

The --images-dir / --annotations-dir options select these folders explicitly.

================================================================================
Output (under --out-dir, default ``data``)
================================================================================
    data/
    ├── images/{train,val,test}/   copied image files
    ├── labels/{train,val,test}/   YOLO .txt labels (class xc yc w h, normalised)
    └── data.yaml                  Ultralytics dataset config (paths + names)

Splitting is by image with a fixed seed, so the same image (and its boxes) never
lands in two splits. The --split-file option instead reproduces the dataset's own
split (its train_val_test_split.txt) for comparability with published results.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Defaults ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
# The extracted WTBD dataset folder, as shipped on Figshare.
DEFAULT_RAW_DIR = ROOT_DIR / "data" / "WT blade defect dataset"
DEFAULT_OUT_DIR = ROOT_DIR / "data"
DEFAULT_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test
EXPECTED_NUM_CLASSES = 6

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
IMAGE_DIR_CANDIDATES = ("images", "JPEGImages", "imgs", "img")
ANNOT_DIR_CANDIDATES = ("annotations", "Annotations", "labels_voc", "xml", "xmls")
# Canonical class order ships in this file inside the dataset folder.
CLASS_DEFINITIONS_FILENAME = "class_definitions.txt"


# --- Dataset discovery ------------------------------------------------------
def _first_existing(base: Path, names: Tuple[str, ...]) -> Optional[Path]:
    for n in names:
        p = base / n
        if p.is_dir():
            return p
    return None


def locate_dirs(
    raw_dir: Path,
    images_dir: Optional[Path],
    annotations_dir: Optional[Path],
) -> Tuple[Path, Path]:
    """Resolve the image and annotation directories.

    Falls back to a flat layout (both = raw_dir) when no sibling subfolders
    are found.
    """
    img = images_dir or _first_existing(raw_dir, IMAGE_DIR_CANDIDATES) or raw_dir
    ann = annotations_dir or _first_existing(raw_dir, ANNOT_DIR_CANDIDATES) or raw_dir
    return img, ann


def find_pairs(images_dir: Path, annotations_dir: Path) -> List[Tuple[Path, Path]]:
    """Pair each .xml annotation with its image by filename stem.

    Returns a sorted list of (image_path, xml_path). XMLs with no matching
    image are skipped with a warning.
    """
    images_by_stem: Dict[str, Path] = {}
    for ext in IMAGE_EXTS:
        for p in images_dir.rglob(f"*{ext}"):
            images_by_stem.setdefault(p.stem, p)

    pairs: List[Tuple[Path, Path]] = []
    missing = 0
    for xml in sorted(annotations_dir.rglob("*.xml")):
        img = images_by_stem.get(xml.stem)
        if img is None:
            missing += 1
            print(f"  warning: no image found for annotation {xml.name}", file=sys.stderr)
            continue
        pairs.append((img, xml))

    if missing:
        print(f"  ({missing} annotation(s) skipped for lack of a matching image)", file=sys.stderr)
    # Deterministic order before any shuffling.
    pairs.sort(key=lambda t: t[0].stem)
    return pairs


# --- VOC parsing ------------------------------------------------------------
class VocObject:
    __slots__ = ("name", "xmin", "ymin", "xmax", "ymax")

    def __init__(self, name: str, xmin: float, ymin: float, xmax: float, ymax: float):
        self.name = name
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax


def parse_voc(xml_path: Path) -> Tuple[Optional[int], Optional[int], List[VocObject]]:
    """Parse a VOC XML file -> (width, height, objects).

    width/height may be None if absent (caller falls back to the image size).
    """
    root = ET.parse(xml_path).getroot()

    width = height = None
    size = root.find("size")
    if size is not None:
        w, h = size.findtext("width"), size.findtext("height")
        width = int(float(w)) if w and w.strip() else None
        height = int(float(h)) if h and h.strip() else None

    objects: List[VocObject] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            continue
        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            print(f"  warning: bad bndbox in {xml_path.name}, skipping object", file=sys.stderr)
            continue
        objects.append(VocObject(name, xmin, ymin, xmax, ymax))
    return width, height, objects


def _image_size(image_path: Path) -> Tuple[int, int]:
    """Read an image's (width, height) without a hard OpenCV dependency."""
    try:
        import cv2  # noqa: PLC0415 — optional, only needed as a fallback

        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except Exception:  # pragma: no cover - best effort fallback
        pass
    raise RuntimeError(
        f"Could not determine size of {image_path.name}: annotation has no <size> "
        f"and the image could not be read. Install opencv-python or fix the XML."
    )


def voc_to_yolo_lines(
    width: int,
    height: int,
    objects: List[VocObject],
    class_to_idx: Dict[str, int],
    xml_name: str,
) -> Tuple[List[str], Counter]:
    """Convert VOC objects to normalised YOLO label lines.

    Returns (lines, per-class instance counter). VOC corners are converted to
    centre/size form, normalised by image dimensions, and clamped to [0, 1].
    """
    lines: List[str] = []
    counts: Counter = Counter()
    for o in objects:
        if o.name not in class_to_idx:
            # Should not happen: classes are discovered from the same XMLs.
            print(f"  warning: unknown class '{o.name}' in {xml_name}, skipping", file=sys.stderr)
            continue
        xc = ((o.xmin + o.xmax) / 2.0) / width
        yc = ((o.ymin + o.ymax) / 2.0) / height
        bw = (o.xmax - o.xmin) / width
        bh = (o.ymax - o.ymin) / height
        # Clamp against off-by-one / out-of-frame boxes.
        xc, yc = min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)
        bw, bh = min(max(bw, 0.0), 1.0), min(max(bh, 0.0), 1.0)
        if bw <= 0 or bh <= 0:
            print(f"  warning: degenerate box for '{o.name}' in {xml_name}, skipping", file=sys.stderr)
            continue
        idx = class_to_idx[o.name]
        lines.append(f"{idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        counts[o.name] += 1
    return lines, counts


# --- Splitting --------------------------------------------------------------
def split_by_image(
    pairs: List[Tuple[Path, Path]], seed: int, ratios: Tuple[float, float, float]
) -> Dict[str, List[Tuple[Path, Path]]]:
    """Split the (image, xml) pairs by image with a fixed seed.

    A single image and all of its boxes stay together in exactly one split.
    """
    items = list(pairs)
    random.Random(seed).shuffle(items)

    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # Test gets the remainder so the three splits sum to n exactly.
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


# Accepted subset labels in the split file, mapped to our canonical names.
_SUBSET_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


def split_from_file(
    pairs: List[Tuple[Path, Path]], split_file: Path
) -> Dict[str, List[Tuple[Path, Path]]]:
    """Assign pairs to splits using the dataset's own split file.

    The file is a CSV with an ``ImageID,Subset`` header (the WTBD
    ``train_val_test_split.txt`` format), e.g. ``0.jpg,val``. Images are matched
    to annotations by filename stem. Entries listed in the file but absent from
    the data are skipped with a warning; images present in the data but not in
    the file are also reported.
    """
    import csv

    # Map stem -> subset from the file.
    stem_to_subset: Dict[str, str] = {}
    with split_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"split file is empty: {split_file}")

    # Detect and skip a header row (non-data first line, e.g. "ImageID,Subset").
    start = 0
    first = [c.strip().lower() for c in rows[0]]
    if first and first[-1] not in _SUBSET_ALIASES and "subset" in first[-1]:
        start = 1
    elif first and first[0] in ("imageid", "image", "filename"):
        start = 1

    bad = 0
    for row in rows[start:]:
        if len(row) < 2 or not row[0].strip():
            continue
        image_id, subset_raw = row[0].strip(), row[-1].strip().lower()
        subset = _SUBSET_ALIASES.get(subset_raw)
        if subset is None:
            bad += 1
            print(f"  warning: unknown subset '{subset_raw}' for {image_id} in split file", file=sys.stderr)
            continue
        stem_to_subset[Path(image_id).stem] = subset
    if bad:
        print(f"  ({bad} split-file row(s) had an unrecognised subset and were skipped)", file=sys.stderr)

    splits: Dict[str, List[Tuple[Path, Path]]] = {"train": [], "val": [], "test": []}
    unassigned = 0
    for image_path, xml_path in pairs:
        subset = stem_to_subset.get(image_path.stem)
        if subset is None:
            unassigned += 1
            print(f"  warning: {image_path.name} not listed in split file, skipping", file=sys.stderr)
            continue
        splits[subset].append((image_path, xml_path))

    listed_missing = len(stem_to_subset) - sum(len(v) for v in splits.values())
    if unassigned:
        print(f"  ({unassigned} image(s) not in the split file were left out)", file=sys.stderr)
    if listed_missing > 0:
        print(f"  ({listed_missing} split-file entry/entries had no matching image+annotation)", file=sys.stderr)
    return splits


# --- Class discovery --------------------------------------------------------
def load_class_definitions(raw_dir: Path) -> Optional[List[str]]:
    """Read the canonical class order from ``class_definitions.txt`` if present.

    The dataset ships one class name per line in its intended index order. This
    gives stable, paper-consistent YOLO indices without hardcoding the names.
    Returns None if the file is absent.
    """
    path = raw_dir / CLASS_DEFINITIONS_FILENAME
    if not path.is_file():
        return None
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return names or None


def discover_classes(pairs: List[Tuple[Path, Path]]) -> List[str]:
    """Collect every distinct VOC class name across all annotations, sorted."""
    names = set()
    for _, xml in pairs:
        _, _, objects = parse_voc(xml)
        for o in objects:
            names.add(o.name)
    return sorted(names)


# --- Writing ----------------------------------------------------------------
def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_split(
    split: str,
    items: List[Tuple[Path, Path]],
    out_dir: Path,
    class_to_idx: Dict[str, int],
) -> Tuple[int, Counter]:
    """Copy images and write YOLO labels for one split.

    Returns (image_count, per-class instance counter).
    """
    img_out = out_dir / "images" / split
    lbl_out = out_dir / "labels" / split
    reset_dir(img_out)
    reset_dir(lbl_out)

    split_counts: Counter = Counter()
    for image_path, xml_path in items:
        width, height, objects = parse_voc(xml_path)
        if not width or not height:
            width, height = _image_size(image_path)

        lines, counts = voc_to_yolo_lines(width, height, objects, class_to_idx, xml_path.name)
        split_counts.update(counts)

        shutil.copy2(image_path, img_out / image_path.name)
        (lbl_out / f"{image_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    return len(items), split_counts


def write_data_yaml(out_dir: Path, class_names: List[str]) -> Path:
    """Write the Ultralytics data.yaml (absolute path + relative split dirs)."""
    yaml_path = out_dir / "data.yaml"
    lines = [
        "# WTBD wind turbine blade defect dataset — Ultralytics config",
        "# Generated by src/prepare_data.py.",
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(class_names)]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


# --- Reporting --------------------------------------------------------------
def print_summary(
    class_names: List[str],
    per_split_images: Dict[str, int],
    per_split_counts: Dict[str, Counter],
) -> None:
    splits = ["train", "val", "test"]
    name_w = max([len("instances")] + [len(c) for c in class_names])
    col_w = 8

    header = f"{'class':<{name_w}}  " + "".join(f"{s:>{col_w}}" for s in splits) + f"{'total':>{col_w}}"
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("Dataset summary")
    print("=" * len(header))

    # Images per split.
    img_cells = [str(per_split_images[s]) for s in splits]
    img_total = sum(per_split_images.values())
    print(f"{'images':<{name_w}}  " + "".join(f"{c:>{col_w}}" for c in img_cells) + f"{img_total:>{col_w}}")
    print(sep)

    # Instances per class per split.
    print(header)
    print(sep)
    col_totals = Counter()
    for name in class_names:
        cells = []
        row_total = 0
        for s in splits:
            v = per_split_counts[s].get(name, 0)
            cells.append(str(v))
            row_total += v
            col_totals[s] += v
        print(f"{name:<{name_w}}  " + "".join(f"{c:>{col_w}}" for c in cells) + f"{row_total:>{col_w}}")

    print(sep)
    grand = sum(col_totals.values())
    tot_cells = [str(col_totals[s]) for s in splits]
    print(f"{'instances':<{name_w}}  " + "".join(f"{c:>{col_w}}" for c in tot_cells) + f"{grand:>{col_w}}")
    print("=" * len(header))


# --- CLI --------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prepare_data.py",
        description=(
            "Convert the WTBD dataset (PASCAL VOC: images + XML) into YOLO format "
            "and build seeded 70/15/15 train/val/test splits BY IMAGE, plus a "
            "data.yaml for Ultralytics."
        ),
        epilog=(
            "Get WTBD from Figshare DOI 10.6084/m9.figshare.30210175 "
            "(https://doi.org/10.6084/m9.figshare.30210175). Extract it yourself "
            "into the --raw-dir; nothing is downloaded automatically. Expected "
            "layout: images/ + annotations/ subfolders (VOC names auto-detected), "
            "or a flat folder of mixed .jpg/.xml. See the module docstring for details."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                   help="Folder holding the extracted raw VOC dataset.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Output root for images/, labels/ and data.yaml.")
    p.add_argument("--images-dir", type=Path, default=None,
                   help="Explicit images folder (overrides auto-detection).")
    p.add_argument("--annotations-dir", type=Path, default=None,
                   help="Explicit annotations folder (overrides auto-detection).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Random seed for the by-image split (ignored with --split-file).")
    p.add_argument("--split-file", type=Path, default=None,
                   help="Use a predefined split instead of resampling. Expects the "
                        "WTBD train_val_test_split.txt format (CSV: ImageID,Subset). "
                        "Reproduces the dataset's official split for paper comparability.")
    p.add_argument("--names", nargs="+", default=None,
                   help="Explicit class names IN ORDER (defines YOLO indices). "
                        "If omitted, names are auto-discovered from the XMLs and sorted.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    raw_dir: Path = args.raw_dir
    if not raw_dir.exists():
        print(
            f"error: raw dataset folder not found: {raw_dir}\n\n"
            "This script does NOT download anything. Get WTBD from Figshare\n"
            "  DOI 10.6084/m9.figshare.30210175\n"
            "  https://doi.org/10.6084/m9.figshare.30210175\n"
            f"and extract it into {raw_dir} (see --help for the expected layout).",
            file=sys.stderr,
        )
        return 1

    images_dir, annotations_dir = locate_dirs(raw_dir, args.images_dir, args.annotations_dir)
    print(f"Images dir:      {images_dir}")
    print(f"Annotations dir: {annotations_dir}")

    pairs = find_pairs(images_dir, annotations_dir)
    if not pairs:
        print(
            f"error: no (image, .xml) pairs found under {raw_dir}.\n"
            "Check that images and XML annotations share filename stems and that\n"
            "the folder layout matches --help.",
            file=sys.stderr,
        )
        return 1
    print(f"Found {len(pairs)} annotated image(s).")

    # Determine class names / index mapping. Priority:
    #   1. explicit --names
    #   2. the dataset's class_definitions.txt (canonical order)
    #   3. auto-discovery from the XML <name> tags (sorted)
    if args.names:
        class_names = list(args.names)
    else:
        class_names = load_class_definitions(raw_dir)
        if class_names:
            print(f"Classes from:    {raw_dir / CLASS_DEFINITIONS_FILENAME}")
        else:
            class_names = discover_classes(pairs)
    if not class_names:
        print("error: no object classes found in the annotations.", file=sys.stderr)
        return 1
    if len(class_names) != EXPECTED_NUM_CLASSES:
        print(
            f"warning: expected {EXPECTED_NUM_CLASSES} classes for WTBD but found "
            f"{len(class_names)}: {class_names}\n"
            "         Verify your annotations, or pass --names to set them explicitly.",
            file=sys.stderr,
        )
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print("Classes:         " + ", ".join(f"{i}={n}" for i, n in enumerate(class_names)))

    # Choose the split source: the dataset's predefined file, or a seeded resample.
    if args.split_file is not None:
        if not args.split_file.exists():
            print(f"error: split file not found: {args.split_file}", file=sys.stderr)
            return 1
        print(f"Split source:    {args.split_file} (predefined)")
        splits = split_from_file(pairs, args.split_file)
    else:
        print(f"Split source:    seeded resample {SPLIT_RATIOS} (seed={args.seed})")
        splits = split_by_image(pairs, args.seed, SPLIT_RATIOS)

    per_split_images: Dict[str, int] = {}
    per_split_counts: Dict[str, Counter] = {}
    for split in ("train", "val", "test"):
        n_imgs, counts = write_split(split, splits[split], args.out_dir, class_to_idx)
        per_split_images[split] = n_imgs
        per_split_counts[split] = counts

    yaml_path = write_data_yaml(args.out_dir, class_names)

    print_summary(class_names, per_split_images, per_split_counts)
    print(f"\nWrote dataset config: {yaml_path}")
    print(f"Images:  {args.out_dir / 'images'}/{{train,val,test}}")
    print(f"Labels:  {args.out_dir / 'labels'}/{{train,val,test}}")
    if args.split_file is not None:
        print(f"Split:   from {args.split_file.name} (official)")
    else:
        print(f"Seed:    {args.seed}  (split is reproducible)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
