#!/bin/bash
set -euo pipefail

# ========= PATHS =========
base_input="/storage/ice-shared/ae8803che/lmkh3"
annotation_path="/home/hice1/lmoukheiber3/SDR/fastmri-plus/Annotations/brain.csv"

# Output directories (TRAIN + VAL only)
train_raw_root="${base_input}/train_labeled_raw"
train_gt_root="${base_input}/train_labeled_gt"

val_raw_root="${base_input}/val_labeled_raw"
val_gt_root="${base_input}/val_labeled_gt"

mkdir -p "$train_raw_root" "$train_gt_root" \
         "$val_raw_root"   "$val_gt_root"


# ================= PYTHON SECTION =================
python3 - <<EOF
import os, glob
import pandas as pd
import numpy as np
import h5py
from PIL import Image, ImageDraw

base_input      = "$base_input"
annotation_path = "$annotation_path"

train_raw_root  = "$train_raw_root"
train_gt_root   = "$train_gt_root"

val_raw_root    = "$val_raw_root"
val_gt_root     = "$val_gt_root"

df = pd.read_csv(annotation_path)

# Summary counters
summary = {
    "train": {"processed": 0, "skipped": 0},
    "val":   {"processed": 0, "skipped": 0}
}


def normalize_to_uint8(arr):
    arr = np.asarray(arr)
    maxval = float(arr.max()) if arr.size else 0.0
    if maxval > 0:
        arr = (np.maximum(arr, 0) / maxval) * 255.0
    else:
        arr = np.zeros_like(arr)
    return arr.astype(np.uint8)


def draw_bboxes(image_pil, labels_list):
    draw = ImageDraw.Draw(image_pil)
    for row in labels_list:
        _, _, _, x0, y0, w, h, label_txt = row
        x0, y0, w, h = float(x0), float(y0), float(w), float(h)
        x1, y1 = x0 + w, y0 + h
        draw.rectangle(((x0, y0), (x1, y1)), outline="white", width=2)
        draw.text((x0, max(0, y0 - 10)), str(label_txt), fill="white")
    return image_pil


def process_volume(h5_path, labels, raw_root, gt_root, split):
    file_id = os.path.splitext(os.path.basename(h5_path))[0]
    print(f"\n=== [{split}] Processing {file_id} ===")

    # Skip if no label column
    if "label" not in labels.columns:
        print(" → No 'label' column → skip.")
        summary[split]["skipped"] += 1
        return None

    # Skip if no labels
    labeled_rows = labels[labels["label"].notna()]
    if labeled_rows.empty:
        print(" → No labels → skip.")
        summary[split]["skipped"] += 1
        return None

    summary[split]["processed"] += 1

    # Bbox rows only
    bbox_rows = labeled_rows[
        (labeled_rows["slice"] > 0) &
        labeled_rows[["x", "y", "width", "height"]].notna().all(axis=1)
    ].copy()

    # Load volume
    with h5py.File(h5_path, "r") as f:
        vol = f["reconstruction_rss"][:]

    vol = vol[:, ::-1, :]  # flip for your format
    num_slices = vol.shape[0]

    raw_dir = os.path.join(raw_root, file_id)
    gt_dir  = os.path.join(gt_root,  file_id)

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    raw_count = 0
    gt_count  = 0

    # Save ALL slices (because volume has labels)
    for slice_idx in range(num_slices):
        slice_num = slice_idx + 1

        arr = normalize_to_uint8(vol[slice_idx])
        img_raw = Image.fromarray(arr)

        raw_path = os.path.join(raw_dir, f"{file_id}_slice_{slice_num:03d}.png")
        img_raw.save(raw_path)
        raw_count += 1

        # GT for bbox slices
        rows_this_slice = bbox_rows[bbox_rows["slice"].astype(int) == slice_num]
        if not rows_this_slice.empty:
            img_gt = draw_bboxes(img_raw.copy(), rows_this_slice.values.tolist())
            gt_path = os.path.join(gt_dir, f"{file_id}_slice_{slice_num:03d}_gt.png")
            img_gt.save(gt_path)
            gt_count += 1

    print(f" → RAW saved: {raw_count}")
    print(f" → GT saved : {gt_count}")


def run_split(split, batch_pattern, subdir, raw_root, gt_root, n_batches):
    for i in range(n_batches):
        batch = batch_pattern.format(i=i)
        batch_dir = os.path.join(base_input, batch, subdir)

        print("\n==============================")
        print(f"Split: {split.upper()} | Batch: {batch}")
        print("==============================")

        if not os.path.isdir(batch_dir):
            print(f" [WARN] Missing directory: {batch_dir}")
            continue

        for h5_path in sorted(glob.glob(os.path.join(batch_dir, "*.h5"))):
            file_id = os.path.splitext(os.path.basename(h5_path))[0]
            labels = df[df["file"] == file_id]

            try:
                process_volume(h5_path, labels, raw_root, gt_root, split)
            except Exception as e:
                print(f"[ERROR] Skipped {h5_path}: {e}")


# TRAIN batches 0–9
run_split("train",
          "brain_multicoil_train_batch_{i}",
          "multicoil_train",
          raw_root=train_raw_root,
          gt_root=train_gt_root,
          n_batches=10)

# VAL batches 0–2
run_split("val",
          "brain_multicoil_val_batch_{i}",
          "multicoil_val",
          raw_root=val_raw_root,
          gt_root=val_gt_root,
          n_batches=3)


# === PRINT SUMMARY ===
print("\n==============================")
print("FINAL SUMMARY")
print("==============================")
for s in ["train", "val"]:
    print(f"{s.upper():5} → processed: {summary[s]['processed']}, skipped: {summary[s]['skipped']}")
EOF


# ========= FILESYSTEM COUNTS =========

echo
echo "=============================="
echo "FILESYSTEM COUNTS (TRAIN + VAL)"
echo "=============================="
echo "TRAIN RAW: $(find "$train_raw_root" -type f -name '*.png' | wc -l)"
echo "TRAIN GT : $(find "$train_gt_root"  -type f -name '*_gt.png' | wc -l)"
echo "VAL   RAW: $(find "$val_raw_root"   -type f -name '*.png' | wc -l)"
echo "VAL   GT : $(find "$val_gt_root"    -type f -name '*_gt.png' | wc -l)"

"""
==============================
FINAL SUMMARY
==============================
TRAIN → processed: 750, skipped: 3719
VAL   → processed: 247, skipped: 1131

==============================
FILESYSTEM COUNTS (TRAIN + VAL)
==============================
TRAIN RAW: 11786
TRAIN GT : 2414
VAL   RAW: 3880
VAL   GT : 863"""