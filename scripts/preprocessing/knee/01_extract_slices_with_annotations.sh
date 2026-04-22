 #!/bin/bash
set -euo pipefail

# ========= PATHS =========
base_input="/storage/ice-shared/ae8803che/lmkh3"
annotation_path="/home/hice1/lmoukheiber3/SDR/fastmri-plus/Annotations/knee.csv"

# Output directories
train_raw_root="${base_input}/knee_train_labeled_raw"
train_gt_root="${base_input}/knee_train_labeled_gt"

val_raw_root="${base_input}/knee_val_labeled_raw"
val_gt_root="${base_input}/knee_val_labeled_gt"

test_raw_root="${base_input}/knee_test_labeled_raw"
test_gt_root="${base_input}/knee_test_labeled_gt"

mkdir -p "$train_raw_root" "$train_gt_root" \
         "$val_raw_root"   "$val_gt_root" \
         "$test_raw_root"  "$test_gt_root"


# ================= PYTHON SECTION =================
python3 - <<EOF
import os, glob
import pandas as pd
import numpy as np
import h5py
from PIL import Image, ImageDraw

base_input = "$base_input"
df = pd.read_csv("$annotation_path")

summary = {s: {"processed": 0, "skipped": 0} for s in ["train", "val", "test"]}


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
        x0, y0, w, h = map(float, (x0, y0, w, h))
        draw.rectangle(((x0, y0), (x0 + w, y0 + h)), outline="white", width=2)
        draw.text((x0, max(0, y0 - 10)), str(label_txt), fill="white")
    return image_pil


def process_volume(h5_path, raw_root, gt_root, split):
    file_id = os.path.splitext(os.path.basename(h5_path))[0]
    labels = df[df["file"] == file_id]

    if "label" not in labels.columns or labels["label"].notna().sum() == 0:
        summary[split]["skipped"] += 1
        return

    summary[split]["processed"] += 1

    bbox_rows = labels[
        (labels["slice"] > 0) &
        labels[["x", "y", "width", "height"]].notna().all(axis=1)
    ]

    with h5py.File(h5_path, "r") as f:
        vol = f["reconstruction_rss"][:]

    vol = vol[:, ::-1, :]
    raw_dir = os.path.join(raw_root, file_id)
    gt_dir  = os.path.join(gt_root,  file_id)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for i in range(vol.shape[0]):
        slice_num = i + 1
        img = Image.fromarray(normalize_to_uint8(vol[i]))
        img.save(os.path.join(raw_dir, f"{file_id}_slice_{slice_num:03d}.png"))

        rows = bbox_rows[bbox_rows["slice"].astype(int) == slice_num]
        if not rows.empty:
            gt = draw_bboxes(img.copy(), rows.values.tolist())
            gt.save(os.path.join(gt_dir, f"{file_id}_slice_{slice_num:03d}_gt.png"))


# ---------------- TRAIN (batched) ----------------
for i in range(5):
    d = os.path.join(base_input, f"knee_multicoil_train_batch_{i}", "multicoil_train")
    for h5 in glob.glob(os.path.join(d, "*.h5")):
        process_volume(h5,
                       "$train_raw_root",
                       "$train_gt_root",
                       "train")

# ---------------- VAL (single dir) ----------------
val_dir = os.path.join(base_input, "knee_multicoil_val", "multicoil_val")
for h5 in glob.glob(os.path.join(val_dir, "*.h5")):
    process_volume(h5,
                   "$val_raw_root",
                   "$val_gt_root",
                   "val")

# ---------------- TEST (single dir) ----------------
test_dir = os.path.join(base_input, "knee_multicoil_test", "multicoil_test")
for h5 in glob.glob(os.path.join(test_dir, "*.h5")):
    process_volume(h5,
                   "$test_raw_root",
                   "$test_gt_root",
                   "test")


print("\n==============================")
print("FINAL SUMMARY")
print("==============================")
for k, v in summary.items():
    print(f"{k.upper():5} → processed: {v['processed']}, skipped: {v['skipped']}")
EOF


# ========= FILESYSTEM COUNTS =========
echo
echo "=============================="
echo "FILESYSTEM COUNTS (KNEE)"
echo "=============================="
echo "TRAIN RAW: $(find "$train_raw_root" -type f -name '*.png' | wc -l)"
echo "TRAIN GT : $(find "$train_gt_root"  -type f -name '*_gt.png' | wc -l)"
echo "VAL   RAW: $(find "$val_raw_root"   -type f -name '*.png' | wc -l)"
echo "VAL   GT : $(find "$val_gt_root"    -type f -name '*_gt.png' | wc -l)"
echo "TEST  RAW: $(find "$test_raw_root"  -type f -name '*.png' | wc -l)"
echo "TEST  GT : $(find "$test_gt_root"   -type f -name '*_gt.png' | wc -l)"


"""
==============================
FINAL SUMMARY
==============================
TRAIN → processed: 819, skipped: 154
VAL   → processed: 155, skipped: 44
TEST  → processed: 0, skipped: 118

==============================
FILESYSTEM COUNTS (KNEE)
==============================
TRAIN RAW: 29423
TRAIN GT : 8057
VAL   RAW: 5585
VAL   GT : 1524
TEST  RAW: 0"""