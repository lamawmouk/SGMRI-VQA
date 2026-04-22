#!/usr/bin/env bash
set -euo pipefail

BASE="/storage/ice-shared/ae8803che/lmkh3"

TRAIN_RAW="$BASE/train_labeled_raw_by_modality"
VAL_RAW="$BASE/val_labeled_raw_by_modality"

echo "=============================="
echo "CHECK TRAIN vs VAL OVERLAP"
echo "=============================="
echo "TRAIN_RAW: $TRAIN_RAW"
echo "VAL_RAW  : $VAL_RAW"
echo

if [[ ! -d "$TRAIN_RAW" ]]; then
  echo "[ERROR] Missing: $TRAIN_RAW"
  exit 1
fi
if [[ ! -d "$VAL_RAW" ]]; then
  echo "[ERROR] Missing: $VAL_RAW"
  exit 1
fi

echo "------------------------------"
echo "1) UNIQUE VOLUME COUNTS"
echo "------------------------------"
train_vols=$(find "$TRAIN_RAW" -mindepth 2 -maxdepth 2 -type d -printf '%f\n' | sort -u | wc -l)
val_vols=$(find "$VAL_RAW" -mindepth 2 -maxdepth 2 -type d -printf '%f\n' | sort -u | wc -l)
echo "train unique volumes: $train_vols"
echo "val   unique volumes: $val_vols"
echo

echo "------------------------------"
echo "2) OVERLAPPING VOLUMES (FOLDER NAMES)"
echo "------------------------------"
overlap_count=$(
  comm -12 \
    <(find "$TRAIN_RAW" -mindepth 2 -maxdepth 2 -type d -printf '%f\n' | sort -u) \
    <(find "$VAL_RAW"   -mindepth 2 -maxdepth 2 -type d -printf '%f\n' | sort -u) \
  | wc -l
)
echo "overlap volume count: $overlap_count"
echo

if [[ "$overlap_count" -gt 0 ]]; then
  echo "Overlapping volume folder names (showing up to first 200):"
  comm -12 \
    <(find "$TRAIN_RAW" -mindepth 2 -maxdepth 2 -type d -printf '%f\n' | sort -u) \
    <(find "$VAL_RAW"   -mindepth 2 -maxdepth 2 -type d -printf '%f\n' | sort -u) \
  | head -200
  echo
  echo "NOTE: Any overlap here suggests train/val leakage at the volume level."
else
  echo "No overlapping volume folder names found ✔"
fi
echo

echo "------------------------------"
echo "3) OPTIONAL: OVERLAP OF PNG FILENAMES (BASENAMES)"
echo "------------------------------"
png_overlap_count=$(
  comm -12 \
    <(find "$TRAIN_RAW" -type f -name '*.png' -printf '%f\n' | sort -u) \
    <(find "$VAL_RAW"   -type f -name '*.png' -printf '%f\n' | sort -u) \
  | wc -l
)
echo "overlap png-basename count: $png_overlap_count"

if [[ "$png_overlap_count" -gt 0 ]]; then
  echo "Example overlapping PNG basenames (first 50):"
  comm -12 \
    <(find "$TRAIN_RAW" -type f -name '*.png' -printf '%f\n' | sort -u) \
    <(find "$VAL_RAW"   -type f -name '*.png' -printf '%f\n' | sort -u) \

  | head -50
fi


"""==============================
CHECK TRAIN vs VAL OVERLAP
==============================
TRAIN_RAW: /storage/ice-shared/ae8803che/lmkh3/train_labeled_raw_by_modality
VAL_RAW  : /storage/ice-shared/ae8803che/lmkh3/val_labeled_raw_by_modality

------------------------------
1) UNIQUE VOLUME COUNTS
------------------------------
train unique volumes: 750
val   unique volumes: 247

------------------------------
2) OVERLAPPING VOLUMES (FOLDER NAMES)
------------------------------
overlap volume count: 0

No overlapping volume folder names found ✔

------------------------------
3) OPTIONAL: OVERLAP OF PNG FILENAMES (BASENAMES)
------------------------------
overlap png-basename count: 0"""
