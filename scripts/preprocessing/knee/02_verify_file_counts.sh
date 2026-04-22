#!/usr/bin/env bash
set -euo pipefail

BASE="/storage/ice-shared/ae8803che/lmkh3"

SPLITS=(train val test)

RAW_SUFFIX="_labeled_raw"
GT_SUFFIX="_labeled_gt"

for split in "${SPLITS[@]}"; do
  RAW_ROOT="${BASE}/knee_${split}${RAW_SUFFIX}"
  GT_ROOT="${BASE}/knee_${split}${GT_SUFFIX}"

  echo "===================="
  echo "SPLIT: ${split^^}"
  echo "RAW_ROOT: ${RAW_ROOT}"
  echo "GT_ROOT : ${GT_ROOT}"
  echo "===================="

  if [[ ! -d "$RAW_ROOT" && ! -d "$GT_ROOT" ]]; then
    echo "  [WARN] missing both roots; did you run the knee extraction?"
    echo
    continue
  fi

  if [[ -d "$RAW_ROOT" ]]; then
    raw_vols=$(find "$RAW_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
    raw_slices=$(find "$RAW_ROOT" -type f -name '*.png' | wc -l)
  else
    raw_vols=0
    raw_slices=0
  fi

  if [[ -d "$GT_ROOT" ]]; then
    gt_vols=$(find "$GT_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
    gt_slices=$(find "$GT_ROOT" -type f -name '*_gt.png' | wc -l)
  else
    gt_vols=0
    gt_slices=0
  fi

  echo "RAW:"
  echo "  volumes=${raw_vols}"
  echo "  slices (pngs)=${raw_slices}"

  echo "GT:"
  echo "  volumes=${gt_vols}"
  echo "  slices (pngs)=${gt_slices}"
  echo
done
"""
====================
SPLIT: TRAIN
RAW_ROOT: /storage/ice-shared/ae8803che/lmkh3/knee_train_labeled_raw
GT_ROOT : /storage/ice-shared/ae8803che/lmkh3/knee_train_labeled_gt
====================
RAW:
  volumes=819
  slices (pngs)=29423
GT:
  volumes=819
  slices (pngs)=8057

====================
SPLIT: VAL
RAW_ROOT: /storage/ice-shared/ae8803che/lmkh3/knee_val_labeled_raw
GT_ROOT : /storage/ice-shared/ae8803che/lmkh3/knee_val_labeled_gt
====================
RAW:
  volumes=155
  slices (pngs)=5585
GT:
  volumes=155
  slices (pngs)=1524

====================
SPLIT: TEST
RAW_ROOT: /storage/ice-shared/ae8803che/lmkh3/knee_test_labeled_raw
GT_ROOT : /storage/ice-shared/ae8803che/lmkh3/knee_test_labeled_gt
====================
RAW:
  volumes=0
  slices (pngs)=0
GT:
  volumes=0
  slices (pngs)=0"""
  