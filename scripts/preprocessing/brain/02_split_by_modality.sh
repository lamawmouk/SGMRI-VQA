#!/usr/bin/env bash
set -euo pipefail

BASE="/storage/ice-shared/ae8803che/lmkh3"
SPLITS=(train val)
MODS=(FLAIR T1 T2)

RAW_NEW_SUFFIX="_labeled_raw_by_modality"
GT_NEW_SUFFIX="_labeled_gt_by_modality"

for split in "${SPLITS[@]}"; do
  RAW_ROOT="${BASE}/${split}${RAW_NEW_SUFFIX}"
  GT_ROOT="${BASE}/${split}${GT_NEW_SUFFIX}"

  echo "===================="
  echo "SPLIT: ${split}"
  echo "RAW_ROOT: ${RAW_ROOT}"
  echo "GT_ROOT : ${GT_ROOT}"
  echo "===================="

  if [[ ! -d "$RAW_ROOT" && ! -d "$GT_ROOT" ]]; then
    echo "  [WARN] missing both roots; did you run the sorting script?"
    echo
    continue
  fi

  total_raw_vols=0
  total_raw_slices=0
  total_gt_vols=0
  total_gt_slices=0

  for mod in "${MODS[@]}"; do
    raw_dir="${RAW_ROOT}/${mod}"
    gt_dir="${GT_ROOT}/${mod}"

    if [[ -d "$raw_dir" ]]; then
      raw_vols=$(find "$raw_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
      raw_slices=$(find "$raw_dir" -type f -name '*.png' | wc -l)
    else
      raw_vols=0
      raw_slices=0
    fi

    if [[ -d "$gt_dir" ]]; then
      gt_vols=$(find "$gt_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
      gt_slices=$(find "$gt_dir" -type f -name '*.png' | wc -l)
    else
      gt_vols=0
      gt_slices=0
    fi

    total_raw_vols=$((total_raw_vols + raw_vols))
    total_raw_slices=$((total_raw_slices + raw_slices))
    total_gt_vols=$((total_gt_vols + gt_vols))
    total_gt_slices=$((total_gt_slices + gt_slices))

    echo "${mod}:"
    echo "  RAW: volumes=${raw_vols}  slices(pngs)=${raw_slices}"
    echo "  GT : volumes=${gt_vols}   slices(pngs)=${gt_slices}"
  done

  echo "--------------------"
  echo "TOTAL (FLAIR + T1 + T2):"
  echo "  RAW: volumes=${total_raw_vols}  slices(pngs)=${total_raw_slices}"
  echo "  GT : volumes=${total_gt_vols}   slices(pngs)=${total_gt_slices}"
  echo
done



"""
SPLIT: train
RAW_ROOT: /storage/ice-shared/ae8803che/lmkh3/train_labeled_raw_by_modality
GT_ROOT : /storage/ice-shared/ae8803che/lmkh3/train_labeled_gt_by_modality
====================
FLAIR:
  RAW: volumes=340  slices(pngs)=5392
  GT : volumes=340   slices(pngs)=1132
T1:
  RAW: volumes=410  slices(pngs)=6394
  GT : volumes=410   slices(pngs)=1282
T2:
  RAW: volumes=0  slices(pngs)=0
  GT : volumes=0   slices(pngs)=0
--------------------
TOTAL (FLAIR + T1 + T2):
  RAW: volumes=750  slices(pngs)=11786
  GT : volumes=750   slices(pngs)=2414

====================
SPLIT: val
RAW_ROOT: /storage/ice-shared/ae8803che/lmkh3/val_labeled_raw_by_modality
GT_ROOT : /storage/ice-shared/ae8803che/lmkh3/val_labeled_gt_by_modality
====================
FLAIR:
  RAW: volumes=107  slices(pngs)=1694
  GT : volumes=107   slices(pngs)=367
T1:
  RAW: volumes=140  slices(pngs)=2186
  GT : volumes=140   slices(pngs)=496
T2:
  RAW: volumes=0  slices(pngs)=0
  GT : volumes=0   slices(pngs)=0
--------------------
TOTAL (FLAIR + T1 + T2):
  RAW: volumes=247  slices(pngs)=3880
  GT : volumes=247   slices(pngs)=863"""