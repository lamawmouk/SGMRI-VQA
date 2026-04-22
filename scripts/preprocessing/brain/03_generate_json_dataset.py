#!/usr/bin/env python3
"""
03_generate_json_dataset.py

Generates a JSON dataset for MRI volumes (train or val) with VARIABLE slice counts.
Stores RELATIVE paths by default for portability across machines.

Usage:
  python3 03_generate_json_dataset.py --split train --data-root /path/to/data
  python3 03_generate_json_dataset.py --split val --data-root /path/to/data
  python3 03_generate_json_dataset.py --split train --data-root ./data --csv ./brain.csv --output ./output.json

For each volume:
  - Includes ALL slices found on disk (no assumptions about count)
  - Slice numbering uses 1-based indexing (matches PNG filenames)
  - Attach slice-level labels (from CSV, converted from 0-based to 1-based)
  - Attach bounding boxes (from CSV)
  - final_diagnosis = union of all labels for that volume
  - Volumes with fewer than MIN_SLICES are filtered out
  - Paths stored as RELATIVE (e.g., train_labeled_raw_by_modality/FLAIR/volume/file.png)
"""

import os
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------------
# DEFAULTS (no hardcoded absolute paths)
# --------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_CSV = str(SCRIPT_DIR / "brain.csv")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
MODALITIES = ["FLAIR", "T1"]
MIN_SLICES = 10
# --------------------------------------------------------


def load_annotations(csv_path):
    """Load annotations from CSV and convert to 1-based indexing."""

    annotations = defaultdict(lambda: defaultdict(list))
    slice_labels = defaultdict(lambda: defaultdict(set))
    volume_labels = defaultdict(set)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        # Detect filename column
        possible_cols = ["file", "file_bfile", "fname", "filename"]
        volume_col = next((c for c in possible_cols if c in headers), None)

        if volume_col is None:
            raise ValueError(f"Could not find filename column in CSV. Columns = {headers}")

        print(f"Detected filename column: {volume_col}")

        for row in reader:
            vol = row.get(volume_col)
            if not vol or row.get("slice") is None or row.get("slice") == "":
                continue

            # Convert 0-based CSV to 1-based (to match PNG filenames)
            slice_idx = int(row["slice"]) + 1
            label = row.get("label", "").strip()

            if label:
                slice_labels[vol][slice_idx].add(label)
                volume_labels[vol].add(label)

            if all(row.get(k) for k in ["x", "y", "width", "height"]):
                try:
                    annotations[vol][slice_idx].append({
                        "x": int(row["x"]),
                        "y": int(row["y"]),
                        "width": int(row["width"]),
                        "height": int(row["height"]),
                        "label": label
                    })
                except ValueError:
                    pass

    return annotations, slice_labels, volume_labels


def build_volume_entry(volume_id, modality, base_path, slice_labels, annotations, volume_labels,
                       relative_paths=False, split="train"):
    """Build JSON entry for one volume.

    Args:
        relative_paths: If True, store paths as relative (e.g., train_labeled_raw_by_modality/FLAIR/...)
                        This makes the JSON portable across different machines.
    """
    volume_path = os.path.join(base_path, modality, volume_id)
    slice_entries = []

    pngs = []
    for f in os.listdir(volume_path):
        if f.endswith(".png") and "_slice_" in f:
            try:
                slice_idx = int(f.split("_slice_")[1].split(".")[0])
                pngs.append((slice_idx, f))
            except ValueError:
                continue

    pngs.sort(key=lambda x: x[0])

    for slice_idx, png in pngs:
        if relative_paths:
            # Store as relative path: train_labeled_raw_by_modality/FLAIR/volume_id/file.png
            image_path = f"{split}_labeled_raw_by_modality/{modality}/{volume_id}/{png}"
        else:
            image_path = os.path.join(volume_path, png)

        slice_entries.append({
            "slice": slice_idx,
            "image_path": image_path,
            "label": sorted(slice_labels[volume_id].get(slice_idx, [])),
            "bounding_boxes": annotations[volume_id].get(slice_idx, [])
        })

    return {
        "volume_id": volume_id,
        "final_diagnosis": sorted(volume_labels.get(volume_id, [])),
        "slices": slice_entries
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON dataset for brain MRI volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 03_generate_json_dataset.py --split train --data-root /path/to/data
  python3 03_generate_json_dataset.py --split val --data-root ../../../data
  python3 03_generate_json_dataset.py --split train --data-root ./data --csv ./brain.csv --output ./train.json
        """
    )
    parser.add_argument("--split", type=str, required=True, choices=["train", "val"],
                        help="Dataset split to process (train or val)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root path to data folder containing train_labeled_raw_by_modality/ and val_labeled_raw_by_modality/")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV,
                        help=f"Path to annotations CSV file (default: {DEFAULT_CSV})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: brain_{split}_volumes.json in script directory)")
    parser.add_argument("--min-slices", type=int, default=MIN_SLICES,
                        help=f"Minimum slices per volume (default: {MIN_SLICES})")
    parser.add_argument("--modalities", type=str, nargs="+", default=MODALITIES,
                        help=f"Modalities to process (default: {MODALITIES})")

    args = parser.parse_args()

    # Set paths - data_root is required, output defaults to script directory
    data_root = Path(args.data_root).resolve()
    base_path = data_root / f"{args.split}_labeled_raw_by_modality"
    output_path = args.output or str(DEFAULT_OUTPUT_DIR / f"brain_{args.split}_volumes.json")

    print(f"=== Generating {args.split.upper()} JSON Dataset ===")
    print(f"Data root: {data_root}")
    print(f"Base path: {base_path}")
    print(f"CSV path: {args.csv}")
    print(f"Output: {output_path}")
    print(f"Modalities: {args.modalities}")
    print(f"Min slices: {args.min_slices}")
    print(f"Path format: RELATIVE (portable)")
    print()

    # Validate data root exists
    if not base_path.exists():
        raise FileNotFoundError(f"Base path not found: {base_path}\nMake sure --data-root points to the correct data folder.")

    # Load annotations
    annotations, slice_labels, volume_labels = load_annotations(args.csv)
    print(f"Loaded annotations for {len(volume_labels)} volumes from CSV\n")

    # Build JSON
    giant_json = {m: {} for m in args.modalities}
    skipped_volumes = []

    for modality in args.modalities:
        mod_dir = base_path / modality
        if not mod_dir.is_dir():
            print(f"[WARN] Missing modality folder: {modality}")
            continue

        print(f"=== Processing {modality} volumes ===")

        for volume_name in sorted(os.listdir(mod_dir)):
            volume_path = mod_dir / volume_name
            if not volume_path.is_dir():
                continue

            volume_entry = build_volume_entry(
                volume_name, modality, str(base_path),
                slice_labels, annotations, volume_labels,
                relative_paths=True, split=args.split
            )

            num_slices = len(volume_entry["slices"])
            if num_slices < args.min_slices:
                skipped_volumes.append((volume_name, num_slices))
                print(f"  SKIPPING: {volume_name} (only {num_slices} slices)")
                continue

            print(f"  Adding: {volume_name} ({num_slices} slices)")
            giant_json[modality][volume_name] = volume_entry

    # Save JSON
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as jf:
        json.dump(giant_json, jf, indent=2)

    print(f"\nSaved to: {output_path}")

    # Report skipped volumes
    if skipped_volumes:
        print(f"\n=== Skipped {len(skipped_volumes)} volumes (< {args.min_slices} slices) ===")
        for vol_name, num_slices in skipped_volumes:
            print(f"  {vol_name}: {num_slices} slices")

    # Statistics
    print("\n=== Statistics ===")
    total_volumes = 0
    total_slices = 0
    slices_with_findings = 0

    for modality in args.modalities:
        for vol_id, vol_data in giant_json[modality].items():
            total_volumes += 1
            for s in vol_data["slices"]:
                total_slices += 1
                if s["bounding_boxes"]:
                    slices_with_findings += 1

    print(f"Total volumes: {total_volumes}")
    print(f"Total slices: {total_slices}")
    print(f"Slices with findings: {slices_with_findings}")
    print(f"Normal slices: {total_slices - slices_with_findings}")


if __name__ == "__main__":
    main()
