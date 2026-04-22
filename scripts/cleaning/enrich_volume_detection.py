#!/usr/bin/env python3
"""Enrich volume-level detection reasoning to mention ALL ground-truth findings."""
import json
import random
import re
import copy

random.seed(42)

# ── Label → characteristic key mapping ──────────────────────────────────

KNEE_LABEL_MAP = {
    'meniscus tear': 'meniscus tear',
    'displaced meniscal tissue': 'displaced meniscal tissue',
    'ligament - acl high grade sprain': 'acl',
    'ligament - acl low grade sprain': 'acl',
    'ligament - mcl high grade sprain': 'mcl',
    'ligament - mcl low-mod grade sprain': 'mcl',
    'ligament - pcl high grade': 'pcl',
    'ligament - pcl low-mod grade sprain': 'pcl',
    'lcl complex - low-mod grade sprain': 'lcl',
    'lcl complex- high grade sprain': 'lcl',
    'joint effusion': 'joint effusion',
    'periarticular cysts': 'periarticular cyst',
    'cartilage - full thickness loss/defect': 'cartilage - full thickness',
    'cartilage - partial thickness loss/defect': 'cartilage - partial thickness',
    'bone- subchondral edema': 'subchondral edema',
    'bone-fracture/contusion/dislocation': 'fracture',
    'bone - lesion': 'bone lesion',
    'soft tissue lesion': 'soft tissue lesion',
    'muscle strain': 'muscle strain',
    'patellar retinaculum - high grade sprain': 'patellar retinaculum',
    'joint bodies': 'joint bodies',
}

BRAIN_LABEL_MAP = {
    'nonspecific white matter lesion': 'nonspecific white matter lesion',
    'nonspecific lesion': 'nonspecific white matter lesion',
    'craniotomy': 'craniotomy',
    'enlarged ventricles': 'enlarged ventricles',
    'edema': 'edema',
    'dural thickening': 'dural thickening',
    'resection cavity': 'resection cavity',
    'lacunar infarct': 'lacunar infarct',
    'encephalomalacia': 'encephalomalacia',
    'mass': 'mass',
    'extra-axial mass': 'extra-axial mass',
    'extra-axial collection': 'extra-axial collection',
    'craniectomy': 'craniectomy',
    'craniectomy with cranioplasty': 'craniectomy with cranioplasty',
    'pineal cyst': 'pineal cyst',
    'possible artifact': 'possible artifact',
    'motion artifact': 'motion artifact',
    'posttreatment change': 'posttreatment change',
    # These don't need clinical descriptions in detection
    'normal for age': None,
    'normal variant': None,
    'absent septum pellucidum': None,
    'paranasal sinus opacification': 'paranasal sinus opacification',
    'intraventricular substance': None,
    'likely cysts': 'likely cysts',
}

# ── Import characteristic dicts from cleanup scripts ────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from clean_knee_qa import KNEE_IMAGING_CHARACTERISTICS, _pluralize_description as knee_pluralize
from clean_brain_qa import BRAIN_IMAGING_CHARACTERISTICS, _pluralize_description as brain_pluralize


def _is_label_mentioned(label, reasoning_lower):
    """Check if a diagnosis label is already mentioned in reasoning."""
    label_lower = label.lower()
    # Direct match
    if label_lower in reasoning_lower:
        return True
    # Key word match (words > 3 chars)
    key_words = [w for w in label_lower.split() if len(w) > 3]
    if key_words and all(w in reasoning_lower for w in key_words):
        return True
    # Common abbreviations
    abbrev_map = {
        'acl': ['anterior cruciate', 'acl'],
        'mcl': ['medial collateral', 'mcl'],
        'pcl': ['posterior cruciate', 'pcl'],
        'lcl': ['lateral collateral', 'lcl'],
    }
    for abbr, terms in abbrev_map.items():
        if abbr in label_lower:
            if any(t in reasoning_lower for t in terms):
                return True
    # Specific partial matches
    partial_checks = {
        'meniscus tear': ['meniscus', 'meniscal', 'intrameniscal'],
        'periarticular cysts': ['periarticular', 'cyst'],
        'joint effusion': ['effusion'],
        'bone- subchondral edema': ['subchondral'],
        'bone-fracture/contusion/dislocation': ['fracture', 'contusion'],
        'cartilage - full thickness': ['full thickness', 'cartilage'],
        'cartilage - partial thickness': ['partial thickness'],
        'displaced meniscal tissue': ['displaced meniscal'],
        'enlarged ventricles': ['ventricul', 'enlarged'],
        'nonspecific white matter lesion': ['white matter lesion', 'hyperintens'],
        'encephalomalacia': ['encephalomalacia', 'volume loss'],
        'lacunar infarct': ['lacunar'],
        'dural thickening': ['dural'],
        'resection cavity': ['resection'],
        'extra-axial mass': ['extra-axial mass'],
        'extra-axial collection': ['extra-axial collection', 'extra-axial fluid'],
        'craniectomy with cranioplasty': ['cranioplasty'],
        'craniectomy': ['craniectomy'],
        'craniotomy': ['craniotomy'],
        'pineal cyst': ['pineal'],
        'edema': ['edema'],
        'mass': ['mass', 'space-occupying'],
    }
    for key, terms in partial_checks.items():
        if key in label_lower:
            if any(t in reasoning_lower for t in terms):
                return True
    return False


def enrich_volume_detection(qa, label_map, char_dict, pluralize_fn, anatomy='knee'):
    """Enrich volume-level detection reasoning to mention all findings."""
    count = 0
    for e in qa:
        if e.get('task') != 'detection':
            continue
        if e.get('level') != 'volume':
            continue

        gt = e.get('ground_truth', {})
        if not isinstance(gt, dict):
            continue
        diagnoses = gt.get('final_diagnosis', [])
        if len(diagnoses) <= 1:
            continue

        reasoning = e.get('reasoning', '')
        reasoning_lower = reasoning.lower()

        # Find diagnoses not mentioned
        missing = []
        for diag in diagnoses:
            if not _is_label_mentioned(diag, reasoning_lower):
                missing.append(diag)

        if not missing:
            continue

        additions = []
        for diag in missing:
            diag_lower = diag.lower()
            char_key = label_map.get(diag_lower)

            if char_key is None:
                # Skip labels we don't have descriptions for (normal, etc.)
                # But add a generic mention for unlisted ones
                if diag_lower not in label_map:
                    additions.append(f'Additionally, {diag.lower()} is identified.')
                continue

            # Use FLAIR-specific descriptions when modality is FLAIR
            modality = e.get('modality', '').upper()
            flair_key = char_key + '_flair'
            if 'FLAIR' in modality and flair_key in char_dict:
                descs = char_dict.get(flair_key)
            else:
                descs = char_dict.get(char_key)
            if descs:
                desc = random.choice(descs)
                additions.append(desc)
            else:
                additions.append(f'Additionally, {diag.lower()} is identified.')

        if additions:
            # Join with existing reasoning
            new_reasoning = reasoning.rstrip('. ') + '. ' + ' '.join(additions)
            e['reasoning'] = new_reasoning
            count += 1

    return count


def process_file(filepath, label_map, char_dict, pluralize_fn, anatomy):
    with open(filepath) as f:
        data = json.load(f)

    # Count before
    total_vol_det = sum(1 for e in data if e.get('task') == 'detection' and e.get('level') == 'volume')
    multi_diag = sum(1 for e in data if e.get('task') == 'detection' and e.get('level') == 'volume'
                     and len(e.get('ground_truth', {}).get('final_diagnosis', [])) > 1)

    enriched = enrich_volume_detection(data, label_map, char_dict, pluralize_fn, anatomy)

    # Verify: count entries still missing findings
    still_missing = 0
    for e in data:
        if e.get('task') != 'detection' or e.get('level') != 'volume':
            continue
        gt = e.get('ground_truth', {})
        diagnoses = gt.get('final_diagnosis', [])
        if len(diagnoses) <= 1:
            continue
        reasoning_lower = e.get('reasoning', '').lower()
        for diag in diagnoses:
            if not _is_label_mentioned(diag, reasoning_lower):
                diag_lower = diag.lower()
                char_key = label_map.get(diag_lower)
                if char_key is not None:  # Only count ones we should have fixed
                    still_missing += 1
                    break

    print(f"  {filepath}")
    print(f"    Volume detection entries: {total_vol_det}")
    print(f"    Multi-diagnosis entries: {multi_diag}")
    print(f"    Enriched: {enriched}")
    print(f"    Still missing (with available descriptions): {still_missing}")

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"    Saved.")


if __name__ == '__main__':
    base = os.path.dirname(__file__)

    print("=== Knee Files ===")
    for fn in [
        'knee/gpt4o_knee_train_volume_qa_volume_qa_pairs.json',
        'knee/gpt4o_knee_test_volume_qa_volume_qa_pairs.json',
    ]:
        process_file(os.path.join(base, fn), KNEE_LABEL_MAP,
                     KNEE_IMAGING_CHARACTERISTICS, knee_pluralize, 'knee')

    print("\n=== Brain Files ===")
    for fn in [
        'brain/gpt4o_brain_train_volume_qa_volume_qa_pairs.json',
        'brain/gpt4o_brain_test_volume_qa_volume_qa_pairs.json',
    ]:
        process_file(os.path.join(base, fn), BRAIN_LABEL_MAP,
                     BRAIN_IMAGING_CHARACTERISTICS, brain_pluralize, 'brain')
