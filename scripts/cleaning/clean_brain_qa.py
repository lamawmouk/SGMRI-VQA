#!/usr/bin/env python3
"""Post-processing cleanup for GPT-4o brain QA data."""

import json
import re
import os
import random
import argparse
import shutil
from collections import Counter, defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRAIN_DIR = os.path.join(BASE_DIR, "brain")

# Brain image is 256x256
# Hemisphere rule: x < 130 = right, x >= 130 = left (radiological convention)
MIDPOINT_X = 130
IMG_SIZE = 256

# Volume metadata directory (for enrichment steps)
DEFAULT_METADATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data_processing", "brain"))



def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def backup(path):
    bak = path + '.bak'
    if not os.path.exists(bak):
        shutil.copy2(path, bak)

def get_bbox_x_values(text):
    """Extract all bbox x values from text like [x,y,w,h] or <bbx>[x,y,w,h]</bbx>"""
    return [int(b[0]) for b in re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text)]

def get_bboxes_from_text(text):
    """Extract all (x, y, w, h) tuples from text."""
    return [(int(a), int(b), int(c), int(d))
            for a, b, c, d in re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text)]

def determine_hemisphere(x_values):
    """Determine hemisphere from bbox x values. Returns 'right', 'left', or 'bilateral'."""
    if not x_values:
        return None
    has_right = any(x < MIDPOINT_X for x in x_values)
    has_left = any(x >= MIDPOINT_X for x in x_values)
    if has_right and has_left:
        return 'bilateral'
    elif has_right:
        return 'right'
    else:
        return 'left'



def load_brain_volume_metadata(split, metadata_dir=None):
    """Load brain volume metadata and flatten across modalities."""
    mdir = metadata_dir or DEFAULT_METADATA_DIR
    path = os.path.join(mdir, f"brain_{split}_volumes.json")
    if not os.path.exists(path):
        print(f"  WARNING: Metadata not found at {path}")
        return {}
    raw = load_json(path)
    flat = {}
    for modality_data in raw.values():
        if isinstance(modality_data, dict):
            for vol_id, vol_data in modality_data.items():
                flat[vol_id] = vol_data
    return flat


def get_volume_bbox_frames(vol_data):
    """Get bboxes organized by frame from volume metadata."""
    frames = {}
    for s in vol_data.get('slices', []):
        bbs = s.get('bounding_boxes', [])
        if bbs:
            frames[s['slice']] = bbs
    return frames




def fix_periventricular_to_hemisphere(qa):
    """Replace 'periventricular region' with correct hemisphere based on bbox."""
    count = 0
    for e in qa:
        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            if 'periventricular' not in text.lower():
                continue
            x_values = get_bbox_x_values(text)
            hem = determine_hemisphere(x_values)
            if not hem:
                continue
            if hem == 'bilateral':
                replacement = 'the bilateral hemispheres'
            else:
                replacement = f'the {hem} hemisphere'
            new_text = re.sub(
                r'(?:the\s+)?periventricular\s+(?:region|area|white matter)',
                replacement, text, flags=re.IGNORECASE
            )
            if new_text != text:
                e[field] = new_text
                count += 1
    return count


def fix_midline_references(qa):
    """Fix 'midline' references to bilateral or correct hemisphere."""
    count = 0
    for e in qa:
        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            if 'midline' not in text.lower():
                continue
            # "midline and right/left" → bilateral
            new_text = re.sub(
                r'(?:the\s+)?midline\s+and\s+(?:right|left)\s+hemisphere',
                'the bilateral hemispheres', text, flags=re.IGNORECASE
            )
            # standalone "midline" → bilateral ventricles (for enlarged ventricles)
            if 'enlarged ventricle' in text.lower() or 'ventriculomegaly' in text.lower():
                new_text = re.sub(
                    r'(?:the\s+)?midline(?:\s+region)?',
                    'the bilateral ventricles', new_text, flags=re.IGNORECASE
                )
            if new_text != text:
                e[field] = new_text
                count += 1
    return count


def remove_lobe_names(qa):
    """Replace specific lobe names (frontal, temporal, parietal, occipital) with hemisphere."""
    count = 0
    lobe_pattern = re.compile(
        r'(?:the\s+)?(?:right|left)\s+(?:frontal|temporal|parietal|occipital)\s+'
        r'(?:lobe|region|area|cortex)',
        re.IGNORECASE
    )
    for e in qa:
        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            if not lobe_pattern.search(text):
                continue
            x_values = get_bbox_x_values(text)
            hem = determine_hemisphere(x_values)
            if not hem:
                continue
            if hem == 'bilateral':
                replacement = 'the bilateral hemispheres'
            else:
                replacement = f'the {hem} hemisphere'
            new_text = lobe_pattern.sub(replacement, text)
            if new_text != text:
                e[field] = new_text
                count += 1
    return count


def remove_anatomical_structures(qa):
    """Replace specific anatomical structures with hemisphere."""
    count = 0
    structures = [
        'basal ganglia', 'thalamus', 'caudate', 'putamen', 'internal capsule',
        'corona radiata', 'centrum semiovale', 'corpus callosum',
        'cerebral peduncle', 'insular cortex', 'sylvian fissure',
    ]
    pattern = re.compile(
        r'(?:the\s+)?(?:right|left)\s+(?:' + '|'.join(structures) + r')',
        re.IGNORECASE
    )
    for e in qa:
        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            if not pattern.search(text):
                continue
            x_values = get_bbox_x_values(text)
            hem = determine_hemisphere(x_values)
            if not hem:
                continue
            replacement = f'the {hem} hemisphere' if hem != 'bilateral' else 'the bilateral hemispheres'
            new_text = pattern.sub(replacement, text)
            if new_text != text:
                e[field] = new_text
                count += 1
    return count


def fix_hemisphere_mismatches(qa):
    """Fix cases where stated hemisphere doesn't match bbox positions."""
    count = 0
    for e in qa:
        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            if 'hemisphere' not in text.lower():
                continue

            x_values = get_bbox_x_values(text)
            if not x_values:
                continue

            correct_hem = determine_hemisphere(x_values)
            if not correct_hem:
                continue

            # Skip bilateral ventricles (valid anatomical term)
            if 'bilateral ventricles' in text.lower():
                continue

            # Check if stated hemisphere matches
            states_right = 'right hemisphere' in text.lower()
            states_left = 'left hemisphere' in text.lower()
            states_bilateral = 'bilateral hemispheres' in text.lower()

            needs_fix = False
            if correct_hem == 'bilateral' and not states_bilateral and (states_right or states_left):
                needs_fix = True
            elif correct_hem == 'right' and not states_right and (states_left or states_bilateral):
                if states_bilateral and len(x_values) == 1:
                    needs_fix = True
                elif states_left:
                    needs_fix = True
            elif correct_hem == 'left' and not states_left and (states_right or states_bilateral):
                if states_bilateral and len(x_values) == 1:
                    needs_fix = True
                elif states_right:
                    needs_fix = True

            if needs_fix:
                if correct_hem == 'bilateral':
                    new_text = re.sub(r'(?:the\s+)?(?:right|left)\s+hemisphere',
                                     'the bilateral hemispheres', text, flags=re.IGNORECASE)
                else:
                    new_text = re.sub(
                        r'(?:the\s+)?(?:right|left|bilateral)\s+hemisphere(?:s)?',
                        f'the {correct_hem} hemisphere', text, flags=re.IGNORECASE
                    )
                if new_text != text:
                    e[field] = new_text
                    count += 1
    return count


def fix_white_matter_reasoning(qa):
    """Fix reasoning that says 'in the white matter' → correct hemisphere."""
    count = 0
    for e in qa:
        reasoning = e.get('reasoning', '')
        if 'in the white matter' not in reasoning.lower():
            continue
        x_values = get_bbox_x_values(reasoning)
        if not x_values:
            # Try answer field
            x_values = get_bbox_x_values(e.get('answer', ''))
        if not x_values:
            continue
        hem = determine_hemisphere(x_values)
        if not hem:
            continue
        replacement = f'in the {hem} hemisphere' if hem != 'bilateral' else 'in the bilateral hemispheres'
        new_text = re.sub(r'in the white matter(?:\s+regions?)?', replacement,
                          reasoning, flags=re.IGNORECASE)
        if new_text != reasoning:
            e['reasoning'] = new_text
            count += 1
    return count


def fix_single_bbox_bilateral(qa):
    """Fix single-bbox entries incorrectly labeled 'bilateral hemispheres'."""
    count = 0
    for e in qa:
        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            if 'bilateral hemispheres' not in text.lower():
                continue
            if 'bilateral ventricles' in text.lower():
                continue

            bboxes = get_bboxes_from_text(text)
            if len(bboxes) != 1:
                continue

            x, y, w, h = bboxes[0]
            # Skip wide artifacts spanning entire image
            if w > 200:
                continue

            correct_hem = 'right' if x < MIDPOINT_X else 'left'
            new_text = re.sub(
                r'(?:the\s+)?bilateral\s+hemispheres',
                f'the {correct_hem} hemisphere', text, flags=re.IGNORECASE
            )
            if new_text != text:
                e[field] = new_text
                count += 1
    return count


def fix_artifact_spanning(qa):
    """Fix artifact bboxes labeled as 'spanning' using majority rule."""
    count = 0
    for e in qa:
        gt = str(e.get('ground_truth', '')).lower()
        if 'artifact' not in gt and 'motion' not in gt:
            continue

        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            x_values = get_bbox_x_values(text)
            if not x_values:
                continue

            right_count = sum(1 for x in x_values if x < MIDPOINT_X)
            left_count = sum(1 for x in x_values if x >= MIDPOINT_X)
            total = right_count + left_count
            if total == 0:
                continue

            # Apply majority rule (60% threshold)
            if right_count / total >= 0.6:
                majority_hem = 'right'
            elif left_count / total >= 0.6:
                majority_hem = 'left'
            else:
                majority_hem = 'bilateral'

            # Fix "spanning" phrases
            if 'spanning' in text.lower():
                if majority_hem == 'bilateral':
                    replacement = 'the bilateral hemispheres'
                else:
                    replacement = f'the {majority_hem} hemisphere'
                new_text = re.sub(
                    r'spanning\s+(?:the\s+)?(?:left\s+and\s+right|right\s+and\s+left)\s+hemispheres?',
                    replacement, text, flags=re.IGNORECASE
                )
                if new_text != text:
                    e[field] = new_text
                    count += 1
    return count




def remove_patient_history_phrases(qa):
    """Remove phrases like 'This patient was reported...', 'Given the patient's history...'"""
    count = 0
    patterns = [
        r'This patient was reported[^.]*\.\s*',
        r'Given the patient\'s history[^.]*\.\s*',
        r'In the original image,\s*',
    ]
    for e in qa:
        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            new_text = text
            for pat in patterns:
                new_text = re.sub(pat, '', new_text, flags=re.IGNORECASE)
            # "In a patient with known history..." - keep for craniotomy captioning
            if e.get('task') == 'captioning' and 'craniotomy' in str(e.get('ground_truth', '')).lower():
                pass
            else:
                new_text = re.sub(r'In a patient with(?:\s+known)?\s+history[^.]*\.\s*', '',
                                  new_text, flags=re.IGNORECASE)
            if new_text != text:
                e[field] = new_text
                count += 1
    return count


def fix_grammar(qa):
    """Fix common grammar issues."""
    count = 0
    fixes = [
        ('unremarkable', 'normal'),
        ('abnormality and shows', 'abnormality shows'),
        ('pathological findings', 'findings'),
        ('categoryies', 'categories'),
        ('1 types', '1 type'),
        ('is In bilateral', 'is in bilateral'),
        ('in bilateral hemispheres', 'in the bilateral hemispheres'),
        ('And, ', ''),
    ]
    for e in qa:
        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            new_text = text
            for old, new in fixes:
                new_text = new_text.replace(old, new)
            if new_text != text:
                e[field] = new_text
                count += 1
    return count


def fix_frame_references(qa):
    """Fix 'from 1 to 1' and stray volume digits."""
    count = 0
    for e in qa:
        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            new_text = text
            # "from 1 to 1" → "across the volume"
            new_text = re.sub(r'from 1 to 1', 'across the volume', new_text)
            # "the 1 frames" → "the volume"
            new_text = re.sub(r'the 1 frames?', 'the volume', new_text)
            # Stray "volumeN" digits
            new_text = re.sub(r'volume\d+', '', new_text)
            # Double "across"
            new_text = re.sub(r'across\s+across', 'across', new_text)
            if new_text != text:
                e[field] = new_text
                count += 1
    return count


def fix_trailing_sentences(qa):
    """Remove trailing incomplete sentences (ending with prepositions)."""
    count = 0
    trailing_pattern = re.compile(r'\s+(?:at|in|on|of|for|with|to|from|by)\s*\.?\s*$')
    for e in qa:
        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            new_text = trailing_pattern.sub('.', text)
            if new_text != text:
                e[field] = new_text
                count += 1
    return count


def remove_false_normal(qa):
    """Remove false 'suggestive of normal' sentences from entries that have actual disease."""
    count = 0
    normal_labels = {'normal for age', 'normal', 'motion artifact', 'normal brain',
                     'normal variant', 'normal finding'}
    for e in qa:
        gt = e.get('ground_truth', {})
        fd = gt.get('final_diagnosis', []) if isinstance(gt, dict) else []
        # Check if there are real (non-normal) diagnoses
        real_diags = [d for d in fd if d.lower() not in normal_labels]
        if not real_diags:
            continue

        reasoning = e.get('reasoning', '')
        # Remove "suggestive of normal" sentences
        new_text = re.sub(
            r'[^.]*(?:suggestive of|consistent with|indicates?)\s+normal(?:\s+(?:age-related\s+)?changes?)?[^.]*\.\s*',
            '', reasoning, flags=re.IGNORECASE
        )
        if new_text != reasoning:
            e['reasoning'] = new_text.strip()
            count += 1
    return count


def remove_duplicate_hemisphere_phrases(qa):
    """Remove duplicated hemisphere references like 'right hemisphere right hemisphere'."""
    count = 0
    for hem in ['right hemisphere', 'left hemisphere', 'bilateral hemispheres']:
        pattern = re.compile(re.escape(hem) + r'\s+' + re.escape(hem), re.IGNORECASE)
        for e in qa:
            for field in ['answer', 'reasoning']:
                text = e.get(field, '')
                new_text = pattern.sub(hem, text)
                if new_text != text:
                    e[field] = new_text
                    count += 1
    return count




# Imaging characteristic descriptions for brain findings (no bboxes)
BRAIN_IMAGING_CHARACTERISTICS = {
    # --- User-provided descriptions (diversified into 8 variations) ---
    'nonspecific white matter lesion': [
        'A nonspecific hyperintense appearance is noted in the white matter, consistent with a nonspecific white matter lesion.',
        'Nonspecific hyperintense signal is identified in the white matter, suggestive of a white matter lesion.',
        'An area of nonspecific hyperintensity is seen in the white matter, representing a nonspecific white matter lesion.',
        'A small focal area of nonspecific hyperintense appearance is observed in the white matter, consistent with a white matter lesion.',
        'Nonspecific hyperintense signal abnormality is noted in the white matter, suggestive of a nonspecific lesion.',
        'A focus of nonspecific hyperintense appearance is identified in the cerebral white matter.',
        'Hyperintense signal with a nonspecific appearance is present in the white matter, consistent with a nonspecific white matter lesion.',
        'An area of nonspecific hyperintensity is observed in the white matter, suggestive of a white matter lesion.',
    ],
    'craniotomy': [
        'A heterogeneous appearance with areas of signal void is noted, consistent with a prior craniotomy.',
        'Heterogeneous signal with focal signal void is identified, indicating a prior craniotomy.',
        'A region of heterogeneous appearance and associated signal void is seen, consistent with prior craniotomy.',
        'Signal void with surrounding heterogeneous appearance is observed, suggestive of a prior craniotomy.',
        'Heterogeneous signal change with areas of signal void is noted, consistent with a craniotomy.',
        'A heterogeneous region with signal void is identified, indicating prior surgical craniotomy.',
        'Focal signal void with adjacent heterogeneous appearance is seen, consistent with a prior craniotomy.',
        'Heterogeneous signal characteristics with signal void are observed, suggestive of a prior craniotomy.',
    ],
    'enlarged ventricles': [
        'Enlarged regions of hypointensity are noted in the ventricles, consistent with ventriculomegaly.',
        'The ventricles demonstrate enlarged regions of hypointense signal, consistent with ventriculomegaly.',
        'Regions of hypointensity are seen with ventricular enlargement, indicating ventriculomegaly.',
        'Enlarged hypointense regions are identified within the ventricular system, suggestive of ventriculomegaly.',
        'Ventricular enlargement with prominent regions of hypointensity is observed, consistent with ventriculomegaly.',
        'The ventricular system shows enlarged hypointense regions, indicating ventriculomegaly.',
        'Enlarged regions of hypointense signal are noted within the ventricles, representing ventriculomegaly.',
        'Prominent hypointense ventricular enlargement is identified, consistent with ventriculomegaly.',
    ],
    # Edema: T1 = hypointensity, FLAIR = hyperintensity. Use 'edema' for T1 (default), 'edema_flair' for FLAIR.
    'edema': [
        'Local hypointensity with distorted architecture is noted, consistent with edema.',
        'An area of local hypointense signal with distorted parenchymal architecture is seen, suggestive of edema.',
        'Distorted architecture with local hypointensity is identified, consistent with cerebral edema.',
        'Local hypointense signal change with architectural distortion is observed, representing edema.',
        'An area of hypointensity with distortion of adjacent brain architecture is noted, consistent with edema.',
        'Parenchymal architecture distortion with local hypointensity is seen, suggestive of edema.',
        'Local hypointense signal with loss of normal architecture is identified, consistent with edema.',
        'Distorted brain architecture with an area of local hypointensity is observed, suggestive of edema.',
    ],
    'edema_flair': [
        'Local hyperintensity with distorted architecture is noted, consistent with edema.',
        'An area of local hyperintense signal with distorted parenchymal architecture is seen, suggestive of edema.',
        'Distorted architecture with local hyperintensity is identified, consistent with cerebral edema.',
        'Local hyperintense signal change with architectural distortion is observed, representing edema.',
        'An area of hyperintensity with distortion of adjacent brain architecture is noted, consistent with edema.',
        'Parenchymal architecture distortion with local hyperintensity is seen, suggestive of edema.',
        'Local hyperintense signal with loss of normal architecture is identified, consistent with edema.',
        'Distorted brain architecture with an area of local hyperintensity is observed, suggestive of edema.',
    ],
    'dural thickening': [
        'Smooth thickening of the dura mater is noted, consistent with dural thickening.',
        'Linear thickening of the dura mater is identified, suggestive of dural thickening.',
        'Nodular thickening of the dura mater is seen, consistent with dural thickening.',
        'Smooth to linear thickening of the dura mater is observed, indicating dural thickening.',
        'Thickening of the dura mater with a smooth appearance is noted, consistent with dural thickening.',
        'The dura mater demonstrates linear thickening, suggestive of a dural thickening process.',
        'Focal nodular thickening of the dura mater is identified, consistent with dural thickening.',
        'Smooth and linear thickening of the dura mater is observed, consistent with dural thickening.',
    ],
    'resection cavity': [
        'Hyperintense areas are noted at the surgical site, consistent with a resection cavity.',
        'A region of hyperintense signal is identified, representing a resection cavity.',
        'Areas of hyperintensity are seen at the site of prior surgical resection, consistent with a resection cavity.',
        'Hyperintense signal is observed at the resection site, consistent with a post-surgical cavity.',
        'A hyperintense area is identified, consistent with a resection cavity from prior surgery.',
        'The resection site demonstrates hyperintense signal characteristics, consistent with a resection cavity.',
        'Focal hyperintense areas are noted at the surgical bed, representing a resection cavity.',
        'A well-defined hyperintense region is seen, consistent with a post-surgical resection cavity.',
    ],
    'lacunar infarct': [
        'A slightly hypointense signal change is noted, consistent with a lacunar infarct.',
        'A small area of slightly hypointense signal is identified, suggestive of a lacunar infarct.',
        'Slightly hypointense signal change is seen in the deep structures, consistent with a lacunar infarct.',
        'A focal area of slightly hypointense signal is observed, representing a lacunar infarct.',
        'A small region of slightly hypointense signal change is noted, consistent with a chronic lacunar infarct.',
        'Slightly hypointense signal abnormality is identified in the deep brain, suggestive of a lacunar infarct.',
        'A punctate area of slightly hypointense signal change is seen, consistent with a lacunar infarct.',
        'The deep structures demonstrate a small area of slightly hypointense signal, consistent with a lacunar infarct.',
    ],
    # --- Diagnoses without user-provided descriptions (kept as-is) ---
    'encephalomalacia': [
        'An area of parenchymal volume loss with abnormal signal is noted, consistent with encephalomalacia.',
        'Focal tissue loss with CSF-signal change is seen, representing encephalomalacia.',
        'Cystic or gliotic change with volume loss is identified, consistent with encephalomalacia.',
        'An area of brain tissue softening with associated signal abnormality is observed, consistent with encephalomalacia.',
        'Parenchymal loss with ex-vacuo dilation is noted, representing chronic encephalomalacia.',
        'A region of focal brain atrophy with volume loss is identified, consistent with encephalomalacia.',
        'Chronic tissue destruction with CSF-intensity signal change is seen, indicating encephalomalacia.',
        'Focal parenchymal defect with surrounding gliosis is noted, consistent with encephalomalacia.',
    ],
    'mass': [
        'A space-occupying lesion with abnormal signal characteristics is identified.',
        'A heterogeneous lesion with mass effect is seen, consistent with a mass.',
        'A focal lesion with abnormal signal intensity is noted, suggestive of a mass.',
        'A parenchymal lesion with mass effect is identified, consistent with an intracranial mass.',
        'A focal space-occupying lesion with heterogeneous signal is noted, suggestive of an intracranial neoplasm.',
        'An intraparenchymal mass with surrounding edema is observed.',
        'A lesion demonstrating mass effect and abnormal signal characteristics is identified.',
        'A focal brain lesion with signal heterogeneity is seen, consistent with a mass.',
    ],
    'extra-axial mass': [
        'An extra-axial lesion is identified along the dural surface with mass effect.',
        'A dural-based lesion is seen outside the brain parenchyma, consistent with an extra-axial mass.',
        'An extra-axial space-occupying lesion with typical dural characteristics is noted.',
        'A lesion arising outside the brain parenchyma is identified, consistent with an extra-axial mass.',
        'A well-circumscribed extra-axial lesion with dural attachment is observed.',
        'An extra-axial mass with displacement of the adjacent brain surface is seen.',
        'A dural-based space-occupying lesion is identified in the extra-axial compartment.',
        'An extra-axial lesion with broad dural base and smooth margins is noted, consistent with a mass.',
    ],
    'extra-axial collection': [
        'An extra-axial fluid collection is identified between the brain and calvarium.',
        'Fluid signal is noted in the extra-axial space, consistent with an extra-axial collection.',
        'A subdural or epidural fluid collection is seen outside the brain parenchyma.',
        'Extra-axial fluid accumulation is identified, consistent with an extra-axial collection.',
        'A crescent-shaped fluid collection is noted in the extra-axial space.',
        'Fluid is seen layering in the subdural or epidural compartment, consistent with an extra-axial collection.',
        'An extra-axial fluid signal intensity collection is identified adjacent to the brain surface.',
        'A thin layer of extra-axial fluid is noted between the brain and the inner table of the skull.',
    ],
    'craniectomy': [
        'A calvarial defect is noted, consistent with a prior craniectomy.',
        'Absence of calvarium is seen over a region, indicating a prior craniectomy.',
        'A large calvarial defect without bone flap replacement is identified, consistent with a craniectomy.',
        'Post-surgical calvarial absence is noted, consistent with a prior craniectomy.',
        'A segment of the calvarium is absent, indicating a prior craniectomy procedure.',
        'Surgical removal of a portion of the skull is evident, consistent with a craniectomy.',
        'A wide calvarial defect with absent bone is seen, representing a prior craniectomy.',
        'The calvarium demonstrates a large surgical defect without reconstruction, consistent with a craniectomy.',
    ],
    'craniectomy with cranioplasty': [
        'Post-surgical changes are noted with cranioplasty material in place, consistent with a prior craniectomy with cranioplasty.',
        'A calvarial defect with reconstruction material is seen, indicating prior craniectomy with cranioplasty.',
        'Cranioplasty material is identified at the site of a prior craniectomy.',
        'Post-surgical changes with prosthetic calvarial reconstruction are noted.',
        'Reconstruction material is seen at a prior craniectomy site, consistent with cranioplasty repair.',
        'A calvarial defect with overlying cranioplasty prosthesis is identified.',
        'Surgical reconstruction of the calvarium is noted, with cranioplasty material at the craniectomy site.',
        'Evidence of prior craniectomy with subsequent cranioplasty reconstruction is observed.',
    ],
    'pineal cyst': [
        'A small well-defined cystic lesion is noted in the pineal region, consistent with a pineal cyst.',
        'A round, fluid-signal lesion is identified in the pineal gland, representing a pineal cyst.',
        'A benign-appearing cystic structure is seen in the pineal region.',
        'A small pineal cyst is noted, a common incidental finding.',
        'A well-circumscribed fluid-intensity lesion is present in the pineal gland.',
        'A simple cystic lesion is identified in the pineal region, with no complicating features.',
        'A small round cyst with fluid signal characteristics is noted in the pineal gland.',
        'A pineal region cyst is observed, consistent with a benign pineal cyst.',
    ],
    'possible artifact': [
        'Signal abnormality is noted, which may represent a possible artifact.',
        'An area of signal distortion is seen, likely representing an artifact.',
        'Signal irregularity is identified, possibly artifactual in nature.',
        'A focus of abnormal signal is noted, which may be artifactual.',
        'An area of signal inhomogeneity is observed, likely related to an imaging artifact.',
        'Focal signal distortion is present, possibly representing a susceptibility or motion artifact.',
        'An artifactual signal abnormality is noted, without corresponding structural abnormality.',
        'Signal heterogeneity is seen, which may represent an imaging artifact rather than true pathology.',
    ],
    'motion artifact': [
        'Motion-related signal distortion is noted across the image.',
        'Ghosting artifacts from patient motion are identified.',
        'Signal blurring consistent with motion artifact is observed.',
        'Motion-related degradation of image quality is noted.',
        'Phase-encoding direction ghosting is seen, consistent with patient motion during acquisition.',
        'The image demonstrates motion-related signal degradation with blurring of anatomic detail.',
        'Artifacts from involuntary patient movement are present, reducing image quality.',
        'Motion artifact is identified, with characteristic ghosting along the phase-encoding direction.',
    ],
    'posttreatment change': [
        'Signal changes consistent with prior treatment are observed.',
        'Post-therapeutic changes with abnormal signal are noted.',
        'Treatment-related signal abnormality is identified in the parenchyma.',
        'Post-treatment changes are noted, with expected signal alterations.',
        'Signal abnormality consistent with post-radiation or post-surgical changes is seen.',
        'The parenchyma demonstrates expected post-treatment signal alterations.',
        'Post-therapeutic parenchymal changes with abnormal signal intensity are identified.',
        'Imaging findings consistent with prior treatment effect are noted in the brain.',
    ],
    'paranasal sinus opacification': [
        'Opacification of the paranasal sinuses is noted, consistent with mucosal thickening or retained secretions.',
        'Paranasal sinus opacification with low signal intensity is identified, suggestive of mucosal disease.',
        'Mucosal thickening with sinus opacification is seen in the paranasal sinuses.',
        'The paranasal sinuses demonstrate opacification, consistent with mucosal thickening or a polyp.',
        'Sinus opacification with signal characteristics suggesting fluid or mucosal thickening is observed.',
        'Paranasal sinus mucosal thickening is noted, appearing as opacification of the sinus cavity.',
        'Opacification of the paranasal sinuses is identified, likely representing mucosal disease or retained fluid.',
        'The sinuses demonstrate opacification consistent with mucosal thickening, fluid, or polyp formation.',
    ],
    'paranasal sinus opacification_flair': [
        'Hyperintense opacification of the paranasal sinuses is noted on FLAIR, consistent with mucosal thickening or retained secretions.',
        'Paranasal sinus opacification with high FLAIR signal is identified, suggestive of mucosal disease.',
        'Mucosal thickening with hyperintense sinus opacification is seen in the paranasal sinuses on FLAIR.',
        'The paranasal sinuses demonstrate hyperintense opacification on FLAIR, consistent with mucosal thickening or a polyp.',
        'Sinus opacification with high signal on FLAIR suggesting fluid or mucosal thickening is observed.',
        'Paranasal sinus mucosal thickening is noted, appearing as hyperintense opacification on FLAIR.',
        'Hyperintense opacification of the paranasal sinuses is identified on FLAIR, likely representing mucosal disease or retained fluid.',
        'The sinuses demonstrate hyperintense opacification on FLAIR, consistent with mucosal thickening, fluid, or polyp formation.',
    ],
    'likely cysts': [
        'A hypointense well-defined lesion is noted, consistent with an intracranial cyst.',
        'A well-circumscribed hypointense structure is identified, suggestive of a cystic lesion.',
        'A focal area of hypointense signal with well-defined margins is seen, consistent with a cyst.',
        'A well-defined hypointense lesion is observed, likely representing an intracranial cyst.',
        'A cystic-appearing hypointense lesion with smooth margins is noted.',
        'A well-circumscribed lesion with hypointense signal characteristics is identified, consistent with a cyst.',
        'A focal hypointense structure is seen with features consistent with an intracranial cyst.',
        'A well-defined area of hypointense signal is noted, representing a likely cystic lesion.',
    ],
    'likely cysts_flair': [
        'A hyperintense well-defined lesion is noted on FLAIR, consistent with an intracranial cyst.',
        'A well-circumscribed hyperintense structure is identified on FLAIR, suggestive of a cystic lesion.',
        'A focal area of hyperintense signal with well-defined margins is seen on FLAIR, consistent with a cyst.',
        'A well-defined hyperintense lesion is observed on FLAIR, likely representing an intracranial cyst.',
        'A cystic-appearing hyperintense lesion with smooth margins is noted on FLAIR.',
        'A well-circumscribed lesion with hyperintense signal on FLAIR is identified, consistent with a cyst.',
        'A focal hyperintense structure is seen on FLAIR, with features consistent with an intracranial cyst.',
        'A well-defined area of hyperintense signal on FLAIR is noted, representing a likely cystic lesion.',
    ],
}


def _pluralize_description(desc, bbox_count):
    """Convert a singular imaging description to plural when multiple bboxes exist."""
    if bbox_count <= 1:
        return desc
    # Replace leading singular articles
    if desc.startswith('A '):
        desc = 'Multiple ' + desc[2].lower() + desc[3:]
    elif desc.startswith('An '):
        desc = 'Multiple ' + desc[3].lower() + desc[4:]
    # Replace singular verbs with plural
    for sing, plur in [
        (' is noted', ' are noted'),
        (' is seen', ' are seen'),
        (' is identified', ' are identified'),
        (' is observed', ' are observed'),
        (' is present', ' are present'),
    ]:
        desc = desc.replace(sing, plur)
    # Replace singular nouns at end of sentence
    for sing, plur in [
        ('a nonspecific white matter lesion.', 'nonspecific white matter lesions.'),
        ('a nonspecific lesion.', 'nonspecific lesions.'),
        ('a white matter lesion.', 'white matter lesions.'),
        ('a meniscus tear.', 'meniscus tears.'),
        ('a meniscal tear.', 'meniscal tears.'),
        ('a tear.', 'tears.'),
        ('a lacunar infarct.', 'lacunar infarcts.'),
        ('a prior craniotomy.', 'prior craniotomy changes.'),
        ('a joint effusion.', 'joint effusion.'),
        ('a periarticular cyst.', 'periarticular cysts.'),
        ('a sprain.', 'sprains.'),
    ]:
        desc = desc.replace(sing, plur)
    return desc


def enrich_detection_reasoning(qa):
    """Enrich detection reasoning for image QA to mention ALL ground truth findings."""
    count = 0
    for e in qa:
        if e.get('task') != 'detection':
            continue
        if e.get('level') != 'slice':
            continue

        gt = e.get('ground_truth', {})
        if not isinstance(gt, dict):
            continue
        labels = gt.get('labels', [])
        if not labels:
            continue

        # Count bboxes per label for plural handling
        bboxes = gt.get('bboxes', [])
        bbox_counts = {}
        for bb in bboxes:
            lbl = bb.get('label', '')
            bbox_counts[lbl] = bbox_counts.get(lbl, 0) + 1

        reasoning = e.get('reasoning', '')
        reasoning_lower = reasoning.lower()

        # Find labels not mentioned in current reasoning
        missing_labels = []
        for label in labels:
            label_lower = label.lower()
            # Check if any key words from the label appear in reasoning
            key_words = [w for w in label_lower.split() if len(w) > 3]
            # Also check the full label and common abbreviations
            found = label_lower in reasoning_lower
            if not found:
                found = any(w in reasoning_lower for w in key_words)
            if not found:
                # Check common abbreviations
                abbrev_map = {
                    'acl': 'anterior cruciate',
                    'mcl': 'medial collateral',
                    'pcl': 'posterior cruciate',
                    'lcl': 'lateral collateral',
                }
                for abbr, full in abbrev_map.items():
                    if abbr in label_lower and (abbr in reasoning_lower or full in reasoning_lower):
                        found = True
                        break
            if not found:
                missing_labels.append(label)

        if not missing_labels:
            continue

        # Generate descriptions for missing labels
        modality = e.get('modality', '').upper()
        additions = []
        for label in missing_labels:
            label_lower = label.lower()
            # Find matching imaging characteristic
            desc = None
            for key, descs in BRAIN_IMAGING_CHARACTERISTICS.items():
                if key == 'edema_flair':
                    continue  # Skip FLAIR variant in general matching
                if key in label_lower or label_lower in key:
                    # Use FLAIR-specific edema descriptions for FLAIR modality
                    if key == 'edema' and 'FLAIR' in modality:
                        descs = BRAIN_IMAGING_CHARACTERISTICS['edema_flair']
                    desc = random.choice(descs)
                    break
                # Partial match
                key_words = key.split()
                if len(key_words) >= 2 and all(w in label_lower for w in key_words):
                    desc = random.choice(descs)
                    break

            if desc:
                # Pluralize if multiple bboxes with this label
                lbl_count = bbox_counts.get(label, 1)
                desc = _pluralize_description(desc, lbl_count)
                additions.append(desc)
            else:
                # Generic fallback
                additions.append(f'Signal abnormality consistent with {label.lower()} is also noted.')

        if additions:
            new_reasoning = reasoning.rstrip('. ') + '. ' + ' '.join(additions)
            e['reasoning'] = new_reasoning
            count += 1

    return count




BRAIN_LABEL_TO_CHAR_KEY = {
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
    'paranasal sinus opacification': 'paranasal sinus opacification',
    'likely cysts': 'likely cysts',
}


def enrich_volume_detection_reasoning(qa):
    """Enrich volume-level detection reasoning to mention ALL ground-truth findings."""
    count = 0
    for e in qa:
        if e.get('task') != 'detection' or e.get('level') != 'volume':
            continue
        gt = e.get('ground_truth', {})
        if not isinstance(gt, dict):
            continue
        diagnoses = gt.get('final_diagnosis', [])
        if len(diagnoses) <= 1:
            continue

        reasoning = e.get('reasoning', '')
        reasoning_lower = reasoning.lower()
        modality = e.get('modality', '').upper()

        missing = []
        for diag in diagnoses:
            diag_lower = diag.lower()
            if diag_lower in reasoning_lower:
                continue
            key_words = [w for w in diag_lower.split() if len(w) > 3]
            if key_words and all(w in reasoning_lower for w in key_words):
                continue
            missing.append(diag)

        if not missing:
            continue

        additions = []
        for diag in missing:
            char_key = BRAIN_LABEL_TO_CHAR_KEY.get(diag.lower())
            if char_key is None:
                continue
            # Use FLAIR-specific descriptions when modality is FLAIR
            flair_key = char_key + '_flair'
            if 'FLAIR' in modality and flair_key in BRAIN_IMAGING_CHARACTERISTICS:
                descs = BRAIN_IMAGING_CHARACTERISTICS[flair_key]
            else:
                descs = BRAIN_IMAGING_CHARACTERISTICS.get(char_key)
            if descs:
                additions.append(random.choice(descs))
            else:
                additions.append(f'Additionally, {diag.lower()} is identified.')

        if additions:
            e['reasoning'] = reasoning.rstrip('. ') + '. ' + ' '.join(additions)
            count += 1

    return count


def add_classification_answer_letters(qa):
    """Add answer letter references (A), (B), etc. to classification answers."""
    count = 0
    for e in qa:
        if e.get('task') != 'classification':
            continue
        choices = e.get('choices', {})
        if not choices:
            continue
        answer = e.get('answer', '')
        # Skip if already has letter reference
        if re.search(r'\([A-D]\)', answer):
            continue

        gt = e.get('ground_truth', {})
        fd = gt.get('final_diagnosis', []) if isinstance(gt, dict) else []
        if not fd:
            # Try labels field (image QA)
            fd = gt.get('labels', []) if isinstance(gt, dict) else []

        # Find matching choice letters
        matched_letters = []
        for letter, choice_text in sorted(choices.items()):
            choice_lower = choice_text.lower()
            for diag in fd:
                diag_lower = diag.lower()
                # Match if diagnosis appears in choice or vice versa
                if diag_lower in choice_lower or choice_lower in diag_lower:
                    matched_letters.append(f'({letter})')
                    break
                # Partial match for common abbreviations
                diag_words = diag_lower.split()
                if len(diag_words) >= 2 and all(w in choice_lower for w in diag_words[:2]):
                    matched_letters.append(f'({letter})')
                    break

        if matched_letters:
            letter_str = '+'.join(matched_letters)
            # Add letter reference at the start of the answer
            if not answer.startswith('('):
                e['answer'] = f'{letter_str} {answer}'
                count += 1
    return count


def fix_enlarged_ventricles_circular(qa):
    """Fix circular reasoning: 'enlarged ventricles suggestive of enlarged ventricles'."""
    alternatives = [
        'ventriculomegaly with disproportionate ventricular enlargement',
        'ventricular enlargement consistent with ventriculomegaly',
        'abnormal ventricular dilation suggesting hydrocephalus or atrophy',
        'prominent ventricles with findings suggestive of ventriculomegaly',
        'ventricular prominence beyond expected limits for age',
        'dilated ventricles consistent with ventriculomegaly',
        'increased ventricular volume suggesting underlying pathology',
        'ventricular enlargement indicating possible hydrocephalus',
    ]
    count = 0
    pattern = re.compile(
        r'enlarged ventricles?\s+(?:which\s+is\s+)?suggestive\s+of\s+enlarged\s+ventricles?',
        re.IGNORECASE
    )
    for e in qa:
        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            if pattern.search(text):
                new_text = pattern.sub(random.choice(alternatives), text)
                e[field] = new_text
                count += 1
    return count




def add_counting_bbox_info(qa, vol_meta):
    """Add bbox listing to counting reasoning from volume metadata."""
    if not vol_meta:
        return 0
    count = 0
    for e in qa:
        if e.get('task') != 'counting':
            continue
        vol_id = e.get('volume_id', '')
        meta = vol_meta.get(vol_id)
        if not meta:
            continue

        reasoning = e.get('reasoning', '')
        # Skip if already has frame-level bbox listing
        if 'Frame' in reasoning and re.search(r'\[\d+,\s*\d+', reasoning):
            continue

        bbox_frames = get_volume_bbox_frames(meta)
        if not bbox_frames:
            continue

        # Build bbox listing
        all_bboxes = []
        lines = []
        for frame_num in sorted(bbox_frames.keys()):
            for bb in bbox_frames[frame_num]:
                bbox_str = f"[{bb['x']},{bb['y']},{bb['width']},{bb['height']}]"
                label = bb.get('label', 'finding')
                lines.append(f"Frame {frame_num}: {bbox_str} ({label})")
                all_bboxes.append(bb)

        total = len(all_bboxes)
        if total == 0:
            continue

        if total <= 15:
            listing = '; '.join(lines)
            bbox_info = f" Specific locations: {listing}."
        else:
            # Summary + full listing
            label_counts = Counter(bb.get('label', 'finding') for bb in all_bboxes)
            summary_parts = [f"{label} ({cnt})" for label, cnt in label_counts.most_common()]
            summary = ', '.join(summary_parts)
            listing = '; '.join(lines)
            bbox_info = f" Found {total} instances across {len(bbox_frames)} frames ({summary}): {listing}."

        e['reasoning'] = reasoning.rstrip() + bbox_info
        count += 1
    return count


def add_bbox_info_to_image_captioning(qa):
    """Add bbox info to image-level captioning reasoning using ground_truth.bboxes."""
    count = 0
    for e in qa:
        if e.get('task') != 'captioning':
            continue
        if e.get('level') != 'slice':
            continue

        gt = e.get('ground_truth', {})
        bboxes = gt.get('bboxes', []) if isinstance(gt, dict) else []
        if not bboxes:
            continue

        reasoning = e.get('reasoning', '')
        # Skip if already has bbox info
        if re.search(r'\[\d+,\s*\d+,\s*\d+,\s*\d+\]', reasoning):
            continue

        # Build bbox listing
        bbox_parts = []
        for bb in bboxes:
            label = bb.get('label', 'finding')
            bbox_str = f"[{bb['x']},{bb['y']},{bb['width']},{bb['height']}]"
            bbox_parts.append(f"{label} at {bbox_str}")

        if bbox_parts:
            # Group by label
            label_groups = defaultdict(list)
            for bb in bboxes:
                label = bb.get('label', 'finding')
                label_groups[label].append(f"[{bb['x']},{bb['y']},{bb['width']},{bb['height']}]")

            n_types = len(label_groups)
            n_total = len(bboxes)
            parts = []
            for label, coords in label_groups.items():
                parts.append(f"{label} at {', '.join(coords)}")
            listing = '; '.join(parts)
            bbox_info = f" There are {n_types} distinct types of findings with {n_total} total instances: {listing}."

            e['reasoning'] = reasoning.rstrip() + bbox_info
            count += 1
    return count


def remove_bbox_listing_from_classification(qa):
    """Remove 'There are N distinct types...' bbox listing from classification reasoning."""
    count = 0
    pattern = re.compile(
        r'\s*There are \d+ distinct types? of findings? with \d+ total instances?:[^.]*\.',
        re.IGNORECASE
    )
    for e in qa:
        if e.get('task') != 'classification':
            continue
        reasoning = e.get('reasoning', '')
        new_text = pattern.sub('', reasoning)
        if new_text != reasoning:
            e['reasoning'] = new_text.strip()
            count += 1
    return count


def add_volume_captioning_bbox_info(qa, vol_meta):
    """Add narrative bbox info to volume captioning reasoning."""
    if not vol_meta:
        return 0
    count = 0
    for e in qa:
        if e.get('task') != 'captioning':
            continue
        if e.get('level') == 'slice':
            continue

        vol_id = e.get('volume_id', '')
        meta = vol_meta.get(vol_id)
        if not meta:
            continue

        reasoning = e.get('reasoning', '')
        # Skip if already has frame-level bbox info
        if 'Frame' in reasoning and re.search(r'\[\d+,\s*\d+', reasoning):
            continue

        bbox_frames = get_volume_bbox_frames(meta)
        if not bbox_frames:
            continue

        # Pick representative frames (up to 3)
        frame_nums = sorted(bbox_frames.keys())
        if len(frame_nums) > 3:
            # Pick start, middle, end
            sample_frames = [frame_nums[0], frame_nums[len(frame_nums)//2], frame_nums[-1]]
        else:
            sample_frames = frame_nums

        parts = []
        for fn in sample_frames:
            bbs = bbox_frames[fn]
            bb = bbs[0]  # First bbox in frame
            label = bb.get('label', 'finding')
            bbox_str = f"[{bb['x']},{bb['y']},{bb['width']},{bb['height']}]"
            parts.append(f"In frame {fn}, a {label.lower()} is at {bbox_str}")

        total_bboxes = sum(len(bbs) for bbs in bbox_frames.values())
        narrative = '. '.join(parts)
        bbox_info = f" The volume shows findings across {len(bbox_frames)} frames with {total_bboxes} total annotations. {narrative}."

        e['reasoning'] = reasoning.rstrip() + bbox_info
        count += 1
    return count




def diversify_detection_reasoning(qa):
    """Diversify detection task reasoning openers for volume QA."""
    openers = [
        'The volume reveals',
        'Upon examination,',
        'Analysis of the volume demonstrates',
        'Review of the imaging data shows',
        'The volume contains',
        'Examination reveals',
        'The imaging demonstrates',
        'Assessment of the volume identifies',
    ]
    # Pattern to match common repetitive openers
    pattern = re.compile(
        r'^(?:The volume (?:shows|reveals|demonstrates|contains)|'
        r'Yes,?\s*(?:the volume|there are)|'
        r'There (?:are|is))\s+',
        re.IGNORECASE
    )
    count = 0
    for e in qa:
        if e.get('task') != 'detection':
            continue
        if e.get('level') == 'slice':
            continue
        reasoning = e.get('reasoning', '')
        m = pattern.match(reasoning)
        if m:
            rest = reasoning[m.end():]
            new_opener = random.choice(openers)
            e['reasoning'] = f'{new_opener} {rest}'
            count += 1
    return count


def diversify_localization(qa):
    """Diversify localization task phrasing for volume QA."""
    verbs = [
        'is located in', 'is identified in', 'is situated in',
        'appears in', 'is observed in', 'is noted in',
        'is present in', 'is seen in',
    ]
    # Match "The {finding} is located in" pattern
    pattern = re.compile(
        r'(The \w[\w\s]*?)\s+is (?:located|found|identified|situated|observed|noted|present|seen) in\b',
        re.IGNORECASE
    )
    count = 0
    for e in qa:
        if e.get('task') != 'localization':
            continue
        if e.get('level') == 'slice':
            continue

        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            matches = list(pattern.finditer(text))
            if not matches:
                continue

            new_text = text
            offset = 0
            for m in matches:
                verb = random.choice(verbs)
                subject = m.group(1)
                replacement = f'{subject} {verb}'
                start = m.start() + offset
                end = m.end() + offset
                new_text = new_text[:start] + replacement + new_text[end:]
                offset += len(replacement) - (m.end() - m.start())

            if new_text != text:
                e[field] = new_text
                count += 1
    return count


def diversify_counting_reasoning(qa):
    """Diversify counting task reasoning openers for volume QA."""
    templates = [
        'The volume contains {n} distinct categories of findings',
        'A total of {n} types of findings are identified',
        'Examination reveals {n} categories of abnormalities',
        'The volume demonstrates {n} distinct types of findings',
        'Analysis identifies {n} categories of findings',
        'Review shows {n} types of abnormalities',
        'The imaging contains {n} distinct categories',
        'Assessment reveals {n} types of findings',
    ]
    pattern = re.compile(
        r'^(?:The volume (?:contains|shows|demonstrates)|'
        r'There (?:are|is)|'
        r'A total of)\s+(\d+)\s+(?:distinct\s+)?(?:categories?|types?)\s+(?:of\s+)?(?:findings?|abnormalit)',
        re.IGNORECASE
    )
    count = 0
    for e in qa:
        if e.get('task') != 'counting':
            continue
        if e.get('level') == 'slice':
            continue
        reasoning = e.get('reasoning', '')
        m = pattern.match(reasoning)
        if m:
            n = m.group(1)
            rest = reasoning[m.end():]
            template = random.choice(templates).format(n=n)
            e['reasoning'] = template + rest
            count += 1
    return count




def remove_global_labels(qa):
    """Remove global diagnosis labels from final_diagnosis field."""
    count = 0
    global_labels = {'Normal for age', 'Motion artifact', 'Normal', 'Normal brain',
                     'Normal variant', 'Normal finding', 'Normal for age changes'}
    for e in qa:
        gt = e.get('ground_truth', {})
        if not isinstance(gt, dict):
            continue
        fd = gt.get('final_diagnosis', [])
        if isinstance(fd, list):
            new_fd = [d for d in fd if d not in global_labels]
            if len(new_fd) != len(fd):
                gt['final_diagnosis'] = new_fd
                count += 1
    return count


def add_clinical_descriptions_to_captioning(qa):
    """Add clinical diagnosis descriptions to captioning reasoning."""
    descriptions = {
        'encephalomalacia': [
            'Encephalomalacia represents localized softening and loss of brain tissue from irreversible damage due to stroke, trauma, infection, or hemorrhage.',
            'This finding indicates encephalomalacia, characterized by focal brain tissue destruction with resulting volume loss.',
            'The appearance is consistent with encephalomalacia, reflecting chronic parenchymal injury and tissue necrosis.',
            'Encephalomalacia is identified, representing permanent brain tissue damage with cystic or gliotic change.',
            'The finding suggests encephalomalacia from prior irreversible parenchymal injury.',
            'This is consistent with encephalomalacia, indicating chronic focal brain tissue loss.',
            'Encephalomalacia is present, reflecting end-stage tissue injury with associated volume loss.',
            'The imaging demonstrates encephalomalacia, a sequela of prior parenchymal destruction.',
        ],
        'craniotomy': [
            'This finding indicates a prior surgical opening into the skull, consistent with a craniotomy.',
            'A craniotomy defect is identified, indicating previous neurosurgical intervention.',
            'The appearance is consistent with a prior craniotomy, showing surgical changes in the calvarium.',
            'Evidence of a prior craniotomy is seen, reflecting previous surgical access to the intracranial contents.',
            'A craniotomy site is identified with associated post-surgical changes.',
            'The finding represents a prior craniotomy with expected post-operative calvaria changes.',
            'Post-craniotomy changes are present, indicating previous neurosurgical procedure.',
            'The imaging shows a craniotomy defect from prior surgical intervention.',
        ],
        'resection cavity': [
            'A resection cavity is identified, representing the space left behind after surgical removal of a tumor or lesion.',
            'The finding is consistent with a post-surgical resection cavity.',
            'A resection cavity is present, indicating prior surgical excision of intracranial pathology.',
            'The appearance represents a resection cavity from previous neurosurgical tumor removal.',
            'A post-surgical resection cavity is identified with expected surrounding changes.',
            'The finding suggests a resection cavity, the expected result of prior lesion excision.',
            'A resection cavity is seen, consistent with prior neurosurgical intervention.',
            'The imaging demonstrates a resection cavity from previous surgical removal of pathology.',
        ],
        'extra-axial mass': [
            'An extra-axial mass is identified, located outside the brain parenchyma but within the skull.',
            'The finding represents an extra-axial mass displacing the adjacent brain surface.',
            'An extra-axial lesion is present, arising outside the brain parenchyma.',
            'The appearance is consistent with an extra-axial mass in the intracranial compartment.',
            'An extra-axial mass is seen, characterized by its location outside the brain substance.',
            'The finding suggests an extra-axial neoplasm based on its location and morphology.',
            'An extra-axial mass lesion is identified with associated mass effect.',
            'The imaging demonstrates an extra-axial mass with typical features of a dural-based lesion.',
        ],
        'extra-axial collection': [
            'An extra-axial collection is identified, representing fluid accumulation inside the skull but outside the brain parenchyma.',
            'The finding represents an extra-axial fluid collection.',
            'An extra-axial collection is present, indicating fluid between the brain and skull.',
            'The appearance is consistent with an extra-axial collection, possibly subdural or epidural.',
            'An extra-axial fluid collection is seen adjacent to the brain surface.',
            'The finding suggests an extra-axial collection with characteristic location and signal.',
            'An extra-axial collection is identified between the brain and calvarium.',
            'The imaging demonstrates an extra-axial fluid collection.',
        ],
        'edema': [
            'Edema is identified, representing abnormal fluid accumulation in the brain tissue causing swelling.',
            'The finding is consistent with cerebral edema, indicating increased water content in the parenchyma.',
            'Brain edema is present, characterized by increased signal reflecting excess tissue fluid.',
            'The appearance suggests vasogenic edema with increased extracellular fluid.',
            'Edema is seen in the brain parenchyma, causing local tissue swelling.',
            'The finding represents cerebral edema with associated parenchymal swelling.',
            'Parenchymal edema is identified, consistent with increased tissue water content.',
            'The imaging demonstrates brain edema with characteristic signal changes.',
        ],
        'lacunar infarct': [
            'A lacunar infarct is identified, representing a small stroke from occlusion of a deep perforating artery.',
            'The finding is consistent with a lacunar infarct in the deep brain structures.',
            'A lacunar infarct is present, indicating small vessel ischemic disease.',
            'The appearance suggests a lacunar infarct from small vessel occlusion.',
            'A lacunar infarct is seen, characteristic of deep perforator artery disease.',
            'The finding represents a lacunar infarct, a small deep brain infarction.',
            'A small lacunar infarct is identified in the expected distribution of perforating arteries.',
            'The imaging demonstrates a lacunar infarct consistent with small vessel disease.',
        ],
        'dural thickening': [
            'Dural thickening is identified, representing an inflammatory or fibrotic response of the dura mater.',
            'The finding is consistent with pachymeningeal thickening.',
            'Dural thickening is present, which may relate to tumors, infections, or autoimmune conditions.',
            'The appearance suggests dural thickening with abnormal meningeal signal.',
            'Dural thickening is seen, indicating an abnormal pachymeningeal process.',
            'The finding represents dural thickening, a sign of meningeal pathology.',
            'Abnormal dural thickening is identified along the convexity.',
            'The imaging demonstrates dural thickening consistent with a pachymeningeal process.',
        ],
        'enlarged ventricles': [
            'Enlarged ventricles are identified, representing ventriculomegaly with increased CSF space.',
            'The finding is consistent with ventriculomegaly, an abnormal increase in ventricular size.',
            'Enlarged ventricles are present, indicating an increase in the brain fluid-filled cavities.',
            'The appearance suggests ventriculomegaly with disproportionate ventricular enlargement.',
            'Ventricular enlargement is seen, consistent with ventriculomegaly.',
            'The finding represents enlarged ventricles, which may indicate hydrocephalus or atrophy.',
            'Ventriculomegaly is identified with enlarged lateral ventricles.',
            'The imaging demonstrates ventricular enlargement beyond normal limits.',
        ],
        'nonspecific white matter lesion': [
            'Nonspecific white matter lesions are identified, which can arise from aging, chronic migraines, or small vessel disease.',
            'The findings are consistent with nonspecific white matter disease.',
            'Nonspecific white matter hyperintensities are present, a common incidental finding.',
            'The appearance suggests nonspecific white matter lesions of uncertain clinical significance.',
            'Scattered white matter lesions are seen, which rarely point to a single disease.',
            'The finding represents nonspecific white matter changes.',
            'Nonspecific white matter signal abnormalities are identified.',
            'The imaging demonstrates nonspecific white matter lesions.',
        ],
        'mass': [
            'A mass is identified, representing an abnormal growth within the brain.',
            'The finding is consistent with an intracranial mass lesion.',
            'A brain mass is present, characterized by its space-occupying behavior.',
            'The appearance suggests a brain tumor based on its morphology and enhancement pattern.',
            'An intracranial mass is seen with associated mass effect.',
            'The finding represents a brain mass requiring further characterization.',
            'A mass lesion is identified in the brain parenchyma.',
            'The imaging demonstrates an intracranial mass with typical neoplastic features.',
        ],
        'pineal cyst': [
            'A pineal cyst is identified, a benign fluid-filled sac in the pineal gland.',
            'The finding is consistent with a pineal cyst, typically an incidental finding.',
            'A pineal cyst is present, a common benign lesion of the pineal gland.',
            'The appearance suggests a simple pineal cyst without complicating features.',
            'A pineal cyst is seen, representing a benign developmental lesion.',
            'The finding represents a pineal cyst, generally of no clinical significance.',
            'A benign-appearing pineal cyst is identified.',
            'The imaging demonstrates a pineal cyst with typical signal characteristics.',
        ],
        'craniectomy': [
            'A craniectomy defect is identified, indicating a portion of skull was surgically removed.',
            'The finding is consistent with a prior craniectomy.',
            'A craniectomy is present, indicating emergency neurosurgical skull removal.',
            'The appearance suggests a prior craniectomy with absent calvaria.',
            'A craniectomy defect is seen, reflecting prior neurosurgical intervention.',
            'The finding represents a craniectomy, where skull bone was removed.',
            'Post-craniectomy changes are identified with absent bone flap.',
            'The imaging demonstrates a craniectomy defect.',
        ],
        'craniectomy with cranioplasty': [
            'A craniectomy with cranioplasty is identified, showing skull removal followed by reconstruction.',
            'The finding is consistent with prior craniectomy and subsequent cranioplasty repair.',
            'Post-craniectomy cranioplasty changes are present, indicating skull reconstruction.',
            'The appearance suggests a prior craniectomy with cranioplasty reconstruction material.',
            'A cranioplasty repair is seen at the site of prior craniectomy.',
            'The finding represents craniectomy followed by cranioplasty reconstruction.',
            'Post-surgical cranioplasty changes are identified.',
            'The imaging demonstrates craniectomy with cranioplasty repair.',
        ],
    }

    count = 0
    for e in qa:
        if e.get('task') != 'captioning':
            continue
        reasoning = e.get('reasoning', '')
        gt = e.get('ground_truth', {})
        if isinstance(gt, dict):
            fd = gt.get('final_diagnosis', gt.get('diagnosis', []))
        else:
            fd = []
        gt_str = ' '.join(str(d).lower() for d in fd) if fd else str(gt).lower()

        for diag, desc_list in descriptions.items():
            if diag in gt_str or diag.replace(' ', '') in gt_str.replace(' ', ''):
                desc = random.choice(desc_list)
                if desc.split('.')[0] not in reasoning:
                    e['reasoning'] = desc + ' ' + reasoning
                    count += 1
                break
    return count


def diversify_nonspecific_wml_bilateral(qa):
    """Diversify hyperintense phrases for bilateral nonspecific WML entries."""
    hyper_phrases = [
        'There appear to be multiple hyperintense lesions.',
        'Multiple hyperintense foci are identified.',
        'Several hyperintense lesions are noted.',
        'Multiple areas of hyperintense signal are observed.',
        'Scattered hyperintense lesions are present.',
        'Multiple foci of hyperintense signal abnormality are seen.',
        'Several areas of increased signal intensity are identified.',
        'Multiple hyperintense signal abnormalities are noted.',
    ]
    reasoning_phrases = [
        'The abnormal region is in the bilateral hemispheres, which is suggestive of a nonspecific white matter lesion.',
        'The findings are in the bilateral hemispheres, which is suggestive of nonspecific white matter lesions.',
        'The areas of concern are located in the bilateral hemispheres, suggestive of nonspecific white matter lesions.',
        'The lesions are situated in the bilateral hemispheres, consistent with nonspecific white matter lesions.',
        'The observed abnormality is in the bilateral hemispheres, which is suggestive of nonspecific white matter lesion.',
        'The signal abnormalities are in the bilateral hemispheres, suggestive of nonspecific white matter disease.',
        'The identified findings span the bilateral hemispheres, which is suggestive of nonspecific white matter lesions.',
        'The hyperintense foci are distributed across the bilateral hemispheres, suggestive of nonspecific white matter lesions.',
    ]
    count = 0
    for e in qa:
        if e.get('task') != 'localization':
            continue
        gt = e.get('ground_truth', {})
        if isinstance(gt, dict):
            fd = gt.get('final_diagnosis', gt.get('diagnosis', []))
        else:
            fd = []
        gt_str = ' '.join(str(d).lower() for d in fd) if fd else str(gt).lower()
        if 'nonspecific white matter' not in gt_str and 'white matter lesion' not in gt_str:
            continue
        answer = e.get('answer', '')
        if 'bilateral' not in answer.lower():
            continue
        e['answer'] = random.choice(hyper_phrases) + ' ' + answer
        e['reasoning'] = random.choice(reasoning_phrases)
        count += 1
    return count


# Main

def process_file(path, split, file_type, vol_meta=None, dry_run=False):
    """Process a single brain QA file."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(path)}")
    print(f"{'='*60}")

    qa = load_json(path)
    if not dry_run:
        backup(path)

    results = {}

    # ── Section 1: Hemisphere & Location Fixes ──
    results['periventricular'] = fix_periventricular_to_hemisphere(qa)
    results['midline'] = fix_midline_references(qa)
    results['lobe_names'] = remove_lobe_names(qa)
    results['anatomical'] = remove_anatomical_structures(qa)
    results['hemisphere_mismatch'] = fix_hemisphere_mismatches(qa)
    results['white_matter_reasoning'] = fix_white_matter_reasoning(qa)
    results['single_bbox_bilateral'] = fix_single_bbox_bilateral(qa)
    results['artifact_spanning'] = fix_artifact_spanning(qa)
    results['duplicate_hemisphere'] = remove_duplicate_hemisphere_phrases(qa)

    # ── Section 2: Text & Grammar Fixes ──
    if file_type == 'image':
        results['patient_history'] = remove_patient_history_phrases(qa)
    results['grammar'] = fix_grammar(qa)
    results['frame_refs'] = fix_frame_references(qa)
    results['trailing'] = fix_trailing_sentences(qa)
    results['false_normal'] = remove_false_normal(qa)

    # ── Section 3: Classification Fixes ──
    results['classification_letters'] = add_classification_answer_letters(qa)
    results['enlarged_ventricles_circular'] = fix_enlarged_ventricles_circular(qa)

    # ── Section 4: BBox Info Enrichment ──
    if file_type == 'volume' and vol_meta:
        results['counting_bbox_info'] = add_counting_bbox_info(qa, vol_meta)
        results['vol_captioning_bbox'] = add_volume_captioning_bbox_info(qa, vol_meta)

    if file_type == 'image':
        results['detection_enrich'] = enrich_detection_reasoning(qa)
        results['img_captioning_bbox'] = add_bbox_info_to_image_captioning(qa)
        results['remove_class_bbox'] = remove_bbox_listing_from_classification(qa)

    # ── Section 5: Reasoning Diversity (volume QA only) ──
    if file_type == 'volume':
        results['detect_diversity'] = diversify_detection_reasoning(qa)
        results['local_diversity'] = diversify_localization(qa)
        results['count_diversity'] = diversify_counting_reasoning(qa)

    # ── Section 6: Clinical Descriptions & Diagnosis ──
    results['clinical_descriptions'] = add_clinical_descriptions_to_captioning(qa)
    results['wml_bilateral'] = diversify_nonspecific_wml_bilateral(qa)
    results['global_labels'] = remove_global_labels(qa)

    total = sum(results.values())
    for name, count in sorted(results.items()):
        if count > 0:
            print(f"  {name}: {count}")
    print(f"  TOTAL: {total}")

    if not dry_run and total > 0:
        save_json(path, qa)
        print(f"  Saved")
    elif dry_run:
        print(f"  (dry run - not saved)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Clean brain QA data')
    parser.add_argument('--split', choices=['train', 'val'], help='Process only this split')
    parser.add_argument('--type', choices=['volume', 'image'], help='Process only this type')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without writing')
    parser.add_argument('--metadata-dir', default=None,
                        help='Path to brain volume metadata directory')
    args = parser.parse_args()

    metadata_dir = args.metadata_dir or DEFAULT_METADATA_DIR

    splits = [args.split] if args.split else ['val', 'train']
    types = [args.type] if args.type else ['volume', 'image']

    for split in splits:
        # Load volume metadata (for enrichment steps)
        vol_meta = load_brain_volume_metadata(split, metadata_dir)
        if vol_meta:
            print(f"Loaded {len(vol_meta)} volume metadata entries for {split}")

        for file_type in types:
            filename = f"gpt4o_brain_{split}_{file_type}_qa_{file_type}_qa_pairs.json"
            path = os.path.join(BRAIN_DIR, filename)
            if os.path.exists(path):
                process_file(path, split, file_type, vol_meta=vol_meta, dry_run=args.dry_run)
            else:
                print(f"Skipping {filename} (not found)")


if __name__ == '__main__':
    main()
