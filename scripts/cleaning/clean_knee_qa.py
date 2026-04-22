#!/usr/bin/env python3
"""Post-processing cleanup for GPT-4o knee QA data."""

import json
import re
import os
import random
import argparse
import shutil
from collections import Counter, defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNEE_DIR = os.path.join(BASE_DIR, "knee")

# Knee image is 320x320
# Midpoint for left/right side determination
MIDPOINT_X = 160
IMG_SIZE = 320

# Fibula annotations file (maps volume_id -> 'left' or 'right')
# 'left' means fibula is on the left side of the image -> left = lateral
# 'right' means fibula is on the right side -> right = lateral
FIBULA_PATH = os.path.join(KNEE_DIR, "knee_fibula_lateral_side.json")

# Volume metadata directory (for enrichment steps)
DEFAULT_METADATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data_processing", "knee"))

# Clinician-verified volume cutoffs (don't modify these)
VERIFIED_CUTOFFS = {
    'val': 'file1001298',   # First 90 val volumes verified
    'train': None,          # First 10 train volumes verified (handled by index)
}
TRAIN_VERIFIED_COUNT = 10


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

def load_fibula_annotations():
    """Load fibula lateral side annotations."""
    if not os.path.exists(FIBULA_PATH):
        print(f"WARNING: Fibula annotations not found at {FIBULA_PATH}")
        print("  Run the fibula annotator first to create this file.")
        return {}
    return load_json(FIBULA_PATH)

def get_verified_volumes(qa, split):
    """Get set of clinician-verified volume IDs that should not be modified."""
    all_vols = sorted(set(e['volume_id'] for e in qa))
    verified = set()

    if split == 'val':
        cutoff = VERIFIED_CUTOFFS.get('val')
        if cutoff:
            for v in all_vols:
                verified.add(v)
                if v == cutoff:
                    break
    elif split == 'train':
        for v in all_vols[:TRAIN_VERIFIED_COUNT]:
            verified.add(v)

    return verified



def load_knee_volume_metadata(split, metadata_dir=None):
    """Load knee volume metadata."""
    mdir = metadata_dir or DEFAULT_METADATA_DIR
    path = os.path.join(mdir, f"knee_{split}_volumes.json")
    if not os.path.exists(path):
        print(f"  WARNING: Metadata not found at {path}")
        return {}
    return load_json(path)


def get_volume_bbox_frames(vol_data):
    """Get bboxes organized by frame from volume metadata."""
    frames = {}
    for s in vol_data.get('slices', []):
        bbs = s.get('bounding_boxes', [])
        if bbs:
            frames[s['slice']] = bbs
    return frames




def swap_terms(text, direction):
    """Swap medial/lateral descriptive terms."""
    if direction == 'med_to_lat':
        pairs = [
            ('medial femoral condyle', 'lateral femoral condyle'),
            ('medial tibial plateau', 'lateral tibial plateau'),
            ('medial compartment', 'lateral compartment'),
            ('medial meniscus', 'lateral meniscus'),
            ('medial menisci', 'lateral menisci'),
            ('medial retinaculum', 'lateral retinaculum'),
            ('medial condyle', 'lateral condyle'),
            ('medial aspect', 'lateral aspect'),
            ('medial region', 'lateral region'),
            ('medial side', 'lateral side'),
            ('medial knee', 'lateral knee'),
            ('medial joint space', 'lateral joint space'),
            ('medial soft tissue', 'lateral soft tissue'),
        ]
    else:  # lat_to_med
        pairs = [
            ('lateral femoral condyle', 'medial femoral condyle'),
            ('lateral tibial plateau', 'medial tibial plateau'),
            ('lateral compartment', 'medial compartment'),
            ('lateral meniscus', 'medial meniscus'),
            ('lateral menisci', 'medial menisci'),
            ('lateral retinaculum', 'medial retinaculum'),
            ('lateral condyle', 'medial condyle'),
            ('lateral aspect', 'medial aspect'),
            ('lateral region', 'medial region'),
            ('lateral side', 'medial side'),
            ('lateral knee', 'medial knee'),
            ('lateral joint space', 'medial joint space'),
            ('lateral soft tissue', 'medial soft tissue'),
        ]
    result = text
    for old, new in pairs:
        result = result.replace(old, new)
        # Handle capitalized versions
        result = result.replace(old[0].upper() + old[1:], new[0].upper() + new[1:])
    return result


def is_collateral_ligament_segment(seg_lower):
    """Check if a text segment is about collateral ligaments (should not be swapped)."""
    return ('collateral ligament' in seg_lower or
            'mcl' in seg_lower or 'lcl' in seg_lower)


def fix_segments_with_bboxes(text, lateral_side):
    """Fix medial/lateral labels in text by checking bbox positions against fibula side."""
    segments = re.split(r'(?<=\.)\s+(?=[A-Z])', text)
    new_segments = []
    changed = False

    for seg in segments:
        seg_lower = seg.lower()
        has_medial_only = 'medial' in seg_lower and 'lateral' not in seg_lower
        has_lateral_only = 'lateral' in seg_lower and 'medial' not in seg_lower

        if not has_medial_only and not has_lateral_only:
            new_segments.append(seg)
            continue

        # Skip collateral ligament segments (MCL/LCL are disease labels)
        if is_collateral_ligament_segment(seg_lower):
            new_segments.append(seg)
            continue

        x_values = get_bbox_x_values(seg)
        if not x_values:
            new_segments.append(seg)
            continue

        avg_x = sum(x_values) / len(x_values)
        bbox_side = 'left' if avg_x < MIDPOINT_X else 'right'

        if has_medial_only:
            # Medial bboxes should be on the OPPOSITE side from fibula
            expected = 'right' if lateral_side == 'left' else 'left'
            if bbox_side != expected:
                new_segments.append(swap_terms(seg, 'med_to_lat'))
                changed = True
                continue
        elif has_lateral_only:
            # Lateral bboxes should be on the SAME side as fibula
            if bbox_side != lateral_side:
                new_segments.append(swap_terms(seg, 'lat_to_med'))
                changed = True
                continue

        new_segments.append(seg)

    return ' '.join(new_segments), changed


def fix_text_by_finding_map(text, finding_sides):
    """Fix medial/lateral in text using finding->side mapping."""
    sentences = re.split(r'(?<=\.)\s+', text)
    new_sentences = []
    changed = False

    for sent in sentences:
        sent_lower = sent.lower()
        has_medial_only = 'medial' in sent_lower and 'lateral' not in sent_lower
        has_lateral_only = 'lateral' in sent_lower and 'medial' not in sent_lower

        if not has_medial_only and not has_lateral_only:
            new_sentences.append(sent)
            continue

        if is_collateral_ligament_segment(sent_lower):
            new_sentences.append(sent)
            continue

        mentioned_side = 'medial' if has_medial_only else 'lateral'
        matched = False
        for finding, correct_side in finding_sides.items():
            if finding in sent_lower or (len(finding.split()) > 1 and finding.split()[0] in sent_lower):
                if mentioned_side != correct_side:
                    direction = 'med_to_lat' if mentioned_side == 'medial' else 'lat_to_med'
                    new_sentences.append(swap_terms(sent, direction))
                    changed = True
                    matched = True
                else:
                    new_sentences.append(sent)
                    matched = True
                break
        if not matched:
            new_sentences.append(sent)

    return ' '.join(new_sentences), changed


def build_finding_sides_map(qa, fibula, verified_vols):
    """Build per-volume finding->anatomical side mapping from localization answers."""
    vol_finding_sides = {}
    for e in qa:
        if e.get('task') != 'localization':
            continue
        vol_id = e['volume_id']
        if vol_id in verified_vols:
            continue
        lateral_side = fibula.get(vol_id)
        if not lateral_side:
            continue

        answer = e.get('answer', '')
        segments = re.split(r'(?<=\.)\s+(?=[A-Z])', answer)
        finding_sides = {}

        for seg in segments:
            x_values = get_bbox_x_values(seg)
            if not x_values:
                continue
            avg_x = sum(x_values) / len(x_values)
            bbox_side = 'left' if avg_x < MIDPOINT_X else 'right'
            anat_side = 'lateral' if bbox_side == lateral_side else 'medial'

            seg_lower = seg.lower()
            for finding in ['displaced meniscal', 'meniscus tear', 'meniscus', 'cartilage',
                            'subchondral edema', 'bone', 'periarticular cyst', 'periarticular',
                            'effusion', 'retinaculum', 'soft tissue',
                            'contusion', 'fracture', 'dislocation']:
                if finding in seg_lower:
                    finding_sides[finding] = anat_side
                    break

        if finding_sides:
            vol_finding_sides[vol_id] = finding_sides

    return vol_finding_sides


def fix_medial_lateral(qa, fibula, verified_vols):
    """Fix medial/lateral labels based on fibula annotations."""
    fix_counts = Counter()

    # Step 1: Fix localization answers (have bboxes)
    for i, e in enumerate(qa):
        if e.get('task') != 'localization':
            continue
        vol_id = e['volume_id']
        if vol_id in verified_vols:
            continue
        lateral_side = fibula.get(vol_id)
        if not lateral_side:
            continue

        new_answer, changed = fix_segments_with_bboxes(e['answer'], lateral_side)
        if changed:
            qa[i]['answer'] = new_answer
            fix_counts[('localization', 'answer')] += 1

    # Step 2: Build finding->side map from fixed localization answers
    vol_finding_sides = build_finding_sides_map(qa, fibula, verified_vols)

    # Step 3: Fix localization reasoning
    for i, e in enumerate(qa):
        if e.get('task') != 'localization':
            continue
        vol_id = e['volume_id']
        if vol_id in verified_vols or vol_id not in vol_finding_sides:
            continue

        lateral_side = fibula.get(vol_id)
        reasoning = e.get('reasoning', '')
        if not reasoning or ('medial' not in reasoning.lower() and 'lateral' not in reasoning.lower()):
            continue

        # Try bbox-based fix first
        new_reasoning, changed = fix_segments_with_bboxes(reasoning, lateral_side)
        if changed:
            qa[i]['reasoning'] = new_reasoning
            fix_counts[('localization', 'reasoning')] += 1
        else:
            # Try finding-map-based fix
            new_reasoning, changed = fix_text_by_finding_map(reasoning, vol_finding_sides[vol_id])
            if changed:
                qa[i]['reasoning'] = new_reasoning
                fix_counts[('localization', 'reasoning')] += 1

    # Step 4: Fix other tasks (captioning, classification, detection) - NOT diagnosis
    for i, e in enumerate(qa):
        task = e.get('task', '')
        if task in ('localization', 'diagnosis'):
            continue
        vol_id = e['volume_id']
        if vol_id in verified_vols or vol_id not in vol_finding_sides:
            continue

        lateral_side = fibula.get(vol_id)
        finding_sides = vol_finding_sides[vol_id]

        for field in ['answer', 'reasoning']:
            text = e.get(field, '')
            if not text or ('medial' not in text.lower() and 'lateral' not in text.lower()):
                continue

            # Try bbox-based fix first
            new_text, changed = fix_segments_with_bboxes(text, lateral_side)
            if changed:
                qa[i][field] = new_text
                fix_counts[(task, field)] += 1
            else:
                # Try finding-map-based fix
                new_text, changed = fix_text_by_finding_map(text, finding_sides)
                if changed:
                    qa[i][field] = new_text
                    fix_counts[(task, field)] += 1

    return fix_counts




def fix_grammar(qa):
    """Fix common grammar issues."""
    count = 0
    fixes = [
        ('unremarkable', 'normal'),
        ('pathological findings', 'findings'),
        ('categoryies', 'categories'),
        ('1 types', '1 type'),
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
    """Remove trailing incomplete sentences."""
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




# Imaging characteristic descriptions for knee findings (no bboxes)
KNEE_IMAGING_CHARACTERISTICS = {
    # --- User-provided descriptions (diversified into 8 variations) ---
    'meniscus tear': [
        'High intrameniscal signal extending to the articular surface is noted, consistent with a meniscus tear.',
        'Intrameniscal high signal reaching the articular surface is identified, suggestive of a meniscal tear.',
        'High signal within the meniscus extending to the articular surface is seen, consistent with a tear.',
        'Intrameniscal signal communicating with the articular surface is noted, indicating a meniscus tear.',
        'High signal intensity within the meniscus reaching the articular surface is observed, consistent with a meniscal tear.',
        'An area of high intrameniscal signal is identified extending to the articular surface, suggestive of a tear.',
        'Abnormal high signal within the meniscus is seen communicating with the articular surface, consistent with a tear.',
        'High intrameniscal signal that extends to the articular surface is noted, representing a meniscus tear.',
    ],
    'acl': [
        'Ligament discontinuity with fiber orientation parallel to the tibial plateau is noted, consistent with an ACL injury.',
        'The ACL demonstrates discontinuity with fibers oriented parallel to the tibial plateau, suggestive of a tear.',
        'Abnormal fiber orientation parallel to the tibial plateau with ligament discontinuity is identified, consistent with an ACL injury.',
        'ACL discontinuity is noted with fibers showing orientation parallel to the tibial plateau, indicating a tear.',
        'Disruption of the ACL with fiber orientation parallel to the tibial plateau is observed, consistent with ligament injury.',
        'The anterior cruciate ligament shows discontinuity and abnormal fiber orientation parallel to the tibial plateau.',
        'Ligament fiber discontinuity with orientation parallel to the tibial plateau is seen, suggestive of an ACL tear.',
        'The ACL fibers demonstrate discontinuity and orientation parallel to the tibial plateau, consistent with a tear.',
    ],
    'mcl': [
        'High signal superficial to the medial collateral ligament is noted, consistent with an MCL sprain.',
        'Signal abnormality superficial to the MCL is identified, suggestive of a ligament sprain.',
        'High signal intensity is seen superficial to the medial collateral ligament, consistent with an MCL injury.',
        'Superficial high signal along the medial collateral ligament is observed, indicating an MCL sprain.',
        'Abnormal high signal superficial to the MCL is noted, consistent with a medial collateral ligament sprain.',
        'High signal is identified superficial to the medial collateral ligament, suggestive of a sprain.',
        'Signal increase superficial to the MCL is seen, consistent with a medial collateral ligament injury.',
        'The medial collateral ligament demonstrates high signal superficially, consistent with an MCL sprain.',
    ],
    'pcl': [
        'A swollen and thickened posterior cruciate ligament with increased internal signal is noted, consistent with a PCL injury.',
        'The PCL appears swollen and thickened with increased internal signal, suggestive of a sprain.',
        'Swelling and thickening of the posterior cruciate ligament with increased internal signal is identified, consistent with a PCL injury.',
        'The PCL demonstrates thickening and swelling with increased internal signal, indicating ligament injury.',
        'A thickened and swollen PCL with abnormally increased internal signal is observed, consistent with a sprain.',
        'Increased internal signal within a swollen and thickened posterior cruciate ligament is noted, suggestive of injury.',
        'The posterior cruciate ligament is swollen and thickened with increased internal signal, consistent with a PCL sprain.',
        'Swelling of the PCL with increased internal signal and ligament thickening is seen, consistent with a PCL injury.',
    ],
    'joint effusion': [
        'High signal intensity with increased joint distension is noted, consistent with a joint effusion.',
        'A bright fluid band with increased distension of the joint capsule is seen, indicating joint effusion.',
        'Increased joint distension with high signal intensity is identified, consistent with an effusion.',
        'A bright fluid band with high signal intensity is observed within the joint space, consistent with effusion.',
        'High signal intensity fluid with distension of the joint capsule is noted, representing a joint effusion.',
        'Increased distension of the joint with bright fluid signal is seen, suggestive of a joint effusion.',
        'A high signal intensity fluid band with joint distension is identified, consistent with effusion.',
        'Joint distension with a bright fluid band and high signal intensity is observed, consistent with a joint effusion.',
    ],
    'periarticular cyst': [
        'A well-defined smooth-walled fluid-filled mass with homogeneously high signal is noted, consistent with a periarticular cyst.',
        'A smooth-walled fluid-filled mass with homogeneous high signal is identified adjacent to the joint, representing a periarticular cyst.',
        'A well-defined fluid-filled mass with smooth walls and homogeneously high signal is seen, consistent with a periarticular cyst.',
        'A homogeneously high-signal fluid-filled mass with well-defined smooth walls is observed, suggestive of a periarticular cyst.',
        'A smooth-walled mass with homogeneous high signal and fluid characteristics is noted, consistent with a periarticular cyst.',
        'A well-circumscribed smooth-walled fluid-filled mass demonstrating homogeneously high signal is identified.',
        'A fluid-filled mass with smooth walls and homogeneously high signal is seen, representing a periarticular cyst.',
        'A well-defined periarticular mass with smooth walls and homogeneous high fluid signal is observed, consistent with a cyst.',
    ],
    'cartilage - partial thickness': [
        'A bright gap with high-signal fluid filling the cartilage defect is noted, consistent with a partial thickness cartilage lesion.',
        'High-signal fluid is seen filling a partial thickness cartilage defect, creating a bright gap at the articular surface.',
        'A bright gap is identified within the articular cartilage with high-signal fluid filling the defect, representing partial thickness cartilage loss.',
        'Partial thickness cartilage loss with a bright gap of high-signal fluid filling the defect is observed.',
        'A high-signal fluid-filled bright gap is noted within the cartilage, consistent with a partial thickness defect.',
        'Bright signal is seen filling a partial cartilage defect, consistent with high-signal fluid in a partial thickness lesion.',
        'A focal bright gap with high-signal fluid is identified within the cartilage, representing a partial thickness defect.',
        'High-signal fluid filling a bright gap in the cartilage is observed, consistent with a partial thickness cartilage defect.',
    ],
    'cartilage - full thickness': [
        'A bright gap with high-signal fluid filling a full thickness cartilage defect is noted, with complete loss of the cartilage layer.',
        'High-signal fluid fills a full thickness cartilage defect, creating a bright gap extending to the subchondral bone.',
        'A bright gap is identified with high-signal fluid filling a full thickness defect of the articular cartilage.',
        'Full thickness cartilage loss with a bright gap of high-signal fluid filling the defect is observed.',
        'A high-signal fluid-filled bright gap extending through the full cartilage thickness is noted, consistent with a full thickness defect.',
        'Bright signal is seen filling a full thickness cartilage defect with exposure of subchondral bone.',
        'A focal bright gap with high-signal fluid extends through the entire cartilage thickness, representing a full thickness defect.',
        'Complete cartilage loss with high-signal fluid filling the defect as a bright gap is observed.',
    ],
    'bone marrow edema': [
        'Abnormal fluid signals in the bone marrow are noted, consistent with bone marrow edema.',
        'Fluid signal abnormality within the bone marrow is identified, suggestive of bone marrow edema.',
        'Abnormal fluid signals are seen within the osseous marrow, consistent with bone marrow edema.',
        'The bone marrow demonstrates abnormal fluid signals, indicating marrow edema.',
        'Abnormal fluid signal intensity is observed within the bone marrow, consistent with edema.',
        'Fluid signals within the bone marrow are noted with abnormal characteristics, suggestive of marrow edema.',
        'Abnormal marrow fluid signals are identified, consistent with bone marrow edema.',
        'The osseous structures demonstrate abnormal fluid signals in the marrow, representing bone marrow edema.',
    ],
    'subchondral edema': [
        'Abnormal fluid signals in the subchondral bone marrow are noted, consistent with subchondral edema.',
        'Subchondral fluid signal abnormality is identified beneath the articular surface, suggestive of subchondral edema.',
        'Abnormal fluid signals are seen in the subchondral marrow, consistent with subchondral edema.',
        'The subchondral bone demonstrates abnormal fluid signals, indicating subchondral edema.',
        'Abnormal fluid signal intensity is observed in the subchondral marrow, consistent with edema.',
        'Subchondral marrow fluid signals are noted with abnormal characteristics, suggestive of subchondral edema.',
        'Abnormal fluid signals in the marrow beneath the articular surface are identified, consistent with subchondral edema.',
        'The subchondral region demonstrates abnormal fluid signals, representing subchondral edema.',
    ],
    # --- Diagnoses without user-provided descriptions (kept as-is) ---
    'displaced meniscal tissue': [
        'Displaced meniscal fragment is identified, indicating meniscal extrusion or bucket-handle tear.',
        'Abnormal meniscal tissue is seen in a displaced position, consistent with a displaced meniscal fragment.',
        'A fragment of meniscal tissue is noted outside its normal anatomic position.',
        'Meniscal tissue displacement is observed, suggesting a bucket-handle or flap tear.',
        'A meniscal fragment is identified in an abnormal location, consistent with displacement from a complex tear.',
        'Displaced meniscal material is seen within the joint, indicating a significant meniscal injury.',
        'A portion of the meniscus is noted in a displaced position, suggestive of a bucket-handle tear.',
        'Meniscal tissue is identified outside the expected anatomic boundaries, consistent with meniscal displacement.',
    ],
    'lcl': [
        'Abnormal signal in the lateral collateral ligament complex is noted, consistent with an LCL injury.',
        'Signal abnormality is identified along the lateral collateral ligament, suggestive of a sprain.',
        'The lateral collateral ligament complex demonstrates abnormal signal intensity, indicating injury.',
        'Periligamentous changes along the LCL are noted, consistent with a sprain.',
        'Thickening and increased signal of the lateral collateral ligament complex is observed.',
        'The LCL complex shows abnormal signal with surrounding edema, indicating injury.',
        'Signal changes within the lateral collateral ligament are identified, consistent with a sprain.',
        'Abnormal signal and morphology of the LCL complex is noted, suggestive of a ligamentous injury.',
    ],
    'fracture': [
        'A linear signal abnormality through the bone is noted, consistent with a fracture.',
        'Fracture line with surrounding marrow edema is identified.',
        'Bone discontinuity with adjacent edema is seen, consistent with a fracture or contusion.',
        'Signal abnormality through the osseous structure is noted, suggestive of a fracture.',
        'A low-signal line traversing the bone marrow is identified, consistent with a fracture.',
        'Cortical disruption with surrounding marrow signal change is observed, indicating a fracture.',
        'A linear area of signal abnormality is seen within the bone, consistent with an occult fracture.',
        'Osseous signal abnormality with a linear pattern is noted, suggestive of a fracture or contusion.',
    ],
    'bone lesion': [
        'A focal bone lesion with abnormal signal characteristics is identified.',
        'An osseous lesion is noted with signal abnormality, requiring further characterization.',
        'A focal area of abnormal bone signal is seen, consistent with a bone lesion.',
        'An osseous signal abnormality is identified, representing a bone lesion.',
        'A well-defined or ill-defined lesion within the bone is observed with abnormal signal.',
        'A focal osseous abnormality is noted, consistent with a bone lesion.',
        'An area of altered bone marrow signal is identified, representing a focal bone lesion.',
        'A bone lesion with abnormal signal intensity is seen, requiring clinical correlation.',
    ],
    'soft tissue lesion': [
        'An area of altered signal intensity is noted in the soft tissue, suggestive of a soft tissue lesion.',
        'Abnormal soft tissue signal is identified, consistent with a soft tissue lesion.',
        'A focal soft tissue abnormality with abnormal signal characteristics is seen.',
        'Signal abnormality within the soft tissues is noted, representing a soft tissue lesion.',
        'A mass-like signal abnormality is observed in the periarticular soft tissues.',
        'Focal soft tissue signal change is identified, consistent with a soft tissue lesion.',
        'An area of abnormal signal is noted within the soft tissues adjacent to the knee.',
        'A soft tissue abnormality with heterogeneous signal characteristics is seen.',
    ],
    'muscle strain': [
        'Abnormal signal within the muscle fibers is noted, consistent with a muscle strain.',
        'Muscle edema with feathery signal pattern is identified, suggestive of a strain.',
        'Increased signal within the musculature is seen, consistent with a muscle strain or contusion.',
        'Signal abnormality within the muscle is noted, indicating a strain injury.',
        'Intramuscular signal change with a feathery pattern is observed, consistent with a muscle strain.',
        'Focal muscle edema is identified, suggestive of an acute strain injury.',
        'Abnormal signal along the muscle fibers is seen, consistent with a partial muscle tear or strain.',
        'The muscle demonstrates focal signal abnormality, indicating a strain or contusion.',
    ],
    'patellar retinaculum': [
        'Signal abnormality along the patellar retinaculum is noted, consistent with a retinacular injury.',
        'Abnormal signal is identified in the patellar retinaculum, suggestive of a sprain.',
        'Periretinacular signal changes are seen, indicating injury to the patellar retinaculum.',
        'The patellar retinaculum demonstrates abnormal signal, consistent with a high-grade sprain.',
        'Thickening and signal abnormality of the patellar retinaculum is observed, indicating injury.',
        'Disruption of the patellar retinaculum with surrounding edema is noted.',
        'Abnormal signal intensity within the retinaculum is identified, consistent with a retinacular sprain.',
        'The patellar retinaculum shows signal change and irregularity, suggestive of injury.',
    ],
    'joint bodies': [
        'A small intra-articular body is identified within the joint space.',
        'Loose body is noted within the joint, consistent with an intra-articular joint body.',
        'A focal signal abnormality within the joint is seen, suggestive of a loose body.',
        'An intra-articular body is identified, possibly from a cartilage or osteochondral fragment.',
        'A small rounded structure is noted within the joint space, consistent with a loose body.',
        'An osteochondral fragment is identified as a loose body within the joint.',
        'A joint body is observed within the articular space, likely from prior cartilage or bone injury.',
        'A small intra-articular loose body is seen, consistent with an osteochondral fragment.',
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
        ('a meniscus tear.', 'meniscus tears.'),
        ('a meniscal tear.', 'meniscal tears.'),
        ('a tear.', 'tears.'),
        ('a joint effusion.', 'joint effusion.'),
        ('a periarticular cyst.', 'periarticular cysts.'),
        ('a sprain.', 'sprains.'),
        ('a partial thickness defect.', 'partial thickness defects.'),
        ('a full thickness defect.', 'full thickness defects.'),
        ('a partial thickness cartilage lesion.', 'partial thickness cartilage lesions.'),
        ('a nonspecific white matter lesion.', 'nonspecific white matter lesions.'),
        ('a nonspecific lesion.', 'nonspecific lesions.'),
        ('a white matter lesion.', 'white matter lesions.'),
        ('a lacunar infarct.', 'lacunar infarcts.'),
        ('a prior craniotomy.', 'prior craniotomy changes.'),
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
            # Check if key words from the label appear in reasoning
            key_words = [w for w in label_lower.split() if len(w) > 3]
            found = label_lower in reasoning_lower
            if not found:
                found = any(w in reasoning_lower for w in key_words)
            if not found:
                # Check abbreviations
                for abbr in ['acl', 'mcl', 'pcl', 'lcl']:
                    if abbr in label_lower and abbr in reasoning_lower:
                        found = True
                        break
            if not found:
                missing_labels.append(label)

        if not missing_labels:
            continue

        # Generate descriptions for missing labels
        additions = []
        for label in missing_labels:
            label_lower = label.lower()
            desc = None
            for key, descs in KNEE_IMAGING_CHARACTERISTICS.items():
                if key in label_lower or label_lower in key:
                    desc = random.choice(descs)
                    break
                # Partial matches for compound labels
                if 'acl' in key and 'acl' in label_lower:
                    desc = random.choice(descs)
                    break
                if 'mcl' in key and 'mcl' in label_lower:
                    desc = random.choice(descs)
                    break
                if 'pcl' in key and 'pcl' in label_lower:
                    desc = random.choice(descs)
                    break
                if 'lcl' in key and 'lcl' in label_lower:
                    desc = random.choice(descs)
                    break
                if 'cartilage' in key and 'cartilage' in label_lower:
                    if 'partial' in key and 'partial' in label_lower:
                        desc = random.choice(descs)
                        break
                    if 'full' in key and 'full' in label_lower:
                        desc = random.choice(descs)
                        break
                    if 'partial' not in key and 'full' not in key:
                        continue
                if 'fracture' in key and ('fracture' in label_lower or 'contusion' in label_lower):
                    desc = random.choice(descs)
                    break
                if 'subchondral' in key and 'subchondral' in label_lower:
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




KNEE_LABEL_TO_CHAR_KEY = {
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

        missing = []
        for diag in diagnoses:
            diag_lower = diag.lower()
            # Check direct match
            if diag_lower in reasoning_lower:
                continue
            # Key word match
            key_words = [w for w in diag_lower.split() if len(w) > 3]
            if key_words and all(w in reasoning_lower for w in key_words):
                continue
            # Abbreviation match
            for abbr in ['acl', 'mcl', 'pcl', 'lcl']:
                if abbr in diag_lower and abbr in reasoning_lower:
                    break
            else:
                missing.append(diag)
                continue
            # abbr was found, skip

        if not missing:
            continue

        additions = []
        for diag in missing:
            char_key = KNEE_LABEL_TO_CHAR_KEY.get(diag.lower())
            if char_key is None:
                continue
            descs = KNEE_IMAGING_CHARACTERISTICS.get(char_key)
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
        if isinstance(gt, dict):
            fd = gt.get('final_diagnosis', gt.get('labels', []))
        else:
            fd = []

        # Find matching choice letters
        matched_letters = []
        for letter, choice_text in sorted(choices.items()):
            choice_lower = choice_text.lower()
            for diag in fd:
                diag_lower = diag.lower()
                if diag_lower in choice_lower or choice_lower in diag_lower:
                    matched_letters.append(f'({letter})')
                    break
                diag_words = diag_lower.split()
                if len(diag_words) >= 2 and all(w in choice_lower for w in diag_words[:2]):
                    matched_letters.append(f'({letter})')
                    break

        if matched_letters:
            letter_str = '+'.join(matched_letters)
            if not answer.startswith('('):
                e['answer'] = f'{letter_str} {answer}'
                count += 1
    return count




def restructure_counting(qa, vol_meta):
    """Remove bboxes from counting reasoning and rebuild with clean summary format."""
    if not vol_meta:
        return 0
    count = 0
    for e in qa:
        if e.get('task') != 'counting':
            continue
        if e.get('level') == 'slice':
            continue

        vol_id = e.get('volume_id', '')
        meta = vol_meta.get(vol_id)
        if not meta:
            continue

        # Count findings from metadata
        bbox_frames = get_volume_bbox_frames(meta)
        if not bbox_frames:
            continue

        label_counts = Counter()
        for bbs in bbox_frames.values():
            for bb in bbs:
                label_counts[bb.get('label', 'finding')] += 1

        total = sum(label_counts.values())
        n_categories = len(label_counts)

        # Build clean summary
        parts = [f"{label} ({cnt} instance{'s' if cnt > 1 else ''})"
                 for label, cnt in label_counts.most_common()]
        listing = ', '.join(parts)

        new_reasoning = (
            f"The volume contains {n_categories} distinct "
            f"{'categories' if n_categories > 1 else 'category'} of findings: "
            f"{listing}. Total = {total} abnormal finding{'s' if total > 1 else ''}."
        )

        old_reasoning = e.get('reasoning', '')
        # Only update if it changes the content (preserve if already clean)
        if re.search(r'\[\d+,\s*\d+', old_reasoning):
            # Has bboxes - needs cleanup
            e['reasoning'] = new_reasoning
            count += 1
        elif 'distinct categories' not in old_reasoning.lower() and 'distinct category' not in old_reasoning.lower():
            e['reasoning'] = new_reasoning
            count += 1
    return count




def add_localization_bboxes(qa, vol_meta):
    """Add bbox listings from volume metadata to localization answers."""
    if not vol_meta:
        return 0
    count = 0
    for e in qa:
        if e.get('task') != 'localization':
            continue
        if e.get('level') == 'slice':
            continue

        vol_id = e.get('volume_id', '')
        meta = vol_meta.get(vol_id)
        if not meta:
            continue

        answer = e.get('answer', '')
        # Skip if already has bboxes
        if re.search(r'\[\d+,\s*\d+,\s*\d+,\s*\d+\]', answer):
            continue

        bbox_frames = get_volume_bbox_frames(meta)
        if not bbox_frames:
            continue

        # Group by label
        label_bboxes = defaultdict(list)
        for frame_num in sorted(bbox_frames.keys()):
            for bb in bbox_frames[frame_num]:
                label = bb.get('label', 'finding')
                bbox_str = f"<bbx>[{bb['x']},{bb['y']},{bb['width']},{bb['height']}]</bbx>"
                label_bboxes[label].append(f"Frame {frame_num}: {bbox_str}")

        # Build localization answer
        parts = []
        for label, locations in label_bboxes.items():
            loc_str = '; '.join(locations)
            parts.append(f"{label}: {loc_str}")

        bbox_listing = '. '.join(parts)
        e['answer'] = answer.rstrip('. ') + '. ' + bbox_listing + '.'
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
            sample_frames = [frame_nums[0], frame_nums[len(frame_nums)//2], frame_nums[-1]]
        else:
            sample_frames = frame_nums

        parts = []
        for fn in sample_frames:
            bbs = bbox_frames[fn]
            bb = bbs[0]
            label = bb.get('label', 'finding')
            bbox_str = f"[{bb['x']},{bb['y']},{bb['width']},{bb['height']}]"
            parts.append(f"In frame {fn}, a {label.lower()} is at {bbox_str}")

        total_bboxes = sum(len(bbs) for bbs in bbox_frames.values())
        narrative = '. '.join(parts)
        bbox_info = f" The volume shows findings across {len(bbox_frames)} frames with {total_bboxes} total annotations. {narrative}."

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


def remove_duplicate_captioning_bboxes(qa):
    """Remove duplicated bbox listings in captioning reasoning."""
    count = 0
    for e in qa:
        if e.get('task') != 'captioning':
            continue
        reasoning = e.get('reasoning', '')
        # Find repeated "In frame N, ..." or "Frame N:" blocks
        # Check for duplicated "The volume shows findings across..." blocks
        pattern = re.compile(
            r'(The volume shows findings across \d+ frames with \d+ total annotations\.[^.]+\.)'
        )
        matches = list(pattern.finditer(reasoning))
        if len(matches) > 1:
            # Keep only the first occurrence
            first_end = matches[0].end()
            for m in matches[1:]:
                reasoning = reasoning[:m.start()] + reasoning[m.end():]
            if reasoning != e.get('reasoning', ''):
                e['reasoning'] = reasoning.strip()
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




def add_clinical_descriptions_to_captioning(qa):
    """Add clinical diagnosis descriptions to captioning reasoning."""
    descriptions = {
        'meniscus tear': [
            'A meniscus tear is a knee injury affecting the cartilage that cushions the joint.',
            'This finding represents a meniscus tear, a common knee cartilage injury.',
            'The appearance is consistent with a meniscus tear affecting the fibrocartilage.',
            'A meniscus tear is identified, indicating damage to the knee cartilage.',
            'The finding suggests a meniscal injury with disruption of the cartilage.',
            'A meniscus tear is present, a frequent cause of knee pain and mechanical symptoms.',
            'The imaging demonstrates a meniscus tear with abnormal meniscal morphology.',
            'A meniscal tear is identified, representing cartilage damage in the knee joint.',
        ],
        'acl': [
            'An ACL injury is identified, affecting the ligament connecting the thighbone to the shinbone.',
            'The finding is consistent with an ACL tear or sprain.',
            'An ACL injury is present, involving the anterior cruciate ligament of the knee.',
            'The appearance suggests ACL damage with abnormal ligament signal.',
            'An ACL sprain is identified, a common sports-related knee injury.',
            'The finding represents an ACL injury with disruption of ligament fibers.',
            'An ACL tear is seen, indicating damage to this key stabilizing ligament.',
            'The imaging demonstrates ACL pathology with abnormal ligament morphology.',
        ],
        'mcl': [
            'An MCL sprain is identified, caused by lateral impact or twisting of the knee.',
            'The finding is consistent with MCL injury affecting the medial stabilizer.',
            'An MCL sprain is present, involving the medial collateral ligament.',
            'The appearance suggests MCL damage with periligamentous signal changes.',
            'An MCL injury is identified, a common knee ligament sprain.',
            'The finding represents MCL pathology with abnormal ligament signal.',
            'An MCL sprain is seen, indicating damage to the medial stabilizing ligament.',
            'The imaging demonstrates MCL injury with characteristic signal abnormality.',
        ],
        'pcl': [
            'A PCL injury is identified, affecting the ligament at the back of the knee.',
            'The finding is consistent with PCL sprain or tear.',
            'A PCL injury is present, involving the posterior cruciate ligament.',
            'The appearance suggests PCL damage with abnormal signal intensity.',
            'A PCL sprain is identified, affecting the posterior knee stabilizer.',
            'The finding represents PCL pathology with disrupted ligament fibers.',
            'A PCL injury is seen, indicating damage to this posterior knee ligament.',
            'The imaging demonstrates PCL injury with characteristic morphologic changes.',
        ],
        'joint effusion': [
            'Joint effusion is identified, representing abnormal fluid accumulation in the knee joint.',
            'The finding is consistent with joint effusion, indicating excess synovial fluid.',
            'Joint effusion is present, with increased fluid in the joint space.',
            'The appearance suggests joint effusion with distension of the joint capsule.',
            'Effusion is identified in the knee joint, a sign of underlying pathology.',
            'The finding represents joint effusion, excess fluid within the articular space.',
            'Joint effusion is seen, indicating inflammation or injury to the knee.',
            'The imaging demonstrates joint effusion with fluid distending the suprapatellar bursa.',
        ],
        'cartilage': [
            'Cartilage damage is identified, representing loss of the smooth articular surface.',
            'The finding is consistent with articular cartilage defect in the knee.',
            'Cartilage loss is present, indicating damage to the joint surface.',
            'The appearance suggests cartilage injury with focal surface irregularity.',
            'A cartilage defect is identified, affecting the articular surface.',
            'The finding represents cartilage pathology with loss of normal thickness.',
            'Cartilage damage is seen, consistent with degenerative or traumatic change.',
            'The imaging demonstrates cartilage loss with abnormal signal at the joint surface.',
        ],
        'periarticular cyst': [
            'A periarticular cyst is identified, arising from degenerative joint disease or ligament damage.',
            'The finding is consistent with a periarticular cyst adjacent to the joint.',
            'A periarticular cyst is present, a fluid-filled structure near the joint.',
            'The appearance suggests a periarticular cyst related to joint pathology.',
            'A cystic lesion is identified in the periarticular region.',
            'The finding represents a periarticular cyst, commonly associated with meniscal tears.',
            'A periarticular cyst is seen, indicating underlying joint derangement.',
            'The imaging demonstrates a periarticular cyst adjacent to the knee joint.',
        ],
        'bone marrow edema': [
            'Bone marrow edema is identified, representing fluid buildup within the bone marrow.',
            'The finding is consistent with bone marrow edema from acute injury or stress.',
            'Bone marrow edema is present, indicating increased fluid in the osseous structures.',
            'The appearance suggests bone marrow edema with abnormal marrow signal.',
            'Bone marrow edema is identified, a common finding after trauma or contusion.',
            'The finding represents bone marrow edema, reflecting osseous injury.',
            'Bone marrow edema is seen, consistent with trabecular microfracture or contusion.',
            'The imaging demonstrates bone marrow edema with characteristic signal changes.',
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




def wrap_bare_bboxes(qa):
    """Wrap bare [x,y,w,h] coordinates with <bbx> tags in localization answers."""
    count = 0
    pattern = re.compile(r'(?<!<bbx>)\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\](?!</bbx>)')
    for e in qa:
        if e.get('task') != 'localization':
            continue
        answer = e.get('answer', '')
        new_answer = pattern.sub(r'<bbx>[\1,\2,\3,\4]</bbx>', answer)
        if new_answer != answer:
            e['answer'] = new_answer
            count += 1
    return count


# Main

def process_file(path, split, file_type, fibula_data, vol_meta=None, dry_run=False):
    """Process a single knee QA file."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(path)}")
    print(f"{'='*60}")

    qa = load_json(path)
    if not dry_run:
        backup(path)

    fibula = fibula_data.get(split, {})
    verified_vols = get_verified_volumes(qa, split)
    print(f"  Verified volumes (skipped): {len(verified_vols)}")
    print(f"  Fibula annotations available: {len(fibula)}")

    results = {}

    # ── Section 1: Medial/Lateral Fixes ──
    if fibula:
        med_lat_counts = fix_medial_lateral(qa, fibula, verified_vols)
        for (task, field), count in med_lat_counts.items():
            results[f'med_lat_{task}_{field}'] = count

    # ── Section 2: Grammar & Text Fixes ──
    results['grammar'] = fix_grammar(qa)
    results['frame_refs'] = fix_frame_references(qa)
    results['trailing'] = fix_trailing_sentences(qa)

    # ── Section 3: Classification Fixes ──
    results['classification_letters'] = add_classification_answer_letters(qa)

    # ── Section 4: Counting Restructure (volume QA only) ──
    if file_type == 'volume' and vol_meta:
        results['counting_restructure'] = restructure_counting(qa, vol_meta)

    # ── Section 5: Localization Enrichment (volume QA only) ──
    if file_type == 'volume' and vol_meta:
        results['localization_bboxes'] = add_localization_bboxes(qa, vol_meta)

    # ── Section 6: BBox Info Enrichment ──
    if file_type == 'volume' and vol_meta:
        results['vol_captioning_bbox'] = add_volume_captioning_bbox_info(qa, vol_meta)

    if file_type == 'image':
        results['detection_enrich'] = enrich_detection_reasoning(qa)
        results['img_captioning_bbox'] = add_bbox_info_to_image_captioning(qa)
        results['remove_class_bbox'] = remove_bbox_listing_from_classification(qa)

    # ── Section 7: Reasoning Diversity (volume QA only) ──
    if file_type == 'volume':
        results['count_diversity'] = diversify_counting_reasoning(qa)

    # ── Section 8: Duplicate Removal ──
    if file_type == 'volume':
        results['duplicate_captioning'] = remove_duplicate_captioning_bboxes(qa)

    # ── Section 9: Clinical Descriptions ──
    results['clinical_descriptions'] = add_clinical_descriptions_to_captioning(qa)

    # ── Section 10: BBox Tag Wrapping (volume QA only) ──
    if file_type == 'volume':
        results['bbx_tags'] = wrap_bare_bboxes(qa)

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
    parser = argparse.ArgumentParser(description='Clean knee QA data')
    parser.add_argument('--split', choices=['train', 'val'], help='Process only this split')
    parser.add_argument('--type', choices=['volume', 'image'], help='Process only this type')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without writing')
    parser.add_argument('--metadata-dir', default=None,
                        help='Path to knee volume metadata directory')
    args = parser.parse_args()

    # Load fibula annotations
    fibula_data = load_fibula_annotations()
    if not fibula_data:
        print("WARNING: No fibula annotations. Medial/lateral fixes will be skipped.")

    metadata_dir = args.metadata_dir or DEFAULT_METADATA_DIR

    splits = [args.split] if args.split else ['val', 'train']
    types = [args.type] if args.type else ['volume', 'image']

    for split in splits:
        # Load volume metadata (for enrichment steps)
        vol_meta = load_knee_volume_metadata(split, metadata_dir)
        if vol_meta:
            print(f"Loaded {len(vol_meta)} volume metadata entries for {split}")

        for file_type in types:
            filename = f"gpt4o_knee_{split}_{file_type}_qa_{file_type}_qa_pairs.json"
            path = os.path.join(KNEE_DIR, filename)
            if os.path.exists(path):
                process_file(path, split, file_type, fibula_data or {},
                             vol_meta=vol_meta, dry_run=args.dry_run)
            else:
                print(f"Skipping {filename} (not found)")


if __name__ == '__main__':
    main()
