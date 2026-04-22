"""
Fix singular descriptions in detection reasoning for labels with multiple bboxes.

Scans all 4 image QA files. For each detection entry where a label has >1 bbox,
finds sentences describing that label and converts from singular to plural.

Key rule: Only apply is→are when the sentence subject is already pluralized
(i.e., starts with "Multiple"). For sentences with mass noun subjects like
"signal" or "loss", leave the verb singular to avoid ungrammatical output.
"""

import json
import re
import sys


def _label_keywords(label):
    """Get matching keywords for a label (lowercased)."""
    label_lower = label.lower()
    keywords = set()
    keywords.add(label_lower)
    for w in label_lower.split():
        if len(w) > 3:
            keywords.add(w)
    abbrev_map = {
        'acl': {'acl', 'anterior cruciate'},
        'mcl': {'mcl', 'medial collateral'},
        'pcl': {'pcl', 'posterior cruciate'},
        'lcl': {'lcl', 'lateral collateral'},
    }
    for abbr, expansions in abbrev_map.items():
        if abbr in label_lower:
            keywords.update(expansions)
    return keywords


def _sentence_matches_label(sentence, keywords):
    """Check if a sentence is about a label based on keywords."""
    s_lower = sentence.lower()
    return any(kw in s_lower for kw in keywords)


def _pluralize_sentence(sentence):
    """
    Convert a singular sentence to plural for multi-bbox findings.

    Strategy:
    - Sentences starting with "A/An": Convert to "Multiple ...", pluralize nouns, change is→are
    - Other sentences: Only fix specific noun endings, NOT the verb (mass nouns stay singular)
    """
    s = sentence
    started_with_article = False

    # --- Step 1: Replace leading articles ---
    if s.startswith('A '):
        s = 'Multiple ' + s[2].lower() + s[3:]
        started_with_article = True
    elif s.startswith('An '):
        s = 'Multiple ' + s[3].lower() + s[4:]
        started_with_article = True

    # --- Step 2: Replace specific "a/an <noun>" phrases (removes article + pluralizes) ---
    # Run BEFORE general noun regex so "a nonspecific white matter lesion" is fully handled
    for sing, plur in [
        ('a nonspecific white matter lesion.', 'nonspecific white matter lesions.'),
        ('a nonspecific lesion.', 'nonspecific lesions.'),
        ('a white matter lesion.', 'white matter lesions.'),
        ('a meniscus tear.', 'meniscus tears.'),
        ('a meniscal tear.', 'meniscal tears.'),
        ('a lacunar infarct.', 'lacunar infarcts.'),
        ('a chronic lacunar infarct.', 'chronic lacunar infarcts.'),
        ('a prior craniotomy.', 'prior craniotomy changes.'),
        ('a craniotomy.', 'craniotomy changes.'),
        ('a surgical craniotomy.', 'surgical craniotomy changes.'),
        ('a periarticular cyst.', 'periarticular cysts.'),
        ('a partial thickness cartilage defect.', 'partial thickness cartilage defects.'),
        ('a partial thickness chondral defect.', 'partial thickness chondral defects.'),
        ('a partial thickness defect.', 'partial thickness defects.'),
        ('a partial thickness cartilage lesion.', 'partial thickness cartilage lesions.'),
        ('a full thickness cartilage defect.', 'full thickness cartilage defects.'),
        ('a full thickness chondral defect.', 'full thickness chondral defects.'),
        ('a full thickness defect.', 'full thickness defects.'),
        ('a mass.', 'masses.'),
        ('an intracranial mass.', 'intracranial masses.'),
        ('an intracranial neoplasm.', 'intracranial neoplasms.'),
        ('a space-occupying lesion.', 'space-occupying lesions.'),
        ('a possible artifact.', 'possible artifacts.'),
        ('a resection cavity.', 'resection cavities.'),
        ('a post-surgical cavity.', 'post-surgical cavities.'),
        ('a post-surgical resection cavity.', 'post-surgical resection cavities.'),
        ('a bone lesion.', 'bone lesions.'),
        ('an extra-axial mass.', 'extra-axial masses.'),
        ('an extra-axial collection.', 'extra-axial collections.'),
        ('a pineal cyst.', 'pineal cysts.'),
        # Mid-sentence (no trailing period)
        ('a nonspecific white matter lesion,', 'nonspecific white matter lesions,'),
        ('a nonspecific lesion,', 'nonspecific lesions,'),
        ('a meniscus tear,', 'meniscus tears,'),
        ('a meniscal tear,', 'meniscal tears,'),
        # "consistent/suggestive with a <noun>" patterns
        ('consistent with a tear.', 'consistent with tears.'),
        ('suggestive of a tear.', 'suggestive of tears.'),
        ('indicating a tear.', 'indicating tears.'),
        ('representing a tear.', 'representing tears.'),
        ('consistent with a sprain.', 'consistent with sprains.'),
        ('suggestive of a sprain.', 'suggestive of sprains.'),
        ('indicating a sprain.', 'indicating sprains.'),
        ('consistent with a ligament sprain.', 'consistent with ligament sprains.'),
        ('consistent with an MCL sprain.', 'consistent with MCL sprains.'),
        ('consistent with an MCL injury.', 'consistent with MCL injuries.'),
        ('consistent with an ACL injury.', 'consistent with ACL injuries.'),
        ('consistent with a PCL injury.', 'consistent with PCL injuries.'),
        ('consistent with ligament injury.', 'consistent with ligament injuries.'),
        ('consistent with a medial collateral ligament sprain.', 'consistent with medial collateral ligament sprains.'),
    ]:
        s = s.replace(sing, plur)
        # Also try with capital A
        s = s.replace(sing[0].upper() + sing[1:], plur[0].upper() + plur[1:] if plur[0].islower() else plur)

    # --- Step 3: Singular is→are ONLY if we changed to "Multiple" ---
    if started_with_article:
        for sing, plur in [
            (' is noted', ' are noted'),
            (' is seen', ' are seen'),
            (' is identified', ' are identified'),
            (' is observed', ' are observed'),
            (' is present', ' are present'),
        ]:
            s = s.replace(sing, plur)

        # Pluralize remaining countable nouns after "Multiple" (catches anything step 2 missed)
        noun_plurals = [
            (r'(Multiple\b.+?) \bfocus\b', r'\1 foci'),
            (r'(Multiple\b.+?) \bareas?\b', r'\1 areas'),
            (r'(Multiple\b.+?) \babnormality\b', r'\1 abnormalities'),
            (r'(Multiple\b.+?) \blesions?\b', r'\1 lesions'),
            (r'(Multiple\b.+?) \bdefects?\b', r'\1 defects'),
            (r'(Multiple\b.+?) \bbands?\b', r'\1 bands'),
            (r'(Multiple\b.+?) \bregion\b(?! of)', r'\1 regions'),
        ]
        for pat, repl in noun_plurals:
            s = re.sub(pat, repl, s, flags=re.IGNORECASE)

    # "focus" → "foci" standalone (for "Multiple" sentences)
    if started_with_article and 'focus' in s.lower() and 'foci' not in s.lower():
        s = re.sub(r'\bfocus\b', 'foci', s, flags=re.IGNORECASE)

    return s


def fix_detection_plurals(qa, vol_bbox_counts=None, dry_run=False):
    """
    Fix singular descriptions in detection reasoning for multi-bbox labels.
    For image QA: uses inline ground_truth.bboxes.
    For volume QA: uses vol_bbox_counts dict from metadata.
    Returns (count_fixed, details_list).
    """
    count = 0
    details = []

    for e in qa:
        if e.get('task') != 'detection':
            continue

        level = e.get('level', '')

        if level == 'slice':
            # Image-level: use inline bboxes
            gt = e.get('ground_truth', {})
            if not isinstance(gt, dict):
                continue
            bboxes = gt.get('bboxes', [])
            bbox_counts = {}
            for bb in bboxes:
                lbl = bb.get('label', '')
                bbox_counts[lbl] = bbox_counts.get(lbl, 0) + 1
        elif vol_bbox_counts is not None:
            # Volume-level: use metadata bbox counts
            vol_id = e.get('volume_id', '')
            bbox_counts = vol_bbox_counts.get(vol_id, {})
        else:
            continue

        multi_labels = {l: c for l, c in bbox_counts.items() if c > 1}
        if not multi_labels:
            continue

        reasoning = e.get('reasoning', '')

        # Split into sentences
        sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', reasoning)

        modified = False
        new_sentences = []
        for sentence in sentences:
            new_sentence = sentence
            for label, cnt in multi_labels.items():
                keywords = _label_keywords(label)
                if _sentence_matches_label(sentence, keywords):
                    # Check if sentence has singular patterns worth fixing
                    has_singular_article = bool(re.match(r'^A[n]?\s+[a-z]', sentence))
                    has_singular_noun = any(
                        p in sentence.lower()
                        for p in [
                            'a nonspecific white matter lesion',
                            'a nonspecific lesion',
                            'a white matter lesion',
                            'a meniscus tear',
                            'a meniscal tear',
                            'a lacunar infarct',
                            'a prior craniotomy',
                            'a craniotomy',
                            'a periarticular cyst',
                            'a mass',
                            'a space-occupying lesion',
                            'a possible artifact',
                            'a resection cavity',
                            'a sprain',
                            'a tear',
                            'a partial thickness',
                            'a full thickness',
                            'a bone lesion',
                            'an extra-axial',
                        ]
                    )
                    if has_singular_article or has_singular_noun:
                        new_sentence = _pluralize_sentence(new_sentence)
                        if new_sentence != sentence:
                            modified = True
            new_sentences.append(new_sentence)

        if modified:
            new_reasoning = ' '.join(new_sentences)
            if new_reasoning != reasoning:
                details.append({
                    'volume_id': e.get('volume_id', ''),
                    'slice_num': e.get('slice_num', ''),
                    'multi_labels': multi_labels,
                    'old': reasoning[:250],
                    'new': new_reasoning[:250],
                })
                if not dry_run:
                    e['reasoning'] = new_reasoning
                count += 1

    return count, details


def _load_volume_bbox_counts(meta_path):
    """Load volume metadata and return {vol_id: {label: count}} dict."""
    with open(meta_path) as f:
        meta = json.load(f)

    vol_bbox_counts = {}
    first_val = next(iter(meta.values()))

    if isinstance(first_val, dict) and 'volume_id' not in first_val and 'slices' not in first_val:
        # Brain format: {modality: {vol_id: {...}}}
        for mod_data in meta.values():
            if not isinstance(mod_data, dict):
                continue
            for vol_id, vol_data in mod_data.items():
                counts = {}
                for sl in vol_data.get('slices', []):
                    for bb in sl.get('bounding_boxes', []):
                        lbl = bb.get('label', '')
                        counts[lbl] = counts.get(lbl, 0) + 1
                vol_bbox_counts[vol_id] = counts
    else:
        # Knee format: {vol_id: {...}}
        for vol_id, vol_data in meta.items():
            if not isinstance(vol_data, dict):
                continue
            counts = {}
            for sl in vol_data.get('slices', []):
                for bb in sl.get('bounding_boxes', []):
                    lbl = bb.get('label', '')
                    counts[lbl] = counts.get(lbl, 0) + 1
            vol_bbox_counts[vol_id] = counts

    return vol_bbox_counts


DATA_PROC = os.environ.get('SGMRIQA_DATA_PROCESSING', 'data_processing')


def main():
    dry_run = '--dry-run' in sys.argv

    # Image-level files (bbox counts from inline ground_truth)
    image_files = [
        ('brain_val_img', 'brain/gpt4o_brain_val_image_qa_image_qa_pairs.json', None),
        ('brain_train_img', 'brain/gpt4o_brain_train_image_qa_image_qa_pairs.json', None),
        ('knee_val_img', 'knee/gpt4o_knee_val_image_qa_image_qa_pairs.json', None),
        ('knee_train_img', 'knee/gpt4o_knee_train_image_qa_image_qa_pairs.json', None),
    ]

    # Volume-level files (bbox counts from metadata)
    volume_files = [
        ('brain_val_vol', 'brain/gpt4o_brain_val_volume_qa_volume_qa_pairs.json',
         f'{DATA_PROC}/brain/brain_val_volumes.json'),
        ('brain_train_vol', 'brain/gpt4o_brain_train_volume_qa_volume_qa_pairs.json',
         f'{DATA_PROC}/brain/brain_train_volumes.json'),
        ('knee_val_vol', 'knee/gpt4o_knee_val_volume_qa_volume_qa_pairs.json',
         f'{DATA_PROC}/knee/knee_val_volumes.json'),
        ('knee_train_vol', 'knee/gpt4o_knee_train_volume_qa_volume_qa_pairs.json',
         f'{DATA_PROC}/knee/knee_train_volumes.json'),
    ]

    all_files = image_files + volume_files

    total = 0
    for name, path, meta_path in all_files:
        print(f'\n{"="*60}')
        print(f'Processing: {name} ({path})')
        print(f'{"="*60}')

        with open(path) as f:
            data = json.load(f)

        vol_bbox_counts = None
        if meta_path:
            vol_bbox_counts = _load_volume_bbox_counts(meta_path)

        count, details = fix_detection_plurals(data, vol_bbox_counts=vol_bbox_counts, dry_run=dry_run)
        total += count
        print(f'  Fixed: {count} entries')

        for d in details[:5]:
            slice_info = f' slice {d["slice_num"]}' if d.get("slice_num") else ''
            print(f'  {d["volume_id"]}{slice_info} (labels: {d["multi_labels"]})')
            print(f'    OLD: {d["old"]}')
            print(f'    NEW: {d["new"]}')
            print()

        if not dry_run and count > 0:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f'  Saved {path}')
        elif dry_run:
            print(f'  [DRY RUN - not saved]')

    print(f'\n{"="*60}')
    print(f'TOTAL: {total} entries fixed')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
