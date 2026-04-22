"""
Fix ligament localization answers missing spatial location text.
Adds anatomical location prefix before <bbx> tags for ACL, MCL, PCL, LCL entries
that currently just have "at <bbx>" or bare "<bbx>".

Also fixes false edema mentions in knee detection reasoning where edema
is not a ground truth label.
"""

import json
import random
import re
import sys

random.seed(42)

# Anatomical location prefixes for each ligament (diversified)
LIGAMENT_LOCATIONS = {
    'acl': [
        'In the anterior cruciate ligament region,',
        'In the anterior compartment of the knee,',
        'In the anterior cruciate ligament,',
        'In the intercondylar region at the ACL,',
        'In the anterior aspect of the knee at the ACL,',
        'At the anterior cruciate ligament,',
        'In the central knee at the anterior cruciate ligament,',
        'In the intercondylar notch at the ACL,',
    ],
    'mcl': [
        'In the medial compartment at the MCL,',
        'In the medial compartment at the medial collateral ligament,',
        'Along the medial aspect of the knee at the MCL,',
        'In the medial knee at the medial collateral ligament,',
    ],
    'pcl': [
        'In the posterior compartment at the PCL,',
        'In the posterior cruciate ligament region,',
        'In the posterior aspect of the knee at the PCL,',
        'At the posterior cruciate ligament,',
    ],
    'lcl': [
        'In the lateral compartment at the LCL complex,',
        'In the lateral compartment at the lateral collateral ligament complex,',
        'Along the lateral aspect of the knee at the LCL,',
        'In the lateral knee at the lateral collateral ligament complex,',
    ],
}

# Sentences to remove from detection reasoning when edema is NOT in ground truth.
# These are our enrichment-appended bone marrow edema descriptions.
EDEMA_ENRICHMENT_SENTENCES = [
    'Abnormal fluid signals in the bone marrow are noted, consistent with bone marrow edema.',
    'Fluid signal abnormality within the bone marrow is identified, suggestive of bone marrow edema.',
    'Abnormal fluid signals are seen within the osseous marrow, consistent with bone marrow edema.',
    'The bone marrow demonstrates abnormal fluid signals, indicating marrow edema.',
    'Abnormal fluid signal intensity is observed within the bone marrow, consistent with edema.',
    'Fluid signals within the bone marrow are noted with abnormal characteristics, suggestive of marrow edema.',
    'Abnormal marrow fluid signals are identified, consistent with bone marrow edema.',
    'The osseous structures demonstrate abnormal fluid signals in the marrow, representing bone marrow edema.',
    # Also GPT-4o patterns
    'There is abnormal marrow signal with fluid characteristics, consistent with bone marrow edema.',
    'There are abnormal fluid signals seen within the bone marrow, suggestive of bone marrow edema.',
]

# Fracture descriptions that mention edema - replace with ones that don't
FRACTURE_WITH_EDEMA = [
    'Fracture line with surrounding marrow edema is identified.',
    'Bone discontinuity with adjacent edema is seen, consistent with a fracture or contusion.',
]

FRACTURE_WITHOUT_EDEMA = [
    'A linear signal abnormality through the bone is noted, consistent with a fracture.',
    'Signal abnormality through the osseous structure is noted, suggestive of a fracture.',
    'A low-signal line traversing the bone marrow is identified, consistent with a fracture.',
    'Cortical disruption with surrounding marrow signal change is observed, indicating a fracture.',
    'A linear area of signal abnormality is seen within the bone, consistent with an occult fracture.',
    'Osseous signal abnormality with a linear pattern is noted, suggestive of a fracture or contusion.',
]


def fix_ligament_localization(qa, dry_run=False):
    """Add spatial location to ligament localization answers that just have bbox."""
    count = 0
    details = []

    for e in qa:
        if e.get('task') != 'localization' or e.get('level') != 'slice':
            continue

        q = e.get('question', '').lower()
        answer = e.get('answer', '')

        # Determine which ligament
        ligament = None
        for lig in ['acl', 'mcl', 'pcl', 'lcl']:
            if lig in q:
                ligament = lig
                break
        if not ligament:
            continue

        # Check if answer is missing spatial location
        before_bbx = re.split(r'<bbx>', answer)[0].strip()
        cleaned = re.sub(r'^[Aa]t\s*$', '', before_bbx).strip()
        if cleaned:
            continue  # Already has location text

        # Add location prefix
        location = random.choice(LIGAMENT_LOCATIONS[ligament])
        # Replace "at <bbx>" or bare "<bbx>" with "Location, <bbx>"
        new_answer = re.sub(r'^(?:[Aa]t\s+)?(<bbx>)', f'{location} \\1', answer)

        if new_answer != answer:
            details.append({
                'volume_id': e.get('volume_id', ''),
                'slice_num': e.get('slice_num', ''),
                'ligament': ligament.upper(),
                'old': answer[:150],
                'new': new_answer[:150],
            })
            if not dry_run:
                e['answer'] = new_answer
            count += 1

    return count, details


def fix_false_edema_detection(qa, dry_run=False):
    """Remove edema mentions from detection reasoning when edema is not a ground truth label."""
    count = 0
    details = []

    for e in qa:
        if e.get('task') != 'detection' or e.get('level') != 'slice':
            continue

        gt = e.get('ground_truth', {})
        labels = [l.lower() for l in gt.get('labels', [])]

        # Skip if edema IS a ground truth label
        if any('edema' in l for l in labels):
            continue

        reasoning = e.get('reasoning', '')
        if 'edema' not in reasoning.lower():
            continue

        new_reasoning = reasoning

        # Remove enrichment-appended edema sentences
        for sent in EDEMA_ENRICHMENT_SENTENCES:
            new_reasoning = new_reasoning.replace(' ' + sent, '')
            new_reasoning = new_reasoning.replace(sent + ' ', '')
            new_reasoning = new_reasoning.replace(sent, '')

        # Replace fracture descriptions that mention edema with clean ones
        for bad_sent in FRACTURE_WITH_EDEMA:
            if bad_sent in new_reasoning:
                replacement = random.choice(FRACTURE_WITHOUT_EDEMA)
                new_reasoning = new_reasoning.replace(bad_sent, replacement)

        # Handle GPT-4o "periligamentous edema" - replace with "periligamentous signal change"
        new_reasoning = new_reasoning.replace('Periligamentous edema and signal abnormality',
                                              'Periligamentous signal abnormality')
        new_reasoning = new_reasoning.replace('periligamentous edema and signal abnormality',
                                              'periligamentous signal abnormality')
        new_reasoning = new_reasoning.replace('periligamentous edema', 'periligamentous signal change')
        new_reasoning = new_reasoning.replace('Periligamentous edema', 'Periligamentous signal change')

        # Handle "with adjacent edema" in fracture context
        new_reasoning = new_reasoning.replace('with adjacent edema is seen', 'with adjacent marrow signal change is seen')
        new_reasoning = new_reasoning.replace('with surrounding marrow edema', 'with surrounding marrow signal change')

        # Clean up double spaces
        new_reasoning = re.sub(r'\s{2,}', ' ', new_reasoning).strip()
        # Clean up leading/trailing dots
        new_reasoning = re.sub(r'^\.\s*', '', new_reasoning)
        new_reasoning = re.sub(r'\s*\.\s*$', '.', new_reasoning)

        if new_reasoning != reasoning:
            details.append({
                'volume_id': e.get('volume_id', ''),
                'slice_num': e.get('slice_num', ''),
                'labels': gt.get('labels', []),
                'old': reasoning[:250],
                'new': new_reasoning[:250],
            })
            if not dry_run:
                e['reasoning'] = new_reasoning
            count += 1

    return count, details


def main():
    dry_run = '--dry-run' in sys.argv

    knee_files = [
        ('knee_val_img', 'knee/gpt4o_knee_val_image_qa_image_qa_pairs.json'),
        ('knee_train_img', 'knee/gpt4o_knee_train_image_qa_image_qa_pairs.json'),
    ]

    total_loc = 0
    total_edema = 0
    for name, path in knee_files:
        print(f'\n{"="*60}')
        print(f'Processing: {name} ({path})')
        print(f'{"="*60}')

        with open(path) as f:
            data = json.load(f)

        # Fix ligament localization
        loc_count, loc_details = fix_ligament_localization(data, dry_run=dry_run)
        total_loc += loc_count
        print(f'\n  Ligament localization fixed: {loc_count}')
        for d in loc_details[:3]:
            print(f'    [{d["ligament"]}] {d["volume_id"]} slice {d["slice_num"]}')
            print(f'      OLD: {d["old"]}')
            print(f'      NEW: {d["new"]}')
            print()

        # Fix false edema in detection
        edema_count, edema_details = fix_false_edema_detection(data, dry_run=dry_run)
        total_edema += edema_count
        print(f'  False edema detection fixed: {edema_count}')
        for d in edema_details[:3]:
            print(f'    {d["volume_id"]} slice {d["slice_num"]} labels={d["labels"]}')
            print(f'      OLD: {d["old"]}')
            print(f'      NEW: {d["new"]}')
            print()

        if not dry_run and (loc_count > 0 or edema_count > 0):
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f'  Saved {path}')
        elif dry_run:
            print(f'  [DRY RUN - not saved]')

    print(f'\n{"="*60}')
    print(f'TOTAL: {total_loc} localization + {total_edema} false edema = {total_loc + total_edema} fixes')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
