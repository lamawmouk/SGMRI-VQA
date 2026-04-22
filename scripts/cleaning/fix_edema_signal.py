"""
Fix edema signal descriptions based on modality:
- T1/T1POST: edema = hypointense (low signal). Fix: "hyperintense" → "hypointense",
  "increased signal" → "hypointense signal change", "signal increase" → "hypointense signal change"
- FLAIR: edema = hyperintense (high signal). Fix: "hypointensity" → "hyperintensity"

Only modifies sentences that mention "edema".
"""

import json
import re
import sys


def fix_edema_sentence_t1(sentence):
    """Fix edema sentence for T1: edema is hypointense, not hyperintense."""
    s = sentence
    # "hyperintense signal" → "hypointense signal" (only in edema context)
    s = s.replace('hyperintense signal', 'hypointense signal')
    s = s.replace('Hyperintense signal', 'Hypointense signal')
    # "Increased signal abnormality is seen" → "Hypointense signal change is seen"
    s = s.replace('Increased signal abnormality is seen in the parenchyma, consistent with edema',
                  'Hypointense signal change with architectural distortion is seen in the parenchyma, consistent with edema')
    # "Diffuse or focal signal increase is noted" → "Diffuse or focal hypointense signal change is noted"
    s = s.replace('Diffuse or focal signal increase is noted in the brain tissue, suggestive of edema',
                  'Diffuse or focal hypointense signal change is noted in the brain tissue, suggestive of edema')
    return s


def fix_edema_sentence_flair(sentence):
    """Fix edema sentence for FLAIR: edema is hyperintense, not hypointense."""
    s = sentence
    s = s.replace('hypointensity', 'hyperintensity')
    s = s.replace('Hypointensity', 'Hyperintensity')
    s = s.replace('hypointense', 'hyperintense')
    s = s.replace('Hypointense', 'Hyperintense')
    return s


def fix_edema_descriptions(qa, dry_run=False):
    """Fix edema signal descriptions based on modality."""
    count = 0
    details = []

    for e in qa:
        if e.get('task') != 'detection':
            continue

        gt = e.get('ground_truth', {})
        # Check for edema label (image level or volume level)
        labels = gt.get('labels', [])
        final_diag = gt.get('final_diagnosis', [])
        all_labels = [l.lower() for l in labels + final_diag]
        if 'edema' not in all_labels:
            continue

        modality = e.get('modality', '').upper()
        reasoning = e.get('reasoning', '')

        # Split into sentences, fix only edema-related ones
        sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', reasoning)
        modified = False
        new_sentences = []

        for sentence in sentences:
            new_sentence = sentence
            if 'edema' in sentence.lower():
                if 'FLAIR' in modality:
                    new_sentence = fix_edema_sentence_flair(sentence)
                else:  # T1, T1POST, T2, etc.
                    new_sentence = fix_edema_sentence_t1(sentence)
                if new_sentence != sentence:
                    modified = True
            new_sentences.append(new_sentence)

        if modified:
            new_reasoning = ' '.join(new_sentences)
            details.append({
                'volume_id': e.get('volume_id', ''),
                'slice_num': e.get('slice_num', ''),
                'modality': modality,
                'old': reasoning[:250],
                'new': new_reasoning[:250],
            })
            if not dry_run:
                e['reasoning'] = new_reasoning
            count += 1

    return count, details


def main():
    dry_run = '--dry-run' in sys.argv

    files = [
        ('brain_val_img', 'brain/gpt4o_brain_val_image_qa_image_qa_pairs.json'),
        ('brain_train_img', 'brain/gpt4o_brain_train_image_qa_image_qa_pairs.json'),
        ('brain_val_vol', 'brain/gpt4o_brain_val_volume_qa_volume_qa_pairs.json'),
        ('brain_train_vol', 'brain/gpt4o_brain_train_volume_qa_volume_qa_pairs.json'),
    ]

    total = 0
    for name, path in files:
        print(f'\n{"="*60}')
        print(f'Processing: {name} ({path})')
        print(f'{"="*60}')

        with open(path) as f:
            data = json.load(f)

        count, details = fix_edema_descriptions(data, dry_run=dry_run)
        total += count
        print(f'  Fixed: {count} entries')

        for d in details[:5]:
            slice_info = f' slice {d["slice_num"]}' if d.get("slice_num") else ''
            print(f'  [{d["modality"]}] {d["volume_id"]}{slice_info}')
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
