#!/usr/bin/env python3
"""
data_cleaning/clean.py — Dataset inventory, scope filtering, and deduplication.

Run from the repo root:
    python3 data_cleaning/clean.py \
        --dataset-root "/Users/tejasskamar/Practicum/Data Set/crack_segmentation_dataset" \
        --output-dir   data_cleaning/outputs

Steps
─────
  1. Inventory   — map every image to its source dataset and material domain
                   via filename prefix.
  2. Scope filter — keep road/pavement sources only; drop concrete-wall,
                   masonry, no-crack sources (domain-shift justification below).
  3. Deduplication — remove exact file-level duplicates (MD5) within each
                     split and across train/test (cross-split duplicates = data leakage).

Outputs
───────
  outputs/
    inventory.csv    — full catalogue of every image
    kept.csv         — images that pass all filters
    dropped.csv      — dropped images with reason
    duplicates.csv   — duplicate pairs (both intra- and cross-split)
    report.md        — human-readable summary + decisions

Domain-shift justification
──────────────────────────
Road crack topology and wall/concrete crack topology differ substantially:
  - Pavement cracks: longitudinal, transverse, alligator — driven by traffic load,
    thermal expansion, water ingress on horizontal surfaces.
  - Wall/concrete cracks: shrinkage, tension, corrosion — different branching
    patterns and edge statistics on vertical surfaces.
Training cross-material without domain adaptation degrades performance
(see CDE-Crack, Unsupervised DA for Crack Segmentation, 2023;
 Deep Domain Adaptation for Pavement Crack Detection, 2021).
If cross-material extension is needed later, change KEEP_DOMAINS below — it is
a one-line change.
"""

import os
import re
import csv
import json
import hashlib
import shutil
import argparse
from collections import defaultdict
from datetime import date


# ── Source catalogue ──────────────────────────────────────────────────────────
# Maps filename prefix → source dataset metadata.
# keep=True  → road / pavement / asphalt domain  → fits our research scope
# keep=False → non-pavement material or no cracks → domain shift risk, drop

SOURCE_CATALOGUE = {
    'CRACK': {
        'source':  'CRACK500',
        'domain':  'road-pavement',
        'keep':    True,
        'material': 'asphalt',
        'note':    'Asphalt road surface, Temple University campus; '
                   'Zhang et al. ICIP 2016 + Yang et al. arXiv 2019. '
                   'Largest pavement component (3,363 images after cropping).',
    },
    'GAPS': {
        'source':  'GAPs384',
        'domain':  'road-pavement',
        'keep':    True,
        'material': 'asphalt',
        'note':    'German Asphalt Pavement distress (GAPs), 384 crack-only '
                   'images selected from 1,969-image GAPs dataset; '
                   'Eisenbach et al. IJCNN 2017. 1.2 mm/pixel resolution.',
    },
    'CFD': {
        'source':  'CrackForest (CFD)',
        'domain':  'road-pavement',
        'keep':    True,
        'material': 'cement-road',
        'note':    'Crack Forest Dataset, 118 cement road images 480×320, '
                   'iPhone 5, Beijing; Shi et al. IEEE ITS 2016. '
                   'Mix: 5 alligator, 95 transverse, 10 longitudinal.',
    },
    'cracktree': {
        'source':  'CrackTree200',
        'domain':  'road-pavement',
        'keep':    True,
        'material': 'cement-road',
        'note':    'CrackTree200, 206 pavement images 800×600; '
                   'Zou et al. Pattern Recognition Letters 2012. '
                   'Widely used benchmark for pavement crack detection.',
    },
    'DeepCrack': {
        'source':  'DeepCrack',
        'domain':  'road-pavement',
        'keep':    True,
        'material': 'asphalt+concrete-pavement',
        'note':    'DeepCrack, 537 images 544×384 across asphalt and concrete '
                   'pavement scenes; yhlleo GitHub. Three textures: bare, '
                   'dirty, rough. Crack width 1–180 px. Clean pre-split.',
    },
    'Sylvie': {
        'source':  'AEL (Chambon)',
        'domain':  'road-pavement',
        'keep':    True,
        'material': 'road-pavement',
        'note':    'AEL dataset (38+15+5 images merged), road pavement; '
                   'Amhaz, Chambon, Idier, Baltazart — minimal path selection. '
                   '63 labels: 23 transverse, 37 longitudinal, 3 healthy.',
    },
    'forest': {
        'source':  'Forest (CFD-variant)',
        'domain':  'road-pavement',
        'keep':    True,
        'material': 'road-pavement',
        'note':    'Road pavement images visually similar to CFD. '
                   'CrackSeg9k (2022) flagged high similarity with CFD and '
                   'excluded it; kept here but deduplicated against full set.',
    },
    'Rissbilder': {
        'source':  'Rissbilder',
        'domain':  'concrete-wall',
        'keep':    False,
        'material': 'concrete-wall',
        'note':    'German "crack images"; collected by wall-climbing inspection '
                   'robot on concrete facades and building walls. '
                   'Largest non-pavement component (3,822 images = 34% of total). '
                   'DROPPED: wrong domain — vertical concrete surfaces have '
                   'different crack topology to horizontal road pavement.',
    },
    'Volker': {
        'source':  'Volker',
        'domain':  'concrete-structure',
        'keep':    False,
        'material': 'concrete-structure',
        'note':    'Concrete/building structural cracks, Bauhaus University context. '
                   'Domain not definitively confirmed but consistent with '
                   'structural inspection imagery, not road pavement. '
                   'DROPPED: uncertain domain + domain-shift risk.',
    },
    'Eugen': {
        'source':  'Eugen Muller',
        'domain':  'unknown-structure',
        'keep':    False,
        'material': 'unknown',
        'note':    'Only 55 images; unverified material domain. '
                   'CrackSeg9k (2022) explicitly discarded this source '
                   'citing "poor quality and much fewer images". '
                   'DROPPED: unverified domain + quality concerns.',
    },
    'noncrack': {
        'source':  'NonCrack (concrete wall)',
        'domain':  'concrete-wall',
        'keep':    False,
        'material': 'concrete-wall',
        'note':    'Negative samples — images with NO cracks from concrete wall '
                   'surfaces (filename: noncrack_noncrack_concrete_wall_*). '
                   'DROPPED: wrong domain (wall, not road) AND no crack signal '
                   '(all-black masks produce empty graphs in Stage 2).',
    },
}

KEEP_DOMAINS = {'road-pavement'}   # ← one-line change to go cross-material


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_prefix(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    m = re.match(r'^([A-Za-z]+)', stem)
    return m.group(1) if m else '__unknown__'


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run(dataset_root: str, output_dir: str, copy_files: bool):
    os.makedirs(output_dir, exist_ok=True)

    splits = ['train', 'test']
    inventory   = []   # all images
    kept        = []   # after scope filter
    dropped     = []   # scope-filtered out
    duplicates  = []   # exact-duplicate pairs

    # ── Step 1: Inventory ────────────────────────────────────────────────────
    print('\n── Step 1: Inventory ────────────────────────────────')
    for split in splits:
        img_dir  = os.path.join(dataset_root, split, 'images')
        mask_dir = os.path.join(dataset_root, split, 'masks')
        if not os.path.isdir(img_dir):
            print(f'  [skip] {split}/images not found')
            continue

        files = sorted(os.listdir(img_dir))
        print(f'  {split}: {len(files)} images')

        for fname in files:
            img_path  = os.path.join(img_dir,  fname)
            mask_path = os.path.join(mask_dir, fname)
            prefix    = get_prefix(fname)
            meta      = SOURCE_CATALOGUE.get(prefix, {
                'source':   f'UNKNOWN ({prefix})',
                'domain':   'unknown',
                'keep':     False,
                'material': 'unknown',
                'note':     'Prefix not in catalogue — dropped by default.',
            })

            row = {
                'split':    split,
                'filename': fname,
                'prefix':   prefix,
                'source':   meta['source'],
                'domain':   meta['domain'],
                'material': meta['material'],
                'keep':     meta['keep'],
                'note':     meta['note'],
                'img_path': img_path,
                'mask_path': mask_path,
                'md5':      '',      # filled in step 3
            }
            inventory.append(row)

    print(f'\n  Total images inventoried: {len(inventory)}')

    # ── Step 2: Scope filter ──────────────────────────────────────────────────
    print('\n── Step 2: Scope filter ─────────────────────────────')
    counts = defaultdict(lambda: defaultdict(int))
    for row in inventory:
        prefix = row['prefix']
        if row['keep']:
            kept.append(row)
            counts[prefix]['kept'] += 1
        else:
            reason = f"domain={row['domain']} — {row['note'].split('.')[0]}"
            row['drop_reason'] = reason
            dropped.append(row)
            counts[prefix]['dropped'] += 1

    print(f'\n  {"Prefix":<12} {"Source":<22} {"Domain":<22} {"Decision":>8} {"Count":>6}')
    print('  ' + '─' * 76)
    for prefix, meta in sorted(SOURCE_CATALOGUE.items(),
                                key=lambda x: not x[1]['keep']):
        k = counts[prefix]['kept']
        d = counts[prefix]['dropped']
        total = k + d
        decision = '✅ KEEP' if meta['keep'] else '❌ DROP'
        print(f'  {prefix:<12} {meta["source"]:<22} {meta["domain"]:<22} '
              f'{decision:>8} {total:>6}')

    print(f'\n  Kept:    {len(kept):>5} images')
    print(f'  Dropped: {len(dropped):>5} images')

    # ── Step 3: Deduplication ─────────────────────────────────────────────────
    print('\n── Step 3: Deduplication (MD5) ──────────────────────')
    print('  Computing hashes...')

    hash_to_row = {}    # md5 → first row seen
    dedup_kept  = []
    n_dup = 0

    for row in kept:
        h = md5(row['img_path'])
        row['md5'] = h

        if h in hash_to_row:
            first = hash_to_row[h]
            duplicates.append({
                'duplicate_split':   row['split'],
                'duplicate_file':    row['filename'],
                'kept_split':        first['split'],
                'kept_file':         first['filename'],
                'md5':               h,
                'cross_split':       row['split'] != first['split'],
            })
            row['drop_reason'] = (f"exact duplicate of "
                                  f"{first['split']}/{first['filename']}")
            dropped.append(row)
            n_dup += 1
        else:
            hash_to_row[h] = row
            dedup_kept.append(row)

    cross_split_dups = sum(1 for d in duplicates if d['cross_split'])
    print(f'  Duplicates removed : {n_dup}')
    print(f'    cross-split (data-leakage risk): {cross_split_dups}')
    print(f'  Final clean set    : {len(dedup_kept)} images')
    final_train = sum(1 for r in dedup_kept if r['split'] == 'train')
    final_test  = sum(1 for r in dedup_kept if r['split'] == 'test')
    print(f'    train: {final_train}   test: {final_test}')

    # ── Write CSVs ────────────────────────────────────────────────────────────
    print('\n── Writing outputs ──────────────────────────────────')
    csv_fields_inv  = ['split','filename','prefix','source','domain','material',
                       'keep','md5','note','img_path','mask_path']
    csv_fields_drop = csv_fields_inv + ['drop_reason']
    csv_fields_dup  = ['duplicate_split','duplicate_file','kept_split',
                       'kept_file','md5','cross_split']

    def write_csv(path, rows, fields):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f'  Saved {len(rows):>5} rows → {path}')

    write_csv(os.path.join(output_dir, 'inventory.csv'),
              inventory, csv_fields_inv)
    write_csv(os.path.join(output_dir, 'kept.csv'),
              dedup_kept, csv_fields_inv)
    write_csv(os.path.join(output_dir, 'dropped.csv'),
              dropped, csv_fields_drop)
    write_csv(os.path.join(output_dir, 'duplicates.csv'),
              duplicates, csv_fields_dup)

    # ── Optional file copy ────────────────────────────────────────────────────
    if copy_files:
        print('\n── Copying clean files ──────────────────────────────')
        for split in splits:
            for subdir in ['images', 'masks']:
                os.makedirs(os.path.join(output_dir, 'clean',
                                          split, subdir), exist_ok=True)
        for row in dedup_kept:
            for key, subdir in [('img_path', 'images'), ('mask_path', 'masks')]:
                src = row[key]
                dst = os.path.join(output_dir, 'clean', row['split'],
                                    subdir, row['filename'])
                if os.path.exists(src):
                    shutil.copy2(src, dst)
        print(f'  Clean files copied → {os.path.join(output_dir, "clean")}')

    # ── Report ────────────────────────────────────────────────────────────────
    _write_report(output_dir, inventory, dedup_kept, dropped,
                  duplicates, cross_split_dups, dataset_root)

    print(f'\nDone. All outputs in: {output_dir}\n')
    return dedup_kept


def _write_report(output_dir, inventory, kept, dropped, duplicates,
                   cross_split_dups, dataset_root):
    final_train = sum(1 for r in kept if r['split'] == 'train')
    final_test  = sum(1 for r in kept if r['split'] == 'test')

    # per-source breakdown of kept
    src_kept = defaultdict(int)
    for r in kept:
        src_kept[r['source']] += 1

    src_dropped = defaultdict(lambda: {'count': 0, 'reason': ''})
    for r in dropped:
        src_dropped[r['source']]['count'] += 1
        if not src_dropped[r['source']]['reason']:
            meta = SOURCE_CATALOGUE.get(r['prefix'], {})
            src_dropped[r['source']]['reason'] = meta.get('domain', 'unknown')

    lines = [
        f'# Dataset Cleaning Report',
        f'**Date:** {date.today()}  ',
        f'**Source:** `{dataset_root}`',
        '',
        '---',
        '',
        '## Summary',
        '',
        f'| | Count |',
        f'|--|--|',
        f'| Original images | {len(inventory)} |',
        f'| Dropped (wrong domain) | {len(dropped) - len(duplicates)} |',
        f'| Dropped (exact duplicates) | {len(duplicates)} |',
        f'| **Final clean set** | **{len(kept)}** |',
        f'| — train | {final_train} |',
        f'| — test  | {final_test} |',
        '',
        '---',
        '',
        '## Step 1 — Source Inventory',
        '',
        'Eleven filename prefixes found across all images:',
        '',
        '| Prefix | Source Dataset | Domain / Material | Count | Decision |',
        '|--------|---------------|-------------------|-------|----------|',
    ]

    for prefix, meta in sorted(SOURCE_CATALOGUE.items(),
                                key=lambda x: not x[1]['keep']):
        total = src_kept.get(meta['source'], 0) + \
                src_dropped.get(meta['source'], {}).get('count', 0)
        decision = '✅ KEEP' if meta['keep'] else '❌ DROP'
        lines.append(
            f'| `{prefix}` | {meta["source"]} | {meta["domain"]} '
            f'| {total} | {decision} |'
        )

    lines += [
        '',
        '---',
        '',
        '## Step 2 — Scope Filter',
        '',
        '**Rule:** keep only `domain = road-pavement`.',
        '',
        '### Kept sources',
        '',
    ]
    for prefix, meta in SOURCE_CATALOGUE.items():
        if not meta['keep']:
            continue
        n = src_kept.get(meta['source'], 0)
        lines.append(f'- **{meta["source"]}** (`{prefix}`) — {n} images  ')
        lines.append(f'  {meta["note"]}')

    lines += [
        '',
        '### Dropped sources',
        '',
    ]
    for prefix, meta in SOURCE_CATALOGUE.items():
        if meta['keep']:
            continue
        n = src_dropped.get(meta['source'], {}).get('count', 0)
        lines.append(f'- **{meta["source"]}** (`{prefix}`) — {n} images  ')
        lines.append(f'  {meta["note"]}')

    lines += [
        '',
        '**Domain-shift justification:**  ',
        'Road crack topology (horizontal, traffic-driven: longitudinal, transverse, alligator)',
        'differs structurally from wall/concrete crack topology (vertical, shrinkage/tension-driven).',
        'Cross-material training without domain adaptation degrades segmentation and topology',
        'prediction performance (CDE-Crack; Unsupervised DA for Crack Segmentation, Automation',
        'in Construction 2023; Deep Domain Adaptation for Pavement Crack Detection, 2021).',
        '',
        '> **To go cross-material:** change `KEEP_DOMAINS` in `clean.py` — one-line change.',
        '',
        '---',
        '',
        '## Step 3 — Deduplication',
        '',
        f'Method: MD5 hash of raw image file bytes. First occurrence kept.',
        '',
        f'| | Count |',
        f'|--|--|',
        f'| Duplicate pairs found | {len(duplicates)} |',
        f'| Cross-split duplicates (data leakage) | {cross_split_dups} |',
        f'| Images removed | {len(duplicates)} |',
        '',
    ]

    if cross_split_dups > 0:
        lines += [
            '⚠️  **Cross-split duplicates detected** — the same image appeared in',
            'both train and test splits. These are removed from the test split to',
            'prevent data leakage. See `duplicates.csv` for the full list.',
            '',
        ]
    else:
        lines.append('✅ No cross-split duplicates found — train/test integrity confirmed.')
        lines.append('')

    lines += [
        '---',
        '',
        '## Final Clean Dataset',
        '',
        f'| Split | Images |',
        f'|-------|--------|',
        f'| train | {final_train} |',
        f'| test  | {final_test} |',
        f'| **total** | **{len(kept)}** |',
        '',
        '### Source breakdown (kept only)',
        '',
        '| Source | Images |',
        '|--------|--------|',
    ]
    for src, n in sorted(src_kept.items(), key=lambda x: -x[1]):
        lines.append(f'| {src} | {n} |')

    lines += [
        '',
        '---',
        '',
        '## Output Files',
        '',
        '| File | Contents |',
        '|------|----------|',
        '| `inventory.csv` | All 11,298 images with prefix, source, domain, keep flag |',
        '| `kept.csv` | Final clean set — use this as input to `build_dataset.py` |',
        '| `dropped.csv` | All dropped images with reason |',
        '| `duplicates.csv` | Duplicate pairs with cross-split flag |',
        '| `report.md` | This file |',
        '| `clean/` | Copied clean images (only if `--copy` flag used) |',
        '',
        '### Using `kept.csv` with Stage 2',
        '',
        '```python',
        '# Read the manifest to get clean image/mask paths',
        'import csv',
        'with open("data_cleaning/outputs/kept.csv") as f:',
        '    rows = list(csv.DictReader(f))',
        'train_masks = [r["mask_path"] for r in rows if r["split"] == "train"]',
        'test_masks  = [r["mask_path"] for r in rows if r["split"] == "test"]',
        '```',
    ]

    report_path = os.path.join(output_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Report saved → {report_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Dataset cleaning: inventory, scope filter, deduplication.'
    )
    p.add_argument(
        '--dataset-root',
        default='/Users/tejasskamar/Practicum/Data Set/crack_segmentation_dataset',
        help='Root of crack_segmentation_dataset (contains train/ and test/).',
    )
    p.add_argument(
        '--output-dir',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs'),
        help='Where to write all output files.',
    )
    p.add_argument(
        '--copy', action='store_true', default=False,
        help='Copy clean images to outputs/clean/. '
             'Without this flag only CSVs and report are written.',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.dataset_root, args.output_dir, copy_files=args.copy)
