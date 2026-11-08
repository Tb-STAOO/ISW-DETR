"""Extract measurement-only transactions and an ordered RGB manifest.

The released legacy annotation format stores acquired/generated weight fields
beside object annotations. Only the seven measurement fields, event order and
scene totals are exported. Categories, boxes and owner labels are not exported.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

FEATURES = ('delta_weight_g', 'stable_before_g', 'stable_after_g',
            'stable_std_g', 'stable_range_g', 'stable_drift_g', 'stable_samples')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def measured_record(image, rows):
    events = sorted((r for r in rows if not r.get('iscrowd', 0)),
                    key=lambda row: (int(row.get('placement_order', 0)), int(row.get('id', 0))))
    if not events or len(events) > 8:
        raise ValueError('Expected one to eight measurement events')
    tokens = []
    for event in events:
        features = event['weight_features']
        values = [float(features[name]) for name in FEATURES]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Nonfinite measurement field')
        tokens.append(values)
    record = dict(schema_version=1, weight_tokens_raw=tokens,
                  weight_mask=[True] * len(tokens),
                  event_order=[int(row.get('placement_order', i + 1)) for i, row in enumerate(events)],
                  scene_total_independent=bool(image.get('scene_total_independent', False)))
    total, baseline = image.get('scene_total_weight_g'), image.get('empty_baseline_g')
    available = total is not None and baseline is not None and math.isfinite(float(total)) and math.isfinite(float(baseline))
    record['scene_total_available'] = available
    if available:
        record.update(scene_total_weight_g=float(total), empty_baseline_g=float(baseline))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, help='The fixed legacy split annotation JSON; targets are kept separate')
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--output', required=True, help='Fresh input-manifest directory')
    args = parser.parse_args()
    annotation = Path(args.annotations).resolve(strict=True)
    images = Path(args.image_root).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a fresh output directory')
    source = json.loads(annotation.read_text(encoding='utf-8-sig'))
    by_image = defaultdict(list)
    for row in source['annotations']:
        by_image[int(row['image_id'])].append(row)
    ordered = sorted(source['images'], key=lambda row: int(row['id']))
    if len({int(row['id']) for row in ordered}) != len(ordered):
        raise ValueError('Duplicate image IDs')
    output.mkdir(parents=True, exist_ok=True)
    transactions = output / 'transactions'
    transactions.mkdir()
    manifest, mapping = [], []
    for image in ordered:
        image_id = int(image['id'])
        path = images / image['file_name']
        if not path.is_file():
            raise FileNotFoundError(path)
        record = measured_record(image, by_image[image_id])
        # Geometry/category perturbation must not influence exported inputs.
        poisoned = [{**row, 'category_id': -999, 'bbox': [-999] * 4} for row in by_image[image_id]]
        if measured_record(image, poisoned) != record:
            raise RuntimeError('Target information affected exported measurements')
        transaction = transactions / f'{image_id:06d}.json'
        write(transaction, record)
        manifest.append(dict(sample_id=image_id, image=str(path.resolve()), transaction=str(transaction),
                             source_image_id_provenance_only=image_id, labels_or_boxes_in_input=False))
        mapping.append(dict(image_id=image_id, image_sha256=sha(path), transaction_sha256=sha(transaction)))
    write(output / 'manifest.json', dict(schema_version=1, samples=manifest))
    write(output / 'PREPARATION_QA.json', dict(passed=True, samples=len(manifest),
          annotation_sha256=sha(annotation), manifest_sha256=sha(output / 'manifest.json'),
          ordering='ascending integer image ID; inference uses real unpadded tail batches',
          target_poisoning_unchanged=True, feature_order=list(FEATURES), files=mapping))
    print(json.dumps(dict(passed=True, samples=len(manifest), manifest=str(output / 'manifest.json'))))


if __name__ == '__main__':
    main()
