"""Evaluate saved batch inference separately from the inference input boundary."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--predictions', required=True, help='Batch output directory')
    parser.add_argument('--output', required=True, help='New evaluation output directory')
    parser.add_argument('--field', choices=['detections', 'visual_detections', 'flat_visual_detections'], default='detections')
    args = parser.parse_args()
    source = Path(args.predictions).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a fresh evaluation directory')
    output.mkdir(parents=True, exist_ok=True)
    provenance = read(source / 'BATCH_PROVENANCE.json')
    expected = {int(row['id']) for row in read(args.annotations)['images']}
    found, records, hashes = [], [], {}
    for item in provenance['outputs']:
        path = source / item['output']
        payload = read(path)
        image_id = int(payload['sample_id'])
        if payload['ground_truth_used'] is not False:
            raise ValueError('Unexpected inference provenance')
        found.append(image_id)
        hashes[item['output']] = sha(path)
        detections = payload[args.field]
        for box, score, category in zip(detections['boxes'], detections['scores'], detections['labels']):
            x1, y1, x2, y2 = box
            records.append(dict(image_id=image_id, category_id=int(category), score=float(score),
                                bbox=[x1, y1, max(0., x2 - x1), max(0., y2 - y1)]))
    if len(found) != len(set(found)) or set(found) != expected:
        raise ValueError('Prediction IDs differ from the complete annotated split')
    with contextlib.redirect_stdout(io.StringIO()):
        ground = COCO(str(Path(args.annotations).resolve()))
        detection = ground.loadRes(records)
        evaluator = COCOeval(ground, detection, 'bbox')
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    names = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
             'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
    metrics = {name: float(value * 100) for name, value in zip(names, evaluator.stats)}
    (output / 'coco_predictions.json').write_text(json.dumps(records), encoding='utf-8')
    result = dict(images=len(found), seed=provenance['seed'], field=args.field,
                  metrics_percent=metrics, prediction_hashes=hashes,
                  annotation_sha256=sha(args.annotations),
                  batch_provenance_sha256=sha(source / 'BATCH_PROVENANCE.json'))
    (output / 'METRICS.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(images=len(found), seed=provenance['seed'], metrics_percent=metrics)))


if __name__ == '__main__':
    main()
