"""Recompute the prelocked 2801-image non-pilot public sensitivity analysis."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reference_sources'))
from public_sensitivity import subset_metrics, canonical, stats
from run_public_evaluation import read,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public-results',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):p.error('Use a fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    source=read(args.public_results)
    if source['status']!='complete' or source['total_configurations']!=63:raise ValueError('Complete 63-configuration evaluation required')
    lock_path=ROOT/'configs/cccm/SENSITIVITY_SUBSET_LOCK.json';lock=read(lock_path)
    pilot=ROOT/'configs/cccm/previous_pilot_selection.json'
    if sha(pilot)!=lock['previous_pilot_selection_sha256']:raise ValueError('Pilot selection changed')
    annotation=Path(args.data_root).resolve()/'annotations/test.json'
    if sha(annotation)!=lock['full_test_annotation_sha256']:raise ValueError('Test annotation changed')
    ids=lock['additional_test_ids']
    membership=dict(full=lock['full_test_ids'],previous=lock['previous_pilot_test_ids'],additional=ids)
    if len(ids)!=2801 or len(set(ids))!=2801 or set(ids)&set(lock['previous_pilot_test_ids']) or canonical(membership)!=lock['membership_canonical_sha256']:
        raise ValueError('Prelocked sensitivity membership changed')
    configs=[('flat','rgb'),('is','original_original'),('is','mean_half')]
    if lock['configurations']!=[dict(visual_variant=v,method=m) for v,m in configs] or lock['seeds']!=[42,3407,2026]:raise ValueError('Sensitivity settings changed')
    records={(r['variant'],r['seed']):r for r in source['source_results']};rows=[]
    for seed in [42,3407,2026]:
        for variant,method in configs:
            entry=records[variant,seed];summary_path=Path(entry['path'])
            if sha(summary_path)!=entry['sha256']:raise ValueError('Full evaluation summary changed')
            summary=read(summary_path)
            prediction=summary_path.parent/f'test_{method}_predictions.json'
            expected=summary['prediction_sha256'].get(str(prediction))
            if not expected or sha(prediction)!=expected:raise ValueError('Full predictions changed')
            metrics=subset_metrics(annotation,prediction,ids,out/f'{variant}_{method}_seed{seed}_coco.txt')
            rows.append(dict(training_seed=seed,visual_variant=variant,method=method,metrics=metrics,
                full_test_AP=summary['metrics'][method]['AP'],additional_minus_full_AP_pp=metrics['AP']-summary['metrics'][method]['AP'],predictions_sha256=expected))
            print(json.dumps(dict(seed=seed,variant=variant,method=method,AP=metrics['AP'])),flush=True)
    lookup={(r['training_seed'],r['visual_variant'],r['method']):r['metrics'] for r in rows}
    gains=dict(full_vs_equal_budget_rgb=stats(lookup[s,'is','mean_half']['AP']-lookup[s,'flat','rgb']['AP'] for s in [42,3407,2026]),
               algorithm_vs_fixed_same_mass=stats(lookup[s,'is','mean_half']['AP']-lookup[s,'is','original_original']['AP'] for s in [42,3407,2026]))
    result=dict(status='complete',images=2801,excluded_previous_pilot_images=168,rows=rows,gain_mean_sample_sd=gains,
        subset_lock_sha256=sha(lock_path),full_result_sha256=sha(args.public_results),
        scope='Prelocked exclusion of previously observed pilot frames; not an independent-recording or blind test')
    (out/'PUBLIC_SENSITIVITY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
