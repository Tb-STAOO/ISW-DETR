"""Execute complete checkpoint-to-results pipelines in an independent directory.

Detector training has separate explicit tools. Controls can optionally be fitted.
It never copies historical AP values into newly computed results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['private','public','all','public-diagnostic'],default='all')
    p.add_argument('--assets-root',required=True,help='Imported resource groups as documented in README')
    p.add_argument('--public-data-root',help='Prepared 8727-image C-CCM data; required for public/all')
    p.add_argument('--output',required=True,help='New directory outside the code package')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--bootstrap-workers',type=int,default=8)
    p.add_argument('--measure-runtime',action='store_true',help='Run original CUDA benchmark while GPU is otherwise idle')
    p.add_argument('--private-weights-root',help='Optional freshly trained detector_seed*.pth exports; leaves reference resources intact')
    p.add_argument('--fit-private-controls',action='store_true',help='Fit BQR/confidence/MAP on training sources for the supplied detector weights')
    p.add_argument('--public-weights-root',help='Optional freshly trained public base/flat/is_seed*/best.pth directory')
    p.add_argument('--external-public-results',help='For phase private, include an existing complete public evaluation in the generated tables')
    args=p.parse_args();out=Path(args.output).resolve();assets=Path(args.assets_root).resolve()
    if out==ROOT or ROOT in out.parents:p.error('Outputs must be outside the source package')
    if out.exists() and any(out.iterdir()):p.error('Use a fresh output directory')
    if args.phase in ['public','all'] and not args.public_data_root:p.error('--public-data-root is required')
    if args.measure_runtime and args.device!='cuda':p.error('Reference runtime protocol requires CUDA')
    if args.fit_private_controls and args.phase not in ['private','all']:p.error('Private control fitting requires a private phase')
    if args.measure_runtime and args.phase=='public-diagnostic':p.error('The metadata diagnostic does not measure detector runtime')
    if args.external_public_results and args.phase!='private':p.error('External public table results are only used with phase private')
    external_public=Path(args.external_public_results).resolve() if args.external_public_results else None
    external_public_hash=sha(external_public) if external_public else None
    if external_public:
        existing=json.loads(external_public.read_text(encoding='utf8'))
        keys={(r['visual_variant'],r['training_seed'],r['method']) for r in existing['rows']}
        if existing['status']!='complete' or existing['total_configurations']!=63 or len(keys)!=63:p.error('External public results must contain all 63 unique configurations')
    out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir()
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',ISW_CONFIG_CACHE=str(out/'runtime_configs'))
    rows=[]
    def execute(name,tool,arguments):
        script=ROOT/'tools'/tool
        cmd=[sys.executable,'-X','utf8','-B',str(script)]+[str(v) for v in arguments]
        log=out/'logs'/(name+'.log')
        with log.open('w',encoding='utf8') as stream:result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        rows.append(dict(step=name,command=cmd,tool_sha256=sha(script),exit_code=result.returncode,log=str(log),log_sha256=sha(log)))
        (out/'PIPELINE_PROGRESS.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
        if result.returncode:raise RuntimeError('Pipeline stopped; inspect '+str(log))
        print(json.dumps(dict(step=name,status='complete')),flush=True)
    mass_diagnostic=args.phase in ['public','all','public-diagnostic']
    if mass_diagnostic:
        execute('public_mass_diagnostic','run_mass_diagnostic.py',['--metadata-root',assets/'public_metadata','--output-dir',out/'public_mass_diagnostic'])
    if args.phase!='public-diagnostic':
        execute('environment','check_environment.py',['--device',args.device,'--output',out/'ENVIRONMENT.json'])
    private=args.phase in ['private','all'];public=args.phase in ['public','all']
    if private:
        protocol=assets/'private_inference/protocol';weights=Path(args.private_weights_root).resolve() if args.private_weights_root else assets/'private_inference/weights'
        dataset=assets/'private_dataset';controls=assets/'private_controls'
        execute('private_inference','run_scalevision.py',['--protocol-root',protocol,'--dataset-root',dataset,'--weights-root',weights,'--output',out/'private','--device',args.device])
        if args.fit_private_controls:
            execute('router_cache','export_router_cache.py',['--dataset-root',dataset,'--protocol-root',protocol,'--weights-root',weights,
                '--validation-inference-root',out/'private','--output',out/'router_cache','--device',args.device])
            for method in ['bqr','confidence','map']:
                execute('fit_'+method,'train_router_controls.py',['--method',method,'--cache-root',out/'router_cache',
                    '--protocol-root',protocol,'--output',out/('fitted_'+method)])
            controls=out/'fitted_controls'
            for method in ['bqr','confidence']:
                (controls/method).mkdir(parents=True)
                for seed in [42,3407,2026]:shutil.copy2(out/f'fitted_{method}/seed{seed}.pth',controls/f'{method}/seed{seed}.pth')
            shutil.copy2(out/'fitted_map/map_selection.json',controls/'map_selection.json')
        common=['--inference-root',out/'private','--protocol-root',protocol,'--weights-root',weights]
        execute('mass_evidence','check_mass_evidence.py',['--dataset-root',dataset,'--inference-root',out/'private',
            '--weights-root',weights,'--control-weights-root',controls,'--output',out/'mass_evidence'])
        execute('private_controls','run_private_controls.py',common+['--control-weights-root',controls,'--output',out/'controls'])
        execute('validation_analysis','run_validation_analysis.py',common+['--control-weights-root',controls,'--output',out/'validation_analysis'])
        execute('shared_pool','run_shared_pool_ablation.py',common+['--output',out/'shared_pool'])
        execute('bootstrap','paired_bootstrap.py',['--controls-root',out/'controls','--annotations',protocol/'inputs/holdout.json','--output',out/'bootstrap','--workers',args.bootstrap_workers])
        execute('diagnostics','run_diagnostics.py',common+['--controls-root',out/'controls','--output',out/'diagnostics'])
        execute('figures','render_figures.py',['--validation-analysis',out/'validation_analysis','--diagnostics',out/'diagnostics','--bootstrap',out/'bootstrap','--protocol-root',protocol,'--dataset-root',dataset,'--output',out/'figures'])
        if args.measure_runtime:
            execute('benchmark','benchmark_scalevision.py',['--inference-root',out/'private','--weights-root',weights,'--control-weights-root',controls,'--output',out/'benchmark'])
            execute('batch1_accuracy','run_scalevision.py',['--protocol-root',protocol,'--dataset-root',dataset,'--weights-root',weights,
                '--output',out/'batch1_accuracy','--device',args.device,'--batch-size',1,'--splits','validation'])
    if public:
        public_weights=Path(args.public_weights_root).resolve() if args.public_weights_root else assets/'public_checkpoints/cccm'
        execute('public_evaluation','run_public_evaluation.py',['--data-root',args.public_data_root,'--weights-root',public_weights,'--device',args.device,'--output',out/'public','--reference-policy','report'])
        execute('public_sensitivity','run_public_sensitivity.py',['--public-results',out/'public/PUBLIC_RESULTS.json','--data-root',args.public_data_root,'--output',out/'public_sensitivity'])
    if private:
        table_args=['--controls',out/'controls/PRIVATE_CONTROLS.json','--validation-analysis',out/'validation_analysis','--output',out/'tables']
        if public:table_args+=['--public-results',out/'public/PUBLIC_RESULTS.json']
        elif external_public:table_args+=['--public-results',external_public]
        if args.measure_runtime:table_args+=['--benchmark',out/'benchmark/COMPUTATIONAL_SUMMARY.json']
        execute('tables','render_tables.py',table_args)
    if external_public and sha(external_public)!=external_public_hash:raise RuntimeError('External public results changed during table generation')
    report=dict(status='complete',phase=args.phase,manuscript='20261103_02',package_release='20261103_02',
        diagnostic_only=args.phase=='public-diagnostic',mass_diagnostic_executed=mass_diagnostic,
        detection_evaluation_performed=args.phase in ['private','public','all'],
        checkpoint_to_results_pipeline_only=args.phase!='public-diagnostic' and not args.fit_private_controls,
        fresh_model_training=args.fit_private_controls,detector_training_performed=False,auxiliary_control_training=args.fit_private_controls,
        external_public_results=None if not external_public else dict(path=str(external_public),sha256=external_public_hash),device=None if args.phase=='public-diagnostic' else args.device,steps=rows,
        cross_hardware_bitwise_equivalence_claimed=False,
        runtime_table_measured=args.measure_runtime,
        manual_figure_review_required=True)
    (out/'PIPELINE_RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
