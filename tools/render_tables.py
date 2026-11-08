"""Generate editable manuscript result tables from recomputed metrics."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from table_display import base_transform,highlight_maxima,CENTER,table_shell


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def stats(rows, fraction=True):
    if fraction and len(rows)==3 and {row.get('seed') for row in rows}=={42,3407,2026}:
        rows=sorted(rows,key=lambda row:[42,3407,2026].index(row['seed']))
    factor = 100 if fraction else 1
    ap = [row['ap' if fraction else 'AP'] * factor for row in rows]
    if len(ap) != 3:
        raise ValueError('Every manuscript entry requires all three seed values')
    if fraction and {row.get('seed') for row in rows}!={42,3407,2026}:
        raise ValueError('Private table rows must contain the three distinct declared seeds')
    result = dict(AP_mean=statistics.mean(ap), AP_sample_SD=statistics.stdev(ap), AP_seeds=ap)
    if fraction:
        result.update({key: statistics.mean(row[key] * 100 for row in rows) for key in ['ap50', 'ap75', 'ar100']})
    return result


def mean_cell(row):
    return f"${row['AP_mean']:.2f}$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', required=True, help='PRIVATE_CONTROLS.json')
    parser.add_argument('--validation-analysis', required=True)
    parser.add_argument('--public-results', help='Completed PUBLIC_RESULTS.json; omit only while public reproduction is pending')
    parser.add_argument('--benchmark', help='Fresh five-path computational summary; required to generate the cost table')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a fresh table output directory')
    output.mkdir(parents=True, exist_ok=True)
    rows = read(args.controls)['rows']
    grouped = {(split, method): stats([row for row in rows if row['split'] == split and row['method'] == method])
               for split in ['validation', 'internal', 'holdout'] for method in sorted({row['method'] for row in rows})}
    methods = [('rt_detr_visual', 'RT-DETRv2', 'RGB'), ('identity_visual', 'I--S visual', 'RGB'),
               ('matched_gaussian', 'Gaussian fusion', 'RGB + mass'), ('original_fixed_CVSR', 'CVSR reference', 'RGB + mass'),
               ('joint_map_adapted', 'MAP-style control', 'RGB + mass'), ('confidence_prediction_adapted', 'Confidence-guided control', 'RGB + mass'),
               ('original_BQR', 'BQR', 'RGB + mass'), ('half_readout_mean_assignment', r'\textbf{ISW-DETR}', 'RGB + mass')]
    lines = [r'\begin{tabular}{@{}lccrrrr@{}}', r'\toprule', r'& & Internal & \multicolumn{4}{c}{Measured transactions} \\',
             r'\cmidrule(lr){4-7}', r'Method & Input & AP & AP & AP$_{50}$ & AP$_{75}$ & AR$_{100}$ \\', r'\midrule']
    for method, label, inputs in methods:
        internal, held = grouped['internal', method], grouped['holdout', method]
        lines.append(f"{label} & {inputs} & {mean_cell(internal)} & {mean_cell(held)} & {held['ap50']:.2f} & {held['ap75']:.2f} & {held['ar100']:.2f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (output / 'state_table_main.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    ablations = [('original_fixed_CVSR', 'Unmodified', 'Unmodified', 'Yes'),
                 ('half_readout_original_assignment', 'Unmodified', 'Half contrast', 'Yes'),
                 ('half_readout_max_assignment', 'Group max', 'Half contrast', 'Yes'),
                 ('full_max_erasure', 'Group max', 'Group max', 'Yes'),
                 ('full_mean_erasure', 'Group mean', 'Group mean', 'Yes'),
                 ('no_IS_new', 'Group mean', 'Half contrast', 'No'),
                 ('half_readout_mean_assignment', 'Group mean', 'Half contrast', 'Yes')]
    lines = [r'\begin{tabular}{@{}lllrr@{}}', r'\toprule', r'\multicolumn{5}{@{}l}{\textbf{A. Internal and measured evaluation}} \\',
             r'Association & Visual score mapping & I--S & Internal AP & Measured AP \\', r'\midrule']
    for method, association, mapping, is_used in ablations:
        lines.append(f'{association} & {mapping} & {is_used} & {mean_cell(grouped["internal", method])} & {mean_cell(grouped["holdout", method])} ' + r'\\')
    lines += [r'\midrule', r'\multicolumn{5}{@{}l}{\textbf{B. Validation-only score controls}} \\',
              r'\multicolumn{4}{@{}l}{Control} & Validation AP \\', r'\midrule']
    calibration = read(Path(args.validation_analysis) / 'CALIBRATION_CONTROL_RESULTS.json')['rows']
    controls = [('original_fixed_CVSR', 'CVSR reference (no contrast control)'),
                ('global_scale_0.5', r'Global candidate scaling, $g=0.5$'),
                ('global_scale_0.75', r'Global candidate scaling, $g=0.75$'),
                ('global_scale_1.25', r'Global candidate scaling, $g=1.25$'),
                ('global_scale_1.5', r'Global candidate scaling, $g=1.5$'),
                ('half_lse_preserving_reassign', 'LSE-preserving half contrast'),
                ('half_readout_mean_assignment', 'ISW-DETR (mean association + half-contrast refinement)')]
    for method, label in controls:
        value = grouped.get(('validation', method))
        if value is None:
            value = stats([row for row in calibration if row['method'] == method])
        lines.append(r'\multicolumn{4}{@{}l}{' + label + '} & ' + mean_cell(value) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (output / 'state_table_ablation.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    tables = ['state_table_main.tex', 'state_table_ablation.tex']
    if args.public_results:
        public = read(args.public_results)
        keys={(r['visual_variant'],r['training_seed'],r['method']) for r in public['rows']}
        methods_public={r['method'] for r in public['rows']}
        expected={(v,seed,m) for v in ['base','flat','is'] for seed in [42,3407,2026] for m in methods_public}
        if public['status']!='complete' or public['total_configurations']!=63 or len(public['rows'])!=63 or len(methods_public)!=7 or keys!=expected:
            raise ValueError('The public table requires all63distinct configurations with all three seeds')
        public_grouped={}
        for variant in ['base','flat','is']:
            for method in sorted(methods_public):
                subset=sorted([r for r in public['rows'] if r['visual_variant']==variant and r['method']==method],key=lambda r:[42,3407,2026].index(r['training_seed']))
                public_grouped[variant,method]=stats([r['metrics'] for r in subset],fraction=False)
        # Three continuous model-level rows; stronger controls remain in TABLE_METRICS.json.
        entries=[('base','rgb','RT-DETRv2-R18','RGB'),
                 ('is','original_original','ISW-DETR (without state-contrast refinement)','RGB + weight'),
                 ('is','mean_half','ISW-DETR','RGB + weight')]
        body='\n'.join(name+' & '+inputs+' & '+mean_cell(public_grouped[variant,method])+r' \\' for variant,method,name,inputs in entries)
        specification='@{}'+CENTER+r'm{0.64\linewidth}*{2}{'+CENTER+'X}@{}'
        display=table_shell(r'Method & Input & AP (\%) \\',body,specification,6)
        display=highlight_maxima(display,'public','English')[0]
        display=(r'\begin{table}[htbp]'+'\n'+r'\centering\tablefont'+'\n'+
            r'\caption{Detection performance on the C-CCM cross-view task with configuration-level reference masses.}'+'\n'+
            r'\label{tab:cccm_public}'+'\n'+display+r'\end{table}'+'\n')
        (output/'cccm_table_main.tex').write_text(display,encoding='utf-8')
        tables.append('cccm_table_main.tex')
    if args.benchmark:
        benchmark = read(args.benchmark)
        lines = [r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule', r'Path & Parameters (M) & Median (ms) & IQR (ms) & P95 (ms) & GPU increment (MiB) \\', r'\midrule']
        for key, label in [('rt_detr_flat', 'RT-DETRv2'), ('identity_state_visual', 'I--S visual'), ('fixed_cvsr', 'CVSR reference'), ('full_bqr', 'BQR'), ('state_contrast_isw_detr', 'ISW-DETR')]:
            row = benchmark['results'][key]
            lo, hi = row['iqr_ms']
            lines.append(f"{label} & {row['parameters']/1e6:.6f} & {row['median_ms']:.2f} & {lo:.2f}--{hi:.2f} & {row['p95_ms']:.2f} & {row['increment_over_resident_mib']:.1f} " + r'\\')
        lines += [r'\bottomrule', r'\end{tabular}']
        (output / 'state_table_cost.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        tables.append('state_table_cost.tex')
    for name,kind in [('state_table_main.tex','main'),('state_table_ablation.tex','ablation'),('state_table_cost.tex','cost')]:
        if name not in tables:continue
        path=output/name
        display=base_transform(path.read_text(encoding='utf-8'),kind,'English')
        path.write_text(highlight_maxima(display,kind,'English')[0],encoding='utf-8')
    metrics=dict(private={split:{method:value for (s,method),value in grouped.items() if s==split} for split in ['validation','internal','holdout']},
        public=None if not args.public_results else {variant:{method:value for (v,method),value in public_grouped.items() if v==variant} for variant in ['base','flat','is']},
        seed_order=[42,3407,2026],display='means only; all displayed scalar maxima/ties bold; intervals not ranked')
    (output/'TABLE_METRICS.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    with (output / 'private_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'TABLE_MANIFEST.json').write_text(json.dumps(dict(tables=tables,
          private_training_seeds=[42, 3407, 2026], metric_units='AP points, AP50/AP75/AR100 percentages',
          manuscript='20261103_02',AP_display='three-seed means only',
          AP_uncertainty='sample SD and per-seed AP retained in TABLE_METRICS.json',
          pending=[] if args.public_results and args.benchmark else
              ([] if args.public_results else ['complete public evaluation']) + ([] if args.benchmark else ['fresh computational benchmark'])), indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(tables=tables, output=str(output))))


if __name__ == '__main__':
    main()
