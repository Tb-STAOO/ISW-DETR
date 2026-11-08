"""Regenerate the five manuscript figures from freshly reproduced analyses."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reference_sources'))
sys.dont_write_bytecode = True
import figure_reference as plots


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-analysis', required=True)
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--bootstrap', required=True, help='Completed paired_bootstrap.py output directory')
    parser.add_argument('--protocol-root', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Use a fresh figure output directory')
    output.mkdir(parents=True, exist_ok=True)
    plots.FIG = output / 'figures'
    plots.FIG.mkdir()
    plots.DATA = output
    plots.PACKAGE = Path(args.protocol_root).resolve()
    plots.HOLDOUT_IMAGES = Path(args.dataset_root).resolve() / 'ScaleVision-MM-External-120/images'
    plots.CLASS_NAMES_PATH = ROOT / 'configs/scalevision/class_display_names.json'
    plots.BOOTSTRAP_SUMMARY = Path(args.bootstrap).resolve() / 'BOOTSTRAP_SUMMARY.json'
    plots.VALIDATION_ANALYSIS = Path(args.validation_analysis).resolve()
    plots.DIAGNOSTICS = Path(args.diagnostics).resolve()
    plots.SEEDS = (42, 3407, 2026)
    plots.MAIN = 'half_readout_mean_assignment'
    plots.SOURCE = {str(ROOT / 'reference_sources/figure_reference.py'): sha(ROOT / 'reference_sources/figure_reference.py')}
    def read(path):
        plots.SOURCE[str(path)] = sha(path)
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    def write(path, payload):
        Path(path).write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    plots.read, plots.write, plots.sha = read, write, sha
    plots.figure_system()
    plots.figure_acquisition()
    plots.figure_effects()
    plots.figure_stress()
    plots.figure_classes(read(plots.DIAGNOSTICS / 'PREDICTION_DIAGNOSTICS.json'))
    outputs, checks = {}, []
    for name in plots.CAPTIONS:
        for suffix in ['pdf', 'svg', 'png']:
            path = plots.FIG / (name + '.' + suffix)
            outputs[path.name] = sha(path)
        path = plots.FIG / (name + '.png')
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            checks.append(dict(figure=name, pixels=list(image.size), dpi=image.info.get('dpi'),
                               editable_SVG_text='<text' in (plots.FIG / (name + '.svg')).read_text(encoding='utf-8')))
    write(output / 'CAPTIONS.json', plots.CAPTIONS)
    write(output / 'FIGURE_MANIFEST.json', dict(figures=5, files=outputs, source_hashes=plots.SOURCE,
          package_versions={name: metadata.version(name) for name in ['matplotlib', 'numpy', 'Pillow']},
          source_function_provenance=str(ROOT / 'reference_sources/SOURCE_PROVENANCE.json'),
          automatic_checks=checks, independent_visual_inspection_still_required=True,
          note='Generated in a separate directory; manuscript PDFs and original images are not overwritten'))
    print(json.dumps(dict(figures=5, files=len(outputs), output=str(output))))


if __name__ == '__main__':
    main()
