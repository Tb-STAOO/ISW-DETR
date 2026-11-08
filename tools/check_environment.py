"""Check dependencies and basic CPU/CUDA operators; not a reproduction test."""
import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    'torch': 'torch', 'torchvision': 'torchvision', 'numpy': 'numpy',
    'scipy': 'scipy', 'Pillow': 'PIL', 'PyYAML': 'yaml',
    'faster-coco-eval': 'faster_coco_eval', 'pycocotools': 'pycocotools',
    'tensorboard': 'tensorboard', 'timm': 'timm', 'matplotlib': 'matplotlib',
    'requests': 'requests',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output', help='Optional environment report JSON')
    args = parser.parse_args()
    report = dict(passed=False, scope='Dependency/import/operator preflight only',
                  python=sys.version, executable=sys.executable,
                  platform=platform.platform(), packages={}, checks={},
                  warnings=[], errors=[])
    if sys.version_info < (3, 11):
        report['errors'].append('Python >=3.11 is required by hashlib.file_digest; reference environment uses 3.12.')
    reference = {}
    for line in (ROOT / 'requirements-tested.txt').read_text(encoding='utf-8-sig').splitlines():
        if '==' in line:
            key, value = line.strip().split('==', 1)
            reference[key] = value
    for distribution, module in MODULES.items():
        try:
            importlib.import_module(module)
            version = metadata.version(distribution)
            report['packages'][distribution] = version
            expected = reference.get(distribution)
            if version != expected:
                report['warnings'].append(f'{distribution}: installed {version}; reference {expected}. Compatibility is not certified by this check.')
        except Exception as error:
            report['errors'].append(f'{distribution}: {type(error).__name__}: {error}')
    if 'torch' in report['packages'] and 'torchvision' in report['packages']:
        try:
            import torch
            from torchvision.ops import nms
            torch.set_num_threads(2)
            available = torch.cuda.is_available()
            report['cuda'] = dict(available=available, build=torch.version.cuda,
                                  cudnn=torch.backends.cudnn.version())
            if available:
                report['cuda'].update(gpu=torch.cuda.get_device_name(0),
                                     capability=list(torch.cuda.get_device_capability(0)),
                                     total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
            if args.device == 'cuda' and not available:
                raise RuntimeError('CUDA was requested but is not available. Check the torch build, GPU and driver.')
            devices = ['cpu']
            if available and args.device in ('auto', 'cuda'):
                devices.append('cuda')
            for device in devices:
                with torch.inference_mode():
                    x = torch.ones(1, 1, 4, 4, device=device)
                    weight = torch.ones(1, 1, 3, 3, device=device)
                    conv = torch.nn.functional.conv2d(x, weight)
                    grid = torch.zeros(1, 2, 2, 2, device=device)
                    sampled = torch.nn.functional.grid_sample(x, grid, align_corners=False)
                    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device=device)
                    kept = nms(boxes, torch.tensor([.9, .8], device=device), .5)
                    if not (bool((conv == 9).all()) and bool(torch.isfinite(sampled).all()) and kept.tolist() == [0]):
                        raise RuntimeError('Unexpected basic operator result on ' + device)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                report['checks'][device + '_operators'] = True
        except Exception as error:
            report['errors'].append(f'Operators: {type(error).__name__}: {error}')
    # Each model tree has its own module registry; import them in separate processes.
    for name, runtime in [('public', ROOT / 'runtime'), ('private', ROOT / 'inference/runtime')]:
        command = [sys.executable, '-I', '-B', '-c',
                   'import sys; sys.path.insert(0, sys.argv[1]); import src; import src.solver; print("IMPORT_OK")',
                   str(runtime)]
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=90)
            report['checks'][name + '_runtime_import'] = result.returncode == 0
            if result.returncode:
                report['errors'].append(name + ' runtime import: ' + result.stderr[-2000:])
        except Exception as error:
            report['errors'].append(f'{name} runtime import: {type(error).__name__}: {error}')
    report['passed'] = not report['errors']
    report['warnings'].append('Passing does not validate dataset availability, checkpoint provenance, full training, complete AP, another OS, or another computer.')
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        destination = Path(args.output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + '\n', encoding='utf-8')
    print(text)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
