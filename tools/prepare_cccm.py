"""Portable CLI for the original C-CCM conversion and verified image retrieval."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-root', required=True, help='Directory containing the three fixed source metadata files')
    parser.add_argument('--data-root', required=True, help='Fresh destination for prepared annotations/images')
    parser.add_argument('--work-dir', required=True, help='Fresh destination for conversion and transport records')
    parser.add_argument('--image-source', help='Optional existing flat PNG directory, verified by member CRC/size before copying')
    parser.add_argument('--metadata-only', action='store_true', help='Convert all annotations, observations and training priors without downloading images')
    args = parser.parse_args()
    metadata = Path(args.metadata_root).resolve()
    data = Path(args.data_root).resolve()
    work = Path(args.work_dir).resolve()
    if data == metadata or data.is_relative_to(metadata) or metadata.is_relative_to(data):
        parser.error('Keep source metadata and prepared output in separate directories')
    lock = read(ROOT / 'configs/cccm/source_metadata_manifest.json')
    for relative, entry in lock['files'].items():
        path = metadata / relative
        if not path.is_file() or sha(path) != entry['sha256']:
            parser.error('Missing or changed fixed source metadata: ' + str(path))
    if args.metadata_only and (data.exists() and any(data.iterdir())):
        parser.error('Metadata conversion requires a fresh output directory')
    if work == metadata or work.is_relative_to(metadata):
        parser.error('Work directory must not modify source metadata')
    source = ROOT / 'data_preparation/cccm_source.py'
    spec = importlib.util.spec_from_file_location('original_cccm_conversion', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE, module.DATA, module.OUT = metadata, data, work
    module.OLD = Path(args.image_source).resolve() if args.image_source else work / 'no_external_image_cache'
    candidates, table, categories = module.setup()
    prepared_lock = read(ROOT / 'configs/cccm/EXPERIMENT_LOCK.json')
    checks = {name: {'expected': expected, 'actual': sha(data / name)}
              for name, expected in prepared_lock['input_hashes'].items()}
    if any(entry['actual'] != entry['expected'] for entry in checks.values()):
        raise RuntimeError('Converted metadata differ from the fixed research inputs: ' + json.dumps(checks))
    qa = dict(passed=True, images_expected=len(candidates), classes=len(categories),
              fixed_input_files=len(checks), converted_hashes=checks,
              original_conversion_sha256=sha(source), image_retrieval_done=False)
    module.write(work / 'PORTABLE_CONVERSION_QA.json', qa)
    if not args.metadata_only:
        verified = module.download(candidates)
        duplicates = module.duplicate_diagnostics(candidates, verified)
        module.preview(candidates)
        module.finalize(candidates, verified, duplicates)
        qa['image_retrieval_done'] = True
        qa['verified_images'] = len(verified)
        module.write(work / 'PORTABLE_CONVERSION_QA.json', qa)
    print(json.dumps({'passed': True, 'images_expected': len(candidates),
                      'classes': len(categories), 'fixed_input_files': len(checks),
                      'image_retrieval_done': qa['image_retrieval_done'],
                      'report': str(work / 'PORTABLE_CONVERSION_QA.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
