"""Build or import checksummed external resource archives; never upload files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

sys.dont_write_bytecode = True


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    part = PurePosixPath(name)
    if part.is_absolute() or '..' in part.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive member: ' + name)
    return part


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    build = sub.add_parser('build')
    build.add_argument('--catalog', required=True, help='JSON containing named groups with source/relative entries')
    build.add_argument('--group', required=True)
    build.add_argument('--output', required=True, help='New ZIP path outside source and code directories')
    extract = sub.add_parser('import')
    extract.add_argument('--archive', required=True)
    extract.add_argument('--sha256', required=True, help='Archive digest provided separately by the author')
    extract.add_argument('--output', required=True, help='Fresh destination directory')
    args = parser.parse_args()
    if args.mode == 'build':
        catalog = json.loads(Path(args.catalog).read_text(encoding='utf-8-sig'))
        rows = catalog['groups'][args.group]
        destination = Path(args.output).resolve()
        if destination.exists():
            parser.error('Refuse overwriting an existing archive')
        seen = set()
        entries = []
        for row in rows:
            relative = safe_name(row['relative']).as_posix()
            if relative == 'RESOURCE_MANIFEST.json' or relative in seen:
                raise ValueError('Duplicate or reserved resource name: ' + relative)
            seen.add(relative)
            source = Path(row['source']).resolve(strict=True)
            if not source.is_file():
                raise ValueError('Resource must be a file: ' + str(source))
            digest = sha(source)
            if row.get('sha256') and row['sha256'] != digest:
                raise ValueError('Source checksum mismatch: ' + str(source))
            entries.append(dict(source=source, relative=relative, bytes=source.stat().st_size, sha256=digest))
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest = dict(group=args.group, files=[{k: v for k, v in row.items() if k != 'source'} for row in entries])
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for row in entries:
                archive.write(row['source'], row['relative'])
                if sha(row['source']) != row['sha256']:
                    raise ValueError('Source changed while building archive')
            archive.writestr('RESOURCE_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        receipt = dict(group=args.group, file=destination.name, sha256=sha(destination),
                       bytes=destination.stat().st_size, members=len(entries))
        destination.with_suffix(destination.suffix + '.sha256.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(receipt))
    else:
        source = Path(args.archive).resolve(strict=True)
        if sha(source) != args.sha256.lower():
            parser.error('Archive checksum mismatch')
        destination = Path(args.output).resolve()
        if destination.exists() and any(destination.iterdir()):
            parser.error('Resource import requires a fresh destination')
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            names = [safe_name(i.filename).as_posix() for i in infos]
            if len(names) != len(set(names)) or names.count('RESOURCE_MANIFEST.json') != 1:
                raise ValueError('Duplicate member or missing manifest')
            for info in infos:
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Symlink members are forbidden')
            manifest = json.loads(archive.read('RESOURCE_MANIFEST.json'))
            rows = manifest['files']
            if set(names) != {row['relative'] for row in rows} | {'RESOURCE_MANIFEST.json'}:
                raise ValueError('Archive members differ from manifest')
            destination.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(destination).free < sum(row['bytes'] for row in rows) + 1024**2:
                raise ValueError('Insufficient destination disk space')
            for row in rows:
                target = destination.joinpath(*safe_name(row['relative']).parts).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError('Member outside destination')
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(row['relative']) as incoming, target.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
                    raise ValueError('Extracted resource checksum mismatch: ' + row['relative'])
            (destination / 'RESOURCE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(dict(passed=True, group=manifest['group'], files=len(rows), output=str(destination))))


if __name__ == '__main__':
    main()
