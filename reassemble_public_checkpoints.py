"""Download or join the GitHub Release parts into the original checkpoint ZIP."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, build_opener


class GitHubReleaseOnly(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != 'https' or target.netloc not in {'github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com'}:
            raise ValueError('GitHub download returned an unexpected redirect.')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def copy_checked(src, dst, row):
    digest = sha256()
    size = 0
    for block in iter(lambda: src.read(2**20), b''):
        size += len(block)
        if size > row['bytes']:
            raise ValueError(f"Size mismatch for {row['name']}.")
        digest.update(block)
        if dst is not None:
            dst.write(block)
    if size != row['bytes'] or digest.hexdigest() != row['sha256']:
        raise ValueError(f"Checksum or size mismatch for {row['name']}.")


def fetch_part(opener, dest, row):
    url = 'https://github.com/Tb-STAOO/ISW-DETR/releases/download/sr-repro-20261108/' + quote(dest.name)
    temp = dest.with_suffix(dest.suffix + '.download')
    try:
        try:
            with opener.open(url, timeout=90) as response, temp.open('wb') as out:
                copy_checked(response, out, row)
        except HTTPError as exc:
            raise ValueError(f'{dest.name}: GitHub returned HTTP {exc.code}.') from None
        except (URLError, TimeoutError):
            raise ValueError(f'{dest.name}: connection interrupted. Run the command again to resume.') from None
        temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent,
                        help='Directory containing FILES_SHA256.json and the parts')
    parser.add_argument('--download', action='store_true', help='Fetch missing parts from the public GitHub Release')
    args = parser.parse_args()
    root = args.directory.resolve()
    spec = json.loads((root/'FILES_SHA256.json').read_text(encoding='utf-8'))['public_checkpoint_transport']
    archive = spec['archive']
    dest = root / archive['name']
    if dest.exists():
        with dest.open('rb') as stream:
            copy_checked(stream, None, archive)
        print(f'Already verified: {dest.name}')
        return
    if args.download:
        opener = build_opener(GitHubReleaseOnly())
        for index, row in enumerate(spec['parts'], 1):
            path = root / row['name']
            if path.exists():
                try:
                    with path.open('rb') as stream:
                        copy_checked(stream, None, row)
                    print(f'{index}/{len(spec["parts"])} already verified: {path.name}', flush=True)
                    continue
                except ValueError:
                    pass
            fetch_part(opener, path, row)
            print(f'{index}/{len(spec["parts"])} downloaded and verified: {path.name}', flush=True)
    pending = dest.with_suffix('.zip.assembling')
    try:
        with pending.open('wb') as out:
            for row in spec['parts']:
                path = root / row['name']
                if not path.is_file():
                    raise ValueError(f'Missing {path.name}. Place all parts in this directory or use --download.')
                with path.open('rb') as stream:
                    copy_checked(stream, out, row)
        with pending.open('rb') as stream:
            copy_checked(stream, None, archive)
        pending.replace(dest)
    finally:
        pending.unlink(missing_ok=True)
    print(f'Reassembled and verified: {dest.name}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
