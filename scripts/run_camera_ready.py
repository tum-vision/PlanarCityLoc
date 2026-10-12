#!/usr/bin/env python3
"""Run the camera-ready evaluation snapshot with the repository's installed dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'reproducibility' / 'camera_ready'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(kind, release=RELEASE):
    """Check the archive and every member before exposing executable bytes."""
    release = Path(release)
    manifest = json.loads((release / 'manifest.json').read_text())
    spec = manifest['snapshots'][kind]
    filename = spec['archive']
    if Path(filename).name != filename:
        raise ValueError('Snapshot archive must be a local filename')
    payload = (release / filename).read_bytes()
    if digest(payload) != spec['archive_sha256']:
        raise ValueError(f'Snapshot archive changed: {kind}')
    from io import BytesIO
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(spec['source_sha256']):
            raise ValueError(f'Snapshot source inventory changed: {kind}')
        sources = {}
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if (path.is_absolute() or '..' in path.parts or str(path) != member.filename
                    or '\\' in member.filename or member.is_dir()
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Unsafe snapshot member')
            data = archive.read(member)
            if digest(data) != spec['source_sha256'][member.filename]:
                raise ValueError(f'Snapshot source changed: {member.filename}')
            sources[member.filename] = data
    entrypoint = {'main': 'pipeline.py', 'ablation': 'pipeline_ablation.py'}[kind]
    if spec['entrypoint'] != entrypoint or entrypoint not in sources:
        raise ValueError('Incorrect snapshot entrypoint')
    return spec, sources


def prepare_run(kind, arguments, *, root=ROOT, release=RELEASE):
    """Keep an exact, isolated source copy beside each new run's evidence."""
    root = Path(root).resolve()
    spec, sources = snapshot(kind, release)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config')
    options, remaining = parser.parse_known_args(arguments)
    output = Path(options.out).absolute()
    if output.exists():
        raise FileExistsError(f'Choose a new --out; refusing to overwrite {output}')
    configuration = None
    if options.config:
        candidate = Path(options.config)
        relative = candidate.as_posix()
        if candidate.is_absolute():
            try:
                relative = candidate.relative_to(root).as_posix()
            except ValueError:
                pass
        # Never fall back to a modified live-tree YAML for a paper experiment.
        if relative not in sources or not relative.startswith('configs/'):
            raise ValueError('Use a configuration included in the camera-ready snapshot')
        configuration = relative
    output.mkdir(parents=True, exist_ok=False)
    source = output / '_camera_ready_source'
    source.mkdir()
    for name, data in sources.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    # These dependencies are not vendored in the source snapshots. Dataset,
    # checkpoint and installed-package identities are recorded by inference.
    for name in ('data', 'checkpoints', 'thirdparty'):
        if (root / name).exists():
            (source / name).symlink_to(root / name, target_is_directory=True)
    args = [*remaining, '--out', str(output)]
    if configuration:
        args += ['--config', str(source / configuration)]
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(source)
    environment['PYTHONNOUSERSITE'] = '1'
    command = [sys.executable, '-u', str(source / spec['entrypoint']), *args]
    receipt = {'snapshot': kind, 'archive_sha256': spec['archive_sha256'],
               'source_sha256': spec['source_sha256'], 'command': command}
    (output / 'camera_ready_source.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return command, environment, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('verify', 'main', 'ablation'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == 'verify':
        if args.arguments:
            parser.error('verify takes no pipeline arguments')
        for kind in ('main', 'ablation'):
            _, sources = snapshot(kind)
            print(f'PASS: {kind}: {len(sources)} exact camera-ready source files')
        return 0
    arguments = args.arguments
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    command, environment, _ = prepare_run(args.mode, arguments)
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
