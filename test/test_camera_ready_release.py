"""CPU-only checks for isolated camera-ready execution and source integrity."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from scripts.run_camera_ready import ROOT, RELEASE, snapshot, prepare_run


class CameraReadyReleaseTests(unittest.TestCase):
    def test_source_inventories_and_eight_feature_protocol(self):
        for kind, count in [('main', 61), ('ablation', 65)]:
            spec, sources = snapshot(kind)
            self.assertEqual(len(sources), count)
            self.assertNotIn('configs/evaluation_camera_ready_v3.json', sources)
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)/'run'
                command, environment, source = prepare_run(kind, ['--out', str(out)])
                probe = (
                    'import json; from pathlib import Path; import pose_solver.selection as s; '
                    'print(json.dumps([str(Path(s.__file__).resolve()), len(s.FEATURES), '
                    's.PROTOCOL]))'
                )
                result = json.loads(subprocess.check_output(
                    [sys.executable, '-c', probe], cwd=source, env=environment, text=True))
                self.assertEqual(result[0], str(source/'pose_solver/selection.py'))
                self.assertEqual(result[1], 8)
                self.assertIn('camera-ready-v2-', result[2])
                self.assertEqual(command[2], str(source/spec['entrypoint']))
                self.assertEqual(environment['PYTHONPATH'], str(source))
                for name, digest in spec['source_sha256'].items():
                    self.assertEqual(hashlib.sha256((source/name).read_bytes()).hexdigest(), digest)
                probe = source/'manifest_probe.py'
                probe.write_text(
                    'from dataclasses import make_dataclass\n'
                    'from pathlib import Path\n'
                    'from types import SimpleNamespace\n'
                    'from unittest.mock import patch\n'
                    'from utils.run_evidence import write_manifest\n'
                    'out = Path(__file__).parent.parent\n'
                    'with patch("subprocess.check_output", return_value="CPU test"), '
                    'patch("importlib.metadata.distributions", return_value=[]):\n'
                    '    write_manifest(out, SimpleNamespace(data_root=str(out / "no_data")), '
                    'make_dataclass("Config", [])(), SimpleNamespace(frames=["fixture"]), '
                    'out / "no_mesh")\n')
                subprocess.run([sys.executable, str(probe)], cwd=ROOT, env=environment, check=True)
                manifest = json.loads((out/'run_manifest.json').read_text())
                self.assertEqual(manifest['code_sha256'], spec['source_sha256'])
                self.assertEqual(manifest['protocol']['protocol'], result[2])

    def test_launch_imports_frozen_modules_even_from_live_repo(self):
        with tempfile.TemporaryDirectory() as folder:
            _, env, source = prepare_run('main', ['--out', str(Path(folder)/'run')])
            # A script's directory, not the working directory, is sys.path[0].
            probe = source/'probe.py'
            probe.write_text('import pose_solver.selection as s\nprint(len(s.FEATURES))\n')
            result = subprocess.check_output([sys.executable, str(probe)], cwd=ROOT, env=env, text=True)
            self.assertEqual(result.strip(), '8')

    def test_no_overwrite_and_no_live_config_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'run'
            with self.assertRaises(ValueError):
                prepare_run('main', ['--out', str(output), '--config', '/outside/new.yaml'])
            self.assertFalse(output.exists())
            cmd, _, source = prepare_run('ablation', ['--out', str(output), '--config',
                                         'configs/pipeline/ablations/moge2_vitl.yaml', '--n', '100'])
            self.assertIn(str(source/'configs/pipeline/ablations/moge2_vitl.yaml'), cmd)
            self.assertIn('--n', cmd)
            with self.assertRaises(FileExistsError):
                prepare_run('main', ['--out', str(output)])

    def test_tampered_archive_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            release = Path(folder)
            (release/'manifest.json').write_bytes((RELEASE/'manifest.json').read_bytes())
            (release/'main.zip').write_bytes((RELEASE/'main.zip').read_bytes()+b'tampered')
            with self.assertRaisesRegex(ValueError, 'archive changed'):
                snapshot('main', release)

    def test_wrong_entrypoint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            release = Path(folder)
            manifest = json.loads((RELEASE/'manifest.json').read_text())
            manifest['snapshots']['main']['entrypoint'] = 'pipeline_ablation.py'
            (release/'manifest.json').write_text(json.dumps(manifest))
            (release/'main.zip').write_bytes((RELEASE/'main.zip').read_bytes())
            with self.assertRaisesRegex(ValueError, 'Incorrect snapshot entrypoint'):
                snapshot('main', release)

    def test_member_hash_inventory_and_traversal_rejected(self):
        for names, declared, error in [
            (['pipeline.py'], {'pipeline.py': 'wrong'}, 'source changed'),
            (['pipeline.py'], {}, 'inventory changed'),
            (['../pipeline.py'], {'../pipeline.py': hashlib.sha256(b'code').hexdigest()}, 'Unsafe'),
        ]:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as folder:
                release = Path(folder)
                with zipfile.ZipFile(release/'main.zip', 'w') as z:
                    for name in names:
                        z.writestr(name, b'code')
                manifest = {'snapshots': {'main': {'archive': 'main.zip',
                    'archive_sha256': hashlib.sha256((release/'main.zip').read_bytes()).hexdigest(),
                    'source_sha256': declared, 'entrypoint': 'pipeline.py'}}}
                (release/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, error):
                    snapshot('main', release)


if __name__ == '__main__':
    unittest.main()
