"""CPU-only checks for direct evaluation commands and recorded provenance."""
import ast
from dataclasses import make_dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pose_solver.selection import FEATURES, PROTOCOL
from utils.run_evidence import current_protocol, write_manifest

ROOT = Path(__file__).resolve().parents[1]


class EvaluationEntrypointTests(unittest.TestCase):
    def test_direct_entrypoints_record_configuration_and_use_common_selector(self):
        self.assertEqual(len(FEATURES), 8)
        self.assertEqual(current_protocol()['protocol'], PROTOCOL)
        for name in ('pipeline.py', 'pipeline_ablation.py'):
            source = (ROOT / name).read_text()
            self.assertIn('write_manifest(', source)
            self.assertIn('from pose_solver.selection import pick_rank_aggregation', source)
            self.assertIn('_initialize_result_files(', source)

    def test_existing_results_are_never_overwritten(self):
        tree = ast.parse((ROOT / 'pipeline.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_initialize_result_files')
        namespace = {'Path': Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'pipeline.py', 'exec'), namespace)
        initialize = namespace[function.name]
        for name in ('metrics.jsonl', 'est_poses.txt', 'run_manifest.json', 'candidates.jsonl'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / name
                target.write_text('existing result')
                with self.assertRaises(FileExistsError):
                    initialize(folder)
                self.assertEqual(target.read_text(), 'existing result')

    def test_manifest_records_current_sources_and_evaluation_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with patch('subprocess.check_output', return_value='CPU test'), \
                    patch('importlib.metadata.distributions', return_value=[]):
                write_manifest(out, SimpleNamespace(data_root=str(out / 'no_data')),
                               make_dataclass('Config', [])(), SimpleNamespace(frames=['fixture']),
                               out / 'no_mesh')
            manifest = json.loads((out / 'run_manifest.json').read_text())
            self.assertEqual(manifest['protocol']['protocol'], PROTOCOL)
            self.assertEqual(manifest['protocol']['selection']['weights'], [.125] * 8)
            for name in ('pipeline.py', 'pipeline_ablation.py', 'pose_solver/selection.py',
                         'configs/evaluation.json', 'configs/ablations.json'):
                self.assertIn(name, manifest['code_sha256'])
            before = (out / 'run_manifest.json').read_bytes()
            with patch('subprocess.check_output', return_value='CPU test'), \
                    patch('importlib.metadata.distributions', return_value=[]):
                with self.assertRaises(FileExistsError):
                    write_manifest(out, SimpleNamespace(data_root=str(out / 'no_data')),
                                   make_dataclass('Config', [])(), SimpleNamespace(frames=['fixture']),
                                   out / 'no_mesh')
            self.assertEqual((out / 'run_manifest.json').read_bytes(), before)

    def test_documented_commands_use_repository_entrypoints(self):
        for name in ('README.md', 'docs/REPRODUCIBILITY.md', 'docs/ABLATIONS.md'):
            text = (ROOT / name).read_text()
            self.assertNotIn('run_camera_ready', text)
            self.assertNotIn('summarize_camera_ready', text)
            self.assertIn('python pipeline', text)
        self.assertFalse((ROOT / 'reproducibility').exists())


if __name__ == '__main__':
    unittest.main()
