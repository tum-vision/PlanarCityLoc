"""CPU regression tests; run without importing GPU models or downloading weights."""

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

from pose_solver.selection import (
    FEATURES, LEGACY_PROTOCOL, PROTOCOL, average_ranks,
    features_for_protocol, pick_rank_aggregation, rank_scores,
)

ROOT = Path(__file__).resolve().parents[1]


class SelectionTests(unittest.TestCase):
    def test_ties_average(self):
        np.testing.assert_array_equal(average_ranks([4, 2, 4, 0]), [2.5, 1, 2.5, 0])

    def test_aggregate_tie_keeps_first(self):
        self.assertEqual(pick_rank_aggregation([{"lbl": "PF"}, {"lbl": "A1"}]), "PF")

    def test_aggregate_tie_is_not_broken_by_sixth_weight_roundoff(self):
        rows = []
        for label, ranks in [('PF', [2, 1, 0, 2, 1, 0]),
                             ('A1', [1, 2, 2, 0, 0, 1]),
                             ('B1', [0, 0, 1, 1, 2, 2])]:
            rows.append(dict(lbl=label, **{key: direction * rank
                        for (key, direction), rank in zip(FEATURES, ranks)}))
        np.testing.assert_array_equal(rank_scores(rows), [1., 1., 1.])
        self.assertEqual(pick_rank_aggregation(rows), 'PF')

    def test_missing_and_nonfinite_tie_at_worst(self):
        rows = [{"lbl": "missing"}, {"lbl": "nan", "iou": float("nan")},
                {"lbl": "valid", "iou": 0.0}, {"lbl": "inf", "iou": float("inf")}]
        scores = rank_scores(rows)
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(scores[0], scores[3])
        self.assertEqual(pick_rank_aggregation(rows), "valid")

    def test_zero_error_is_not_missing(self):
        self.assertEqual(pick_rank_aggregation([
            {"lbl": "PF"}, {"lbl": "A1", "normal_err_deg": 0.0}]), "A1")

    def test_pnp_only_features_cannot_break_a_shared_feature_tie(self):
        rows = [{"lbl": "PF", "iou": .8},
                {"lbl": "A1", "iou": .8, "reproj": 0.0, "centroid_dist_px": 0.0}]
        before = rank_scores(rows)
        np.testing.assert_array_equal(before, [.5, .5])
        self.assertEqual(pick_rank_aggregation(rows), 'PF')
        rows[1].update(reproj=9999, centroid_dist_px=float('nan'))
        np.testing.assert_array_equal(rank_scores(rows), before)

    def test_missing_pnp_residuals_do_not_overrule_a_better_pf_pose(self):
        shared = dict(rec=.9, iou=.8, edge=.7, depth_ncc=.9, normal_err_deg=10.)
        rows = [dict(shared, lbl='PF', score=.9),
                dict(shared, lbl='A1', score=.8, reproj=0., centroid_dist_px=0.)]
        self.assertEqual(pick_rank_aggregation(rows), 'PF')
        # Pin the exact behavior used by the versioned v2 protocol.
        self.assertEqual(pick_rank_aggregation(rows, protocol=LEGACY_PROTOCOL), 'A1')
        np.testing.assert_array_equal(rank_scores(rows, protocol=LEGACY_PROTOCOL), [.4375, .5625])

    def test_pnp_with_better_shared_features_can_still_win(self):
        rows = [{'lbl': 'PF', 'score': .5, 'normal_err_deg': 20.},
                {'lbl': 'A1', 'score': .8, 'normal_err_deg': 10.}]
        self.assertEqual(pick_rank_aggregation(rows), 'A1')

    def test_unknown_protocol_is_rejected(self):
        with self.assertRaises(ValueError):
            rank_scores([{'lbl': 'PF'}], protocol='unrecognized-protocol')

    def test_manifest_records_six_feature_protocol(self):
        from utils.run_evidence import current_protocol
        spec = current_protocol()
        self.assertEqual(spec['protocol'], PROTOCOL)
        self.assertEqual(len(FEATURES), 6)
        self.assertEqual(len(features_for_protocol(LEGACY_PROTOCOL)), 8)
        self.assertEqual(spec['selection']['features'],
                         ['score', 'rec', 'iou', 'edge', 'depth_ncc', 'negative_normal_error'])
        np.testing.assert_allclose(spec['selection']['weights'], np.full(6, 1/6))

    def test_evaluation_labels_are_ignored(self):
        rows = [{"lbl": "PF", "iou": .8}, {"lbl": "A1", "iou": .5}]
        before = rank_scores(rows)
        rows[0].update(t_err=9999, r_err=9999, gt_pose="arbitrary")
        rows[1].update(t_err=0, r_err=0, gt_pose=None)
        np.testing.assert_array_equal(before, rank_scores(rows))

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            pick_rank_aggregation([])


class InferenceBoundaryTests(unittest.TestCase):
    def test_both_entrypoints_use_fixed_shared_feature_selector(self):
        rows = [{'lbl': 'PF', 'iou': .8},
                {'lbl': 'A1', 'iou': .8, 'reproj': 0., 'centroid_dist_px': 0.}]
        for filename in ('pipeline.py', 'pipeline_ablation.py'):
            tree = ast.parse((ROOT / filename).read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == '_pick_rank_aggregation')
            namespace = {}
            exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), namespace)
            self.assertEqual(namespace[function.name](rows), 'PF')

    def test_ablation_runner_does_not_call_removed_renderer_api(self):
        tree = ast.parse((ROOT / 'pipeline_ablation.py').read_text())
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)]
        self.assertNotIn('ignore_buildings', calls)

    def test_coarse_cue_ablation_keeps_candidate_scoring_fixed(self):
        from utils.pipeline_config import PipelineRunConfig
        from utils.ablation_config import candidate_scoring_config
        reference = PipelineRunConfig.load('uavd4l_inTraj')
        cfg = PipelineRunConfig.resolve('uavd4l', 'data/UAVD4L-LoD/inTraj',
            config_path='configs/pipeline/ablations/score_iou_only.yaml')
        self.assertEqual(cfg.cv_depth_weight, 0)
        selected = candidate_scoring_config(cfg, reference)
        self.assertEqual(selected.cv_depth_weight, reference.cv_depth_weight)
        self.assertEqual(selected.cv_edge_weight, reference.cv_edge_weight)
        self.assertEqual(cfg.cv_depth_weight, 0)
        self.assertIs(candidate_scoring_config(reference, reference), reference)

    def test_default_never_retains_cross_render_region_indices(self):
        from utils.pipeline_config import PipelineRunConfig
        for profile in ('uavd4l_inTraj', 'uavd4l_outTraj', 'swiss_epfl_inPlace', 'swiss_epfl_outPlace'):
            self.assertFalse(PipelineRunConfig.load(profile).pnp_retain_inliers)
        for filename in ('pipeline.py', 'pipeline_ablation.py'):
            tree = ast.parse((ROOT / filename).read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == 'run_iterative_semantic_pnp')
            guarded = [n for n in ast.walk(function) if isinstance(n, ast.If)
                       and 'cfg.pnp_retain_inliers' in ast.unparse(n.test)]
            self.assertEqual(len(guarded), 1)
            self.assertIn('_irls_allowed_pids', ast.unparse(guarded[0]))

    def test_gate_only_renders_the_estimated_pose(self):
        for filename in ("pipeline.py", "pipeline_ablation.py"):
            with self.subTest(filename=filename):
                tree = ast.parse((ROOT / filename).read_text())
                function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                and n.name == "run_iterative_semantic_pnp")
                names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
                self.assertNotIn("gt_pose", names)
                self.assertNotIn("n_gt_planes", names)
                self.assertNotIn("w2c_gt", names)
                namespace = {"np": np, "PipelineRunConfig": object,
                             "cv2": SimpleNamespace(setRNGSeed=lambda _: None)}
                exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
                estimate = np.eye(4)
                estimate[0, 3] = 137
                seen = []

                def render(pose, *args, **kwargs):
                    seen.append(pose.copy())
                    return (None, None, None, [{"plane_id": 0}])

                cfg = SimpleNamespace(pnp_skip_below_mask_frac=.08, low_mask_frac=.08,
                    gt_normal_cos=.995, gt_depth_rel=.02, min_pixels=50,
                    pnp_strict_pred_rendered_ratio=1.25, n_ab_iters=0,
                    pnp_strict_min_overlap_px=15, pnp_strict_min_vote_frac=.55,
                    pnp_strict_min_normal_cos=.85, pnp_strict_min_matches=10,
                    pnp_strict_reproj_px=12, pnp_strict_b_max_dist_px=70,
                    pnp_strict_b_lowe_ratio=.6)
                result = namespace[function.name](estimate, SimpleNamespace(get_plane_segmentation=render),
                    np.eye(3), 4, 4, np.zeros((4, 4)), [], cfg, n_pred_planes=3)
                self.assertEqual(len(seen), 1)
                np.testing.assert_array_equal(seen[0], estimate)
                self.assertTrue(result[4]["_gate"]["strict"])
                self.assertEqual(result[4]["_gate"]["reference"], "estimated_pose")


class MetricTests(unittest.TestCase):
    def test_audit_uses_recorded_protocol_not_the_new_default(self):
        from scripts.summarize_camera_ready import verify_candidate_selection
        rows = [{'lbl': 'Particle Filter', 'iou': .9, 'w2c': np.eye(4)},
                {'lbl': 'A-iter1', 'iou': .8, 'reproj': 0.,
                 'centroid_dist_px': 0., 'w2c': np.eye(4)}]
        for row, rank in zip(rows, rank_scores(rows, protocol=LEGACY_PROTOCOL)):
            row['aggregate_rank'] = float(rank)
        record = {'candidates': rows, 'selected': 'A-iter1', 'w2c_pf': np.eye(4)}
        verify_candidate_selection(record, protocol=LEGACY_PROTOCOL)
        with self.assertRaises(ValueError):
            verify_candidate_selection(record, protocol=PROTOCOL)

    def test_repeat_summary_verification_preserves_original_bytes(self):
        from scripts.summarize_camera_ready import save_or_verify_summary
        result = {'n_queries': 379, 'sha256': 'abc', 'metric': .8965078149175553}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'verified_summary.json')
            self.assertEqual(save_or_verify_summary(path, result), 'Wrote verified')
            before = path.read_bytes()
            self.assertEqual(save_or_verify_summary(path, result), 'Verified unchanged')
            # Allowed CPU roundoff must not replace the original numeric bytes.
            self.assertEqual(save_or_verify_summary(path, dict(result, metric=.8965078149175555)),
                             'Verified unchanged')
            self.assertEqual(before, path.read_bytes())

    def test_repeat_summary_verification_rejects_changes_without_overwriting(self):
        from scripts.summarize_camera_ready import save_or_verify_summary
        result = {'n_queries': 379, 'sha256': 'abc', 'metric': 1.0}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'verified_summary.json')
            save_or_verify_summary(path, result)
            before = path.read_bytes()
            for modified in (dict(result, n_queries=378), dict(result, sha256='changed'),
                             dict(result, metric=1.000001)):
                with self.assertRaises(ValueError):
                    save_or_verify_summary(path, modified)
                self.assertEqual(before, path.read_bytes())

    def test_saved_selector_audit_rejects_changed_selection_or_ranks(self):
        from scripts.summarize_camera_ready import verify_candidate_selection
        rows = [{'lbl': 'Particle Filter', 'iou': .1, 'w2c': np.eye(4)},
                {'lbl': 'A-iter1', 'iou': .9, 'w2c': np.eye(4)}]
        for row, rank in zip(rows, rank_scores(rows)):
            row['aggregate_rank'] = float(rank)
        record = {'candidates': rows, 'selected': 'A-iter1', 'w2c_pf': np.eye(4)}
        verify_candidate_selection(record)
        with self.assertRaises(ValueError):
            verify_candidate_selection(dict(record, selected='Particle Filter'))
        rows[1]['aggregate_rank'] += .1
        with self.assertRaises(AssertionError):
            verify_candidate_selection(record)

    def test_missing_buildings_are_removed_without_mutating_the_intact_map(self):
        from utils.lod import LoD
        from utils.map_ablation import without_buildings
        vertices = np.arange(27, dtype=float).reshape(9, 3)
        faces = np.arange(9).reshape(3, 3)
        lod = LoD.from_arrays(vertices, faces)
        lod._building_ids = np.array([0, 1, 2])
        variant = without_buildings(lod, [0])
        np.testing.assert_array_equal(variant.faces, faces[1:])
        np.testing.assert_array_equal(variant.get_building_ids(), [1, 2])
        np.testing.assert_array_equal(lod.faces, faces)
        self.assertIs(without_buildings(lod, []), lod)
        np.testing.assert_array_equal(without_buildings(lod, [2]).faces, faces[:2])

    def test_metric3d_centered_mean_padding(self):
        from PIL import Image
        from utils.normal_estimators import _metric3d_preprocess
        rgb = Image.fromarray(np.full((20, 20, 3), 100, dtype=np.uint8))
        tensor, (top, left, height, width) = _metric3d_preprocess(rgb, (40, 60))
        self.assertEqual((top, left, height, width), (0, 10, 40, 40))
        np.testing.assert_array_equal(tensor.numpy()[:, :, :, :left], 0)
        np.testing.assert_array_equal(tensor.numpy()[:, :, :, left+width:], 0)

    def test_joint_recall_requires_both_errors(self):
        from scripts.summarize_camera_ready import statistics
        result = statistics([(1, 3), (1, 1), (2, 0), (float('nan'), 0)])
        self.assertEqual(result['joint_recall_percent']['2'], 25.0)
        self.assertEqual(result['translation_only_recall_percent']['2'], 50.0)
        self.assertEqual(result['n_failed'], 1)

    def test_metric3d_known_focal_conversion_and_official_clamp(self):
        import torch
        from utils.normal_estimators import _metric3d_metric_depth
        # Original fx=2000 at 4000x3000, model scale=616/3000.
        K = np.array([[320., 0, 320], [0, 320, 240], [0, 0, 1]])
        depth = torch.tensor([-5., 10., 1000.])
        result = _metric3d_metric_depth(depth, K, (3000, 4000), (480, 640))
        np.testing.assert_allclose(result, [0, 10*2000*(616/3000)/1000, 300], rtol=1e-6)
        # Same physical camera represented at twice the target resolution.
        K2 = K.copy()
        K2[:2] *= 2
        np.testing.assert_array_equal(result,
            _metric3d_metric_depth(depth, K2, (3000, 4000), (960, 1280)))
        # Wide input exercises width-limited model resizing, without fx/fy averaging.
        K[1, 1] = 480
        wide = _metric3d_metric_depth(torch.tensor([10.]), K, (1000, 4000), (480, 640))
        np.testing.assert_allclose(wide, [10*2000*(1064/4000)/1000], rtol=1e-6)

    def test_metric3d_rejects_missing_or_invalid_calibration(self):
        import torch
        from utils.normal_estimators import _metric3d_metric_depth, Metric3Dv2Estimator
        estimator = object.__new__(Metric3Dv2Estimator)
        with self.assertRaises(ValueError):
            estimator.infer_depth_and_normal('not-read.jpg', (480, 640))
        for K in (None, np.eye(2), np.zeros((3, 3)), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                _metric3d_metric_depth(torch.ones(1), K, (3000, 4000), (480, 640))

    def test_camera_center_and_rotation(self):
        from scripts.summarize_camera_ready import pose_errors
        pose = np.eye(4)
        pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        pose[:3, 3] = [3, 4, 0]
        np.testing.assert_allclose(pose_errors(pose, np.eye(4)), [5, 90], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
