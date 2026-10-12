Legacy pipeline configs.

Default profiles are FAST (moge2_ext, small CV/PF). For paper-reproduction quality:
  configs/pipeline/uavd4l_inTraj_slow.yaml
  configs/pipeline/uavd4l_outTraj_slow.yaml

Stage-2 cost volume is 4D only (x, y, z, yaw). Pitch/roll are refined by the PF.

6-DoF cost volume (pitch/roll in stage-1 LARGE grid only):
  Use pipeline.py flags for prior-noise / rotation robustness ablations:
    --perturb_gt_r <deg>  --all_6D
  See slurm/run_ablation_compact.sh and run_ablation_robustness_all.sh.

Accidental stage-2 6-DoF (Jun 2025 yaml bug — do NOT use for new runs):
  legacy/uavd4l_inTraj_stage2_6dof.yaml  (~19k stage-2 hyps)
  legacy/uavd4l_outTraj_stage2_6dof.yaml (~16k stage-2 hyps)
  Snapshots: legacy/snapshots/uavd4l_*_20250628.yaml
