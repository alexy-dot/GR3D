# Reconstruction and Object-Block Quality Gates

The project does not treat a plausible screenshot as proof of acceptable geometry. Every scene must retain raw evidence and pass staged checks before it enters an MLLM evaluation.

## Gate A: reconstruction integrity

- All exported points, camera poses, intrinsics, and depth values must be finite.
- Point count, confidence distribution, retained ratio, runtime, peak memory, model revision, checkpoint hash, and exact input frames must be recorded.
- Camera-gravity alignment must report median camera-up disagreement no greater than 10 degrees and 90th-percentile disagreement no greater than 20 degrees. Otherwise the scene is flagged for sensor-assisted orientation or exclusion.
- Dynamic-object duplication, regular ray/grid artifacts, and trajectory jumps must be inspected separately; point count alone is never a quality score.

## Gate B: semantic voxel consistency

- Weighted voxel semantic purity must be at least 0.75.
- The multiview-supported voxel ratio must be reported. A value below 0.25 raises a warning.
- More than 100 retained connected components raises a fragmentation warning for the current short-scene scale.
- Component size, semantic class, point count, voxel count, mean purity, maximum view support, and multiview ratio must be retained.
- Single-view voxels are not automatically removed. On the house pilot, requiring two views removed small but important table and chair components. They remain visible but are marked as lower-confidence evidence.

These are internal-consistency gates. They can detect unstable labels and weak view support but cannot prove geometric accuracy.

## Gate C: OSI-Bench sensor-grounded geometry

Before making outdoor quality claims, compare the RGB/Pi3 result with synchronized OSI-Bench sensor data where the released schema permits:

- camera trajectory: absolute and relative trajectory error against IMU/GPS-derived poses;
- orientation: gravity and heading error against IMU/GPS;
- scale: reconstructed-to-metric distance ratio and drift over sequence length;
- surface geometry: point-to-LiDAR distance, completeness, and precision at stated distance thresholds;
- object blocks: center error, extent error, fragmentation, and merging against available 3D annotations or LiDAR-supported reference clusters;
- temporal persistence: window-boundary alignment error and repeated-region consistency.

Numerical outdoor pass thresholds must be fixed after inspecting the official data schema and one declared pilot sequence, then held constant for the evaluation set. They must not be tuned per scene.

## Gate D: downstream usefulness

Geometry quality and MLLM usefulness are separate outcomes. Evaluate at least:

1. raw OSI-Bench frames;
2. raw frames plus aligned colored point-cloud views;
3. raw frames plus no-ID semantic voxel-block views;
4. an ID-linked GR3D-style control when feasible;
5. an optional LiDAR/oracle-map condition to estimate the upper bound.

Report OSI-Bench task categories separately. An overall score must not hide regressions in metric distance, direction, dynamics, or cross-view object correspondence.
