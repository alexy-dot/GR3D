# GR3D-style semantic voxel blocks: house

Date: 2026-09-20

## Purpose

Validate the missing GR3D object-analysis path on the original-Pi3 static house reconstruction:

```text
ADE20K semantic labels -> point/pixel association -> voxel majority vote
-> same-label six-neighbor components -> no-ID component renders
```

## Semantic segmentation

- Model: `facebook/mask2former-swin-small-ade-semantic`
- Local checkpoint SHA-256: `2533310a1e902580a79e575814f1fa04078a99e2321b9bda7afd4f22bba49189`
- Frames: 8
- Label-map shape: `(8, 238, 420)`
- Task: ADE20K semantic segmentation
- Input geometry observations: 498,273

The model was loaded entirely from a local checkpoint. Transformers 4.57.6 and SciPy 1.15.3 were installed into the WSL Pi3 environment using Windows-downloaded Linux wheels because direct WSL downloads were unreliable.

## Voxel configuration A

- Voxel size: 0.03 Pi3 model units
- Minimum points per voxel: 2
- Minimum voxels per component: 8
- Retained voxels: 13,123
- Retained connected components: 123

This condition is visibly fragmented and is retained as the initial unmodified connectivity result.

## Voxel configuration B

Only the minimum component size changed from 8 to 30 voxels:

- Retained voxels: 12,076
- Retained connected components: 48
- Large retained semantic components include floor, wall, building, plant, grass, ceiling, table, chair, door, and windowpane.

The larger threshold removes many fragments while preserving recognizable object categories such as tables and chairs. It is an initial presentation setting, not a universally valid threshold.

Quality diagnostics for this condition:

- weighted semantic purity: 0.9278;
- multiview-supported voxel ratio: 0.3750;
- median view support: 1;
- internal quality status: pass.

## Multiview-only diagnostic

Only `min_voxel_views` was changed from 1 to 2 while the 30-voxel component threshold was held fixed:

- retained voxels: 3,175;
- retained components: 17;
- weighted semantic purity: 0.9005;
- multiview-supported voxel ratio: 1.0;
- median view support: 2.

This stricter condition removed all retained table and chair components from the largest-component summary. Therefore multiview support should be reported as confidence evidence, but requiring two views globally is too destructive for sparse or occluded objects in this pilot.

## Bounding-box presentation diagnostic

A combined voxel-plus-bounding-box view was rendered after removing components with fewer than 1,000 contributing points. It retained 11,452 voxels from 36 components. The view was more explicit geometrically but visually cluttered because structural classes such as floor, wall, building, and grass create very large overlapping boxes.

Decision: do not use a single all-class box overlay as the default MLLM input. Produce separate layout/stuff and object/thing layers, while retaining the full voxel view and machine-readable boxes for audit.

## Interpretation

- The GR3D-style 2D-semantic-to-3D voxel path is operational end to end.
- Component coloring can produce no-ID orthographic object-block views without modifying source images.
- Semantic connectivity is not reliable instance segmentation: adjacent same-class objects may merge, while noisy geometry may split one object into several components.
- Original Pi3 scale is not verified metric, so voxel size 0.03 must not be transferred directly to OSI-Bench scenes. Outdoor experiments must select voxel size in a metric or explicitly calibrated coordinate system.
