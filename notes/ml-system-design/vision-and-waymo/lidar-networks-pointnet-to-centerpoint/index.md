---
layout: note
title: "Lidar Networks — PointNet to CenterPoint"
description: "Lidar returns a point cloud: ~100–300k points per sweep, each (x, y, z, intensity). Two properties break standard nets: it's an unordered set (any permutation is the same scene…"
note: true
note_collection: "ML system design"
note_section: "Vision and Waymo"
section_order: 6
note_order: 4
math: true
mermaid: false
---
Lidar returns a **point cloud**: ~100–300k points per sweep, each (x, y, z, intensity). Two properties break standard nets: it's an **unordered set** (any permutation is the same scene → the network must be permutation-invariant) and it's **sparse/irregular** (no grid for convolutions).

## PointNet (2017) — learn on raw points
Apply a **shared MLP to every point independently**, then a **symmetric** aggregation (max-pool) over all points → global feature. Max-pool is permutation-invariant by construction; the MLP learns per-point features whose maxima summarize the shape (the points that survive the max are a learned skeleton). Limitation: no local structure (every point sees only itself + the global vector). **PointNet++** fixes this: sample centroids, group neighbors within a radius, run mini-PointNets per group, repeat hierarchically — "convolution-like locality for point sets."

## Voxelize-then-convolve: VoxelNet → PointPillars
- **VoxelNet:** divide space into 3D voxels; encode the points inside each voxel with a tiny PointNet (VFE layer); run 3D convolutions over the voxel grid; detect on the resulting map. Accurate but 3D convs are slow.
- **PointPillars (2019):** collapse the vertical axis — use **pillars** (voxels infinite in z). Per-pillar PointNet → scatter pillar features into a 2D **bird's-eye-view (BEV)** grid → **standard fast 2D CNN** + SSD-style head. Runs at 60+ Hz; the canonical speed/accuracy answer for onboard lidar detection. (Also note **sparse 3D convolutions** — compute only on occupied voxels — as the efficient middle path used by most modern stacks, e.g. SECOND.)

## CenterPoint (2021) — the modern default head
Anchor-free detection in BEV (the lidar twin of CenterNet, [Object Detection - RCNN to DETR](/notes/ml-system-design/vision-and-waymo/object-detection-rcnn-to-detr/)): predict a **heatmap of object centers** per class on the BEV map + regress size/height/orientation/velocity at each center; second stage refines with point features. Why centers beat anchor boxes for driving: cars have arbitrary heading — axis-aligned anchors fit rotated boxes poorly, while a center is rotation-neutral. Velocity regression from two frames gives cheap tracking ("tracking by detection": associate detections to tracks via predicted velocity + Hungarian matching, with a Kalman filter smoothing — say this when asked about tracking).

## BEV: why everyone converges on it
Bird's-eye-view is the natural plane for driving: objects don't overlap (unlike camera perspective), scale is metric and distance-invariant, and it's the coordinate frame in which lidar, camera, radar, the HD map, prediction, and planning can all **fuse and hand off**. Cameras get lifted into BEV too → [BEV Representations and Sensor Fusion](/notes/ml-system-design/vision-and-waymo/bev-representations-and-sensor-fusion/).

## Interview math nugget
Lidar sparsity at range: point density falls ~1/r² — a pedestrian at 60m may be <20 points → motivates camera fusion (dense texture), temporal aggregation (stack last N sweeps with ego-motion compensation, feeding point timestamps as features), and range-conditioned evaluation slices.
