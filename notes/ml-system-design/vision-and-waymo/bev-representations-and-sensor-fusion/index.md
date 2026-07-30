---
layout: note
title: "BEV Representations and Sensor Fusion"
description: "A camera image is a perspective projection — depth is lost. To place camera evidence in the bird's-eye-view plane you must reason about depth. Two canonical mechanisms:"
note: true
note_collection: "ML system design"
note_section: "Vision and Waymo"
section_order: 6
note_order: 1
math: true
mermaid: false
---
## The camera-to-BEV problem
A camera image is a perspective projection — depth is lost. To place camera evidence in the bird's-eye-view plane you must reason about depth. Two canonical mechanisms:

**1. Lift-Splat-Shoot (LSS, 2020) — push pixels out.** For each image pixel, predict a **categorical distribution over D discrete depths**; "lift" the pixel's feature into 3D at every depth bin weighted by that probability (a frustum of weighted features); "splat" all frustum points onto the BEV grid (sum-pool per cell). Fully differentiable; depth uncertainty is represented, not decided. Multi-camera by construction (all frustums splat into one grid).

**2. BEVFormer (2022) — pull features in.** Define a grid of learnable **BEV queries** (one per map cell). Each query projects its cell's 3D reference points into each camera image and **deformable-attends** ([Deformable DETR](/notes/ml-system-design/vision-and-waymo/object-detection-rcnn-to-detr/)'s mechanism: attend to a few learned sample locations, not all pixels) to gather features; temporal self-attention against the previous frame's BEV (ego-motion aligned) adds memory. Transformer-native, strong, heavier.

Mnemonic: LSS *pushes* image features outward along rays; BEVFormer's queries *pull* features inward from images.

## Fusion levels (the classic interview taxonomy)

| Level | What's fused | Pros / cons |
|---|---|---|
| **Early** | raw-ish data: paint lidar points with image semantics (PointPainting) | simple; tight coupling, sensitive to calibration/sync |
| **Mid / feature (BEV fusion)** | each modality → its own BEV feature map → concat/attend → shared heads | **the modern default**: modalities stay independent until a metric common frame |
| **Late** | per-sensor detections fused (e.g., probabilistic box merging) | robust, interpretable, easiest redundancy story; loses cross-modal cues |

Say also: **calibration & time sync** are the silent killers (extrinsics drift, rolling shutter, sweep-vs-frame alignment); design for **graceful degradation** (camera blinded by sun, lidar in fog/rain, radar's role as the weather-robust velocity sensor); and keep a late-fusion-style redundancy path for the safety case even if the primary stack is feature-fused.

## Occupancy grids (modern endpoint)
Predict dense 3D **occupancy** (which voxels are filled + semantics/flow) instead of only boxes → handles arbitrary-shape obstacles (debris, overhanging loads) that box detectors structurally miss. Good "how would you handle unknown objects?" answer.
