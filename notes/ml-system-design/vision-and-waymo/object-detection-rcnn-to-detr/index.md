---
layout: note
title: "Object Detection — R-CNN to DETR"
description: "Task: localize (boxes) + classify every object. Know the two-stage/one-stage/set-prediction trichotomy and the shared vocabulary: IoU (intersection-over-union of boxes), NMS (no…"
note: true
note_collection: "ML system design"
note_section: "Vision and Waymo"
section_order: 6
note_order: 6
math: true
mermaid: false
---
**Task:** localize (boxes) + classify every object. Know the two-stage/one-stage/set-prediction trichotomy and the shared vocabulary: **IoU** (intersection-over-union of boxes), **NMS** (non-max suppression: greedily keep highest-score box, drop overlapping duplicates), **anchors** (predefined reference boxes at each location that the net regresses offsets from), **mAP** (mean average precision over classes at IoU thresholds).

## Two-stage: Faster R-CNN
1. Backbone (+FPN) → feature maps.
2. **Region Proposal Network (RPN):** a tiny conv head slides over features, scoring anchors as object/not + box offsets → ~1–2k proposals.
3. **RoI Align** crops each proposal's features to fixed size → per-proposal head classifies + refines the box.
Accurate, slower; the classic "precision first" choice.

## One-stage: YOLO / RetinaNet
Predict class+box **densely at every feature location** in one pass — fast, real-time. Problem: ~100k mostly-background predictions swamp training → **focal loss** (down-weight easy negatives; derived in [Loss Functions Field Guide](/notes/ml-system-design/foundations/loss-functions-field-guide/)) made one-stage match two-stage accuracy (RetinaNet). Modern anchor-free variants (FCOS, CenterNet) predict centers + sizes directly — **CenterNet's "objects as points" idea is reused by lidar's CenterPoint** ([Lidar Networks - PointNet to CenterPoint](/notes/ml-system-design/vision-and-waymo/lidar-networks-pointnet-to-centerpoint/)).

## Set prediction: DETR
Reframe detection as **direct set prediction**: a Transformer decoder holds N learned **object queries**; each query cross-attends into image features and outputs (class, box); training matches predictions to ground truth one-to-one via **Hungarian (bipartite) matching**, so duplicates are penalized by construction → **no anchors, no NMS** — the pipeline's hand-tuned parts vanish. Original DETR converged slowly and missed small objects; **Deformable DETR** fixed both by attending to a few learned sample points at multiple scales instead of all pixels (deformable attention — the same mechanism [BEVFormer](/notes/ml-system-design/vision-and-waymo/bev-representations-and-sensor-fusion/) uses). DETR-style heads are now everywhere, including 3D.

## Choosing
| Constraint | Pick |
|---|---|
| Real-time / edge | YOLO-family or anchor-free one-stage |
| Max accuracy, offline | DETR-family / Cascade R-CNN |
| Onboard AV (latency + multi-scale) | anchor-free BEV/center-based heads ([CenterPoint](/notes/ml-system-design/vision-and-waymo/lidar-networks-pointnet-to-centerpoint/)) |

Segmentation in one paragraph: **semantic** = per-pixel class (FCN/U-Net: encoder–decoder with skip connections — skips carry the spatial detail the bottleneck lost); **instance** = per-object masks (Mask R-CNN = Faster R-CNN + mask head); **SAM** = promptable foundation model for masks, great for auto-labeling pipelines (→ [auto-labeling](/notes/ml-system-design/vision-and-waymo/planning-imitation-learning-and-the-waymo-stack/)).
