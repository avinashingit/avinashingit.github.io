---
layout: note
title: "CNNs and ResNet"
description: "A convolutional layer slides a small learned filter (e.g., 3×3×C) across the image, computing a dot product at each location → a feature map. Three built-in assumptions (inducti…"
note: true
note_collection: "ML system design"
note_section: "Vision and Waymo"
section_order: 6
note_order: 3
updated: 2026-06-09 12:45:07 -0700
keywords:
  - Vision
  - Training
  - Transformers
  - LLMs
  - Graphs
math: true
mermaid: false
---
## Convolutions from first principles

A convolutional layer slides a small learned filter (e.g., 3×3×C) across the image, computing a dot product at each location → a feature map. Three built-in assumptions (inductive biases) that make CNNs data-efficient on images:
1. **Locality:** nearby pixels relate (edges, textures are local).
2. **Weight sharing / translation equivariance:** a cat detector works anywhere in the frame; one filter, all positions → drastically fewer params than dense layers.
3. **Hierarchy:** stack layers → receptive field grows → edges → textures → parts → objects.

Vocabulary: stride (downsampling), padding, pooling (max/avg downsampling), channels, receptive field, 1×1 conv (per-pixel channel mixing / cheap dimensionality change), depthwise-separable conv (factorize spatial × channel mixing — MobileNet's efficiency trick, relevant for [onboard compute](/notes/ml-system-design/vision-and-waymo/planning-imitation-learning-and-the-waymo-stack/)).

## The degradation problem and ResNet (2015)

Pre-2015: deeper networks *trained worse* — not just overfitting; **training** error rose with depth (a 56-layer net underperformed a 20-layer one). In principle the deeper net could copy the shallow one and set extra layers to identity, but plain stacks couldn't *learn* identity easily, and gradients vanished through many layers.

**Residual learning:** make each block compute a *correction* to its input:

$$y = F(x) + x$$

- If the best thing a block can do is nothing, it just learns $F \to 0$ — trivial.
- Gradients flow through the identity path **undiminished**: $\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + I$ — the "+I" is the gradient highway that lets 100+ layer networks train.

This idea — **residual/skip connections** — is arguably the most reused trick in deep learning: [Transformer blocks](/notes/ml-system-design/foundations/transformers-from-scratch/), U-Net skips, [GraphSAGE's self-concat](/notes/ml-system-design/graph-neural-networks/gcn-graphsage-gat/), DCN's "+x" — all the same medicine.

Plus **BatchNorm** (normalize activations per channel over the batch → stable scales, higher learning rates; at inference uses running statistics — mention train/test discrepancy and its small-batch pain, which is why Transformers use LayerNorm).

## Backbones to name
- **ResNet-50:** the eternal baseline; "backbone" = the pretrained feature extractor reused by detection/segmentation heads.
- **EfficientNet:** compound-scale depth/width/resolution together.
- **ConvNeXt:** ResNet modernized with ViT-era training recipes — evidence that training recipes, not just architecture, drove much of the ViT gains.
- **Feature Pyramid Network (FPN):** combine high-res shallow features with semantic deep features via a top-down pathway — the standard neck for detecting objects at multiple scales (→ [Object Detection - RCNN to DETR](/notes/ml-system-design/vision-and-waymo/object-detection-rcnn-to-detr/)).

## Transfer learning (the default workflow)
Pretrain on ImageNet (or web-scale, or [CLIP](/notes/ml-system-design/vision-and-waymo/clip-and-multimodal-models/)) → fine-tune on your task with a new head. For moderation/perception tasks with limited labels this is assumed; from-scratch training is the exception.
