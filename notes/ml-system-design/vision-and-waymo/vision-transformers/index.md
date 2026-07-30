---
layout: note
title: "Vision Transformers (ViT)"
description: "Cut the image into 16×16 patches; linearly project each patch to a vector (\"patch embedding\"); add position embeddings; prepend a learnable [CLS] token; run a standard Transform…"
note: true
note_collection: "ML system design"
note_section: "Vision and Waymo"
section_order: 6
note_order: 8
updated: 2026-06-09 12:45:07 -0700
keywords:
  - Vision
  - Transformers
  - Embeddings
  - Training
  - LLMs
math: true
mermaid: false
---
## The idea
Cut the image into **16×16 patches**; linearly project each patch to a vector ("patch embedding"); add position embeddings; prepend a learnable [CLS] token; run a standard [Transformer encoder](/notes/ml-system-design/foundations/transformers-from-scratch/); classify from [CLS]. "An image is worth 16×16 words" — no convolutions at all.

For 224×224: (224/16)² = **196 tokens** — comfortably small for full attention.

## ViT vs CNN: the inductive-bias tradeoff
CNNs bake in locality + translation equivariance → data-efficient. ViT assumes almost nothing → must *learn* those regularities → **underperforms CNNs on small data, surpasses them with large-scale pretraining** (JFT-300M in the original paper). Practical fixes for less data: heavy augmentation + distillation from a CNN teacher (**DeiT**), or hybrid stems (conv early, attention late). Self-supervised pretraining (**MAE**: mask 75% of patches, reconstruct pixels; **DINO**: self-distillation) makes ViTs excellent without labels.

Why attention helps vision: **global receptive field at layer 1** (a CNN needs many layers before distant pixels interact) and content-dependent weighting.

## Swin Transformer — making ViT a dense-prediction backbone
Plain ViT keeps one resolution and global attention (O(T²) hurts at high res). **Swin**: compute attention within local **windows** (linear cost), **shift** the windows every other layer so information crosses window borders, and **merge patches** progressively to build a multi-scale pyramid (like a CNN's stages) → drop-in backbone for detection/segmentation with FPN-style necks.

## Where ViTs sit in your interviews
- [CLIP](/notes/ml-system-design/vision-and-waymo/clip-and-multimodal-models/)'s image encoder is a ViT — relevant to Roblox content moderation.
- Camera encoders in [BEV perception](/notes/ml-system-design/vision-and-waymo/bev-representations-and-sensor-fusion/) are CNN or ViT backbones; BEVFormer's deformable attention is a Transformer mechanism.
- Default answer for "image classifier in 2026": fine-tune a pretrained ViT (or use CLIP zero-/few-shot first).
