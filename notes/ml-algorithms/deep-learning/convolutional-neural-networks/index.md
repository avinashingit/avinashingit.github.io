---
layout: note
title: "Convolutional Neural Networks"
description: "Consider a 224×224 RGB image fed into an MLP with 1000 hidden units:"
note: true
note_collection: "ML algorithms"
note_section: "Deep Learning"
section_order: 4
note_order: 2
updated: 2026-06-06 11:30:50 -0700
keywords:
  - Deep Learning
  - Training
  - Transformers
  - Clustering
  - Evaluation
math: true
mermaid: false
---
> CNNs are neural networks specialized for **grid-structured data** (images, audio spectrograms, video). They exploit three key ideas: **local connectivity**, **parameter sharing**, and **translation equivariance**.

---

## Table of Contents

- [1. Why CNNs Instead of MLPs for Images?](#1-why-cnns-instead-of-mlps-for-images)
- [2. The Convolution Operation](#2-the-convolution-operation)
- [3. Hyperparameters of a Conv Layer](#3-hyperparameters-of-a-conv-layer)
- [4. Pooling](#4-pooling)
- [5. Receptive Field](#5-receptive-field)
- [6. Key Architectural Patterns](#6-key-architectural-patterns)
- [7. Famous Architectures](#7-famous-architectures)
- [8. Translation Equivariance vs Invariance](#8-translation-equivariance-vs-invariance)
- [9. Backprop Through Convolution](#9-backprop-through-convolution)
- [10. Strengths and Weaknesses](#10-strengths-and-weaknesses)
- [11. When to Use vs Not Use](#11-when-to-use-vs-not-use)
- [12. CNN vs MLP vs Vision Transformer](#12-cnn-vs-mlp-vs-vision-transformer)
- [13. Common Pitfalls](#13-common-pitfalls)
- [14. Production Considerations](#14-production-considerations)

---

## 1. Why CNNs Instead of MLPs for Images?

Consider a 224×224 RGB image fed into an MLP with 1000 hidden units:

- Input dim: $224 \times 224 \times 3 = 150{,}528$
- First layer weights: $150{,}528 \times 1000 \approx 150$M params

**Problems with MLPs on images:**

- **Massive parameter count** → overfitting, memory issues
- **No spatial structure** — pixels at $(0,0)$ and $(0,1)$ are no more related than $(0,0)$ and $(100,100)$
- **Not translation invariant** — a cat in the top-left vs bottom-right looks like a totally different input

CNNs solve all three: filters are small (local), reused everywhere (shared), and detect the same pattern regardless of location (translation equivariant).

---

## 2. The Convolution Operation

A **convolutional filter (kernel)** is a small matrix that slides over the input, computing dot products at each position.

For a 2D input $\mathbf{X}$ and filter $\mathbf{K}$ of size $k \times k$:

$$Y_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X_{i+m, j+n} \cdot K_{m,n} + b$$

This produces a **feature map** — each location measures how strongly the filter pattern is present at that spot.

> **Note:** Cross-correlation vs convolution Deep learning uses **cross-correlation** but calls it "convolution." True mathematical convolution flips the kernel. Doesn't matter in practice — the kernel is learned anyway.

### Multi-channel input

For RGB input with 3 channels, the filter is also 3D ($k \times k \times C_{\text{in}}$). The filter sums across all input channels:

$$Y_{i,j} = \sum_{c=0}^{C_{\text{in}}-1} \sum_{m,n} X_{i+m, j+n, c} \cdot K_{m,n,c} + b$$

Each filter produces **one** output channel. To get $C_{\text{out}}$ output channels, you use $C_{\text{out}}$ separate filters.

### Parameters per conv layer

$$\text{params} = k \times k \times C_{\text{in}} \times C_{\text{out}} + C_{\text{out}}$$

**Example:** $3 \times 3$ kernel, 64 input channels, 128 output channels: $3 \times 3 \times 64 \times 128 + 128 = 73{,}856$ params — _regardless of image size_.

---

## 3. Hyperparameters of a Conv Layer

|Hyperparameter|Meaning|
|---|---|
|**Kernel size** $k$|Receptive field of one filter (typically 3, 5, 7)|
|**Stride** $s$|Step size when sliding. Larger stride → smaller output|
|**Padding** $p$|Pixels added around input. "Same" padding preserves spatial size|
|**Dilation** $d$|Spacing between kernel elements (atrous convolution)|
|**Channels** $C_{\text{out}}$|Number of filters = output depth|

### Output size formula

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - d(k-1) - 1}{s} \right\rfloor + 1$$

For standard convolution ($d=1$): $H_{\text{out}} = \lfloor (H_{\text{in}} + 2p - k)/s \rfloor + 1$.

- **"Same" padding:** $p = (k-1)/2$ (with $s=1$) → output size = input size
- **"Valid" padding:** $p = 0$ → output shrinks

---

## 4. Pooling

Pooling **downsamples** feature maps:

- **Max pooling:** take max over a window (typically $2\times 2$, stride 2)
- **Average pooling:** take mean
- **Global average pooling:** average over the entire spatial dim → one value per channel (replaces FC layers in modern architectures)

**Purpose:**

- Reduce spatial size → less computation
- Provide some translation invariance
- Increase receptive field

> **Tip:** No learnable parameters Modern architectures often use **strided convolutions** instead of pooling.

---

## 5. Receptive Field

The **receptive field** of a neuron = the region of the input that influences it.

Each conv layer grows the receptive field. For $L$ layers with kernel size $k$ and stride 1:

$$\text{RF} = 1 + L(k-1)$$

Stacking three $3\times 3$ conv layers gives a $7 \times 7$ receptive field — same as one $7 \times 7$ conv, but with **fewer parameters and more nonlinearities**. This is why modern CNNs prefer small kernels.

|Approach|Params|Nonlinearities|
|---|---|---|
|One $7\times 7$ conv|$49 C^2$|1|
|Three $3\times 3$ convs|$27 C^2$|3|

---

## 6. Key Architectural Patterns

### Classic flow

$$\text{Input} \to [\text{Conv} \to \text{BN} \to \text{ReLU} \to \text{Pool}] \times N \to \text{FC} \to \text{Softmax}$$

### Residual block (ResNet)

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

The skip connection lets gradients flow directly backward, enabling networks 100+ layers deep.

> **Important:** Most important architectural innovation in CNNs.

### Bottleneck block (ResNet-50+)

$1\times 1 \to 3\times 3 \to 1\times 1$ — reduces channels, does the heavy 3×3 conv, then restores. Cuts compute.

### Depthwise separable convolution (MobileNet)

Split conv into:

1. **Depthwise:** one filter per input channel (no cross-channel mixing)
2. **Pointwise:** $1\times 1$ conv to mix channels

Reduces params/compute by ~$1/k^2$ — used in mobile/edge models.

### 1×1 convolution

- Mixes information across channels without spatial mixing
- Cheap way to change channel count
- Adds nonlinearity (followed by ReLU)

---

## 7. Famous Architectures

|Architecture|Year|Key Innovation|
|---|---|---|
|**LeNet-5**|1998|First successful CNN (digit recognition)|
|**AlexNet**|2012|ReLU, dropout, GPU training; ImageNet breakthrough|
|**VGG**|2014|Deep nets with stacked 3×3 convs|
|**GoogLeNet/Inception**|2014|Parallel multi-scale filters, 1×1 bottlenecks|
|**ResNet**|2015|Skip connections → very deep networks (50, 101, 152 layers)|
|**DenseNet**|2017|Each layer connected to all subsequent|
|**MobileNet**|2017|Depthwise separable convs for mobile|
|**EfficientNet**|2019|Compound scaling (width, depth, resolution)|
|**ConvNeXt**|2022|Modernized CNN competitive with Vision Transformers|

---

## 8. Translation Equivariance vs Invariance

- **Equivariance:** Shifting the input shifts the output by the same amount. Convolution is equivariant.
- **Invariance:** Output is the same regardless of shift. Achieved via pooling + global pooling.

CNNs are equivariant by construction, invariant only approximately (through pooling and training).

---

## 9. Backprop Through Convolution

Gradients in conv layers also use the chain rule, but the structure is special:

- Gradient w.r.t. **input** = full (transposed) convolution of upstream gradient with flipped kernel
- Gradient w.r.t. **kernel** = convolution of input with upstream gradient

Frameworks handle this.

> **Tip:** Key insight The same filter is reused at every spatial location, so its gradient is the **sum of contributions from every location** where it was applied.

---

## 10. Strengths and Weaknesses

### Strengths

- Parameter-efficient (sharing)
- Translation equivariant
- Hierarchical feature learning (edges → textures → parts → objects)
- Strong inductive bias for spatial/grid data
- Excellent transfer learning (ImageNet pretrained models)

### Weaknesses

- Fixed receptive field (mitigated by depth, dilation, attention)
- Limited global context (compared to Transformers)
- Not rotation/scale invariant by default
- Vision Transformers now match or beat CNNs at scale

---

## 11. When to Use vs Not Use

### Use CNNs when:

- Image, video, audio spectrogram tasks
- Limited compute (vs Vision Transformers)
- Smaller datasets (CNNs have stronger inductive bias → better with less data)
- Edge/mobile deployment

### Don't use CNNs when:

- Sequence/text data (use RNNs/Transformers)
- Very large datasets + compute → Vision Transformers may win
- Highly irregular structures (graphs → GNNs)

---

## 12. CNN vs MLP vs Vision Transformer

|Aspect|MLP|CNN|Vision Transformer|
|---|---|---|---|
|Inductive bias|None|Strong (locality, translation)|Weak|
|Param efficiency|Low|High|Medium|
|Long-range context|Yes (but expensive)|Limited (depth-dependent)|Native (attention)|
|Data efficiency|Low|High|Low (needs pretraining)|
|Best for|Tabular|Vision (small-medium data)|Vision (large data)|

---

## 13. Common Pitfalls

- **Forgetting padding** → spatial size shrinks unexpectedly
- **Wrong input shape order** — PyTorch uses (N, C, H, W); TensorFlow uses (N, H, W, C) by default
- **Not normalizing input** — pretrained models expect specific mean/std
- **BatchNorm before nonlinearity vs after** — usually before in original spec
- **Using FC layers at the end** when global average pooling is cheaper and more robust
- **Pooling too aggressively** — loses spatial info; modern nets use stride-2 convs instead
- **Mismatched train/test augmentation** — e.g., random crop in train, center crop in test

---

## 14. Production Considerations

- **Pretrained backbones** (ResNet, EfficientNet) for transfer learning — almost always start here
- **Image preprocessing pipeline must match training** (resize, normalization, channel order)
- **Quantization** works well on CNNs (more so than Transformers)
- **TensorRT, ONNX, CoreML** for deployment
- **Mixed precision training** for speedup
- **Augmentation:** RandAugment, MixUp, CutMix, AutoAugment for SOTA

---
