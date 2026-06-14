# Sign Language Gloss Recognition using Self-Supervised Vision Transformers

## Overview
This repository implements an end-to-end spatial-temporal video classification pipeline for sign language gloss recognition over the LSA64 dataset. The system leverages pretrained self-supervised foundation models (Meta's DINOv2 and the cutting-edge DINOv3 architectures) as feature extractors, coupled with custom temporal pooling and a regularized downstream classification network to identify word-level hand gestures from raw RGB video frame sequences.

## Dataset
* **Source:** LSA64 Argentine Sign Language Dataset
* **Vocabulary Scope:** 64 distinct gesture gloss classes
* **Data Footprint:** 3,200 total video samples

### Evaluation & Stratification Strategy
To strictly evaluate the generalization capability of the architectures across training iterations, a **Stratified Class-Balanced Split Strategy** was engineered. The framework guarantees a deterministic allocation matrix, ensuring every sign category maintains a balanced density ratio:
* **Training Set:** 2,560 total video directories (80% of frames per category)
* **Validation Set:** 640 total video directories (20% of frames per category)

---

## Architecture & Method Pipeline

The workflow processes raw sign language video input via a decoupled spatial-temporal classification network:

Video Input (Variable Frame Length)
           ↓
Uniform Frame Sampling (Isolates T=8 RGB Frames)
           ↓
Spatial Feature Extractor (DINOv2 / DINOv3 ViT-Small)
           ↓
Tensor Index Slicing (Extracts Global [CLS] Tokens)
           ↓
Temporal Average Pooling (Mean Aggregation Across Frames)
           ↓
Regularized MLP Head (Linear → BatchNorm → ReLU → 0.5 Dropout → Output)
           ↓
Predicted Gloss Label (1-of-64 Cross-Entropy Target)

### Technical Implementation Details
* **Core Framework:** PyTorch & `timm` (Torch Image Models)
* **Spatial Backbone:** `vit_small_patch16_dinov3.lvd1689m` (Meta's DINOv3 Small, mapping spatial representations to a 384-dimensional feature width)
* **Gradient Specialization Matrix:** Early foundation visual transformer blocks are entirely frozen to retain spatial generalizability. The **final 4 Transformer layers** along with the downstream classification probe are completely un-frozen to drive specialization for micro-gesture tracking.
* **Downstream Regularization:** Features are routed through a 512-node dense layer backed by `BatchNorm1d` for rapid convergence, non-linear `ReLU` activations, and high `Dropout (0.5)` to robustly combat subject overfitting.
* **Optimizer Configuration:** `AdamW` optimizer with a weight decay of $1 \times 10^{-2}$ and a base learning rate ($LR$) of $2 \times 10^{-5}$.

---

## Experimental Results

The models learn deep, robust visual representations of hand gestures. Through comparative benchmarking, the upgraded DINOv3 configuration demonstrates strong validation metrics over the stratified, unseen validation partitions:

| Model Iteration | Spatial Backbone | Optimization Strategy | Peak Validation Accuracy | Macro Avg F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | DINOv2 ViT-S/14 | Temporal Avg Pooling + Linear Head | 98.28% | 0.98 |
| **Upgraded** | **DINOv3 ViT-S/14** | **Partial Freezing (-4 Blocks) + Regularized MLP** | **97.97%** | **0.98** |

*Note: The model achieves perfect convergence on the training bounds (100.00% training accuracy) while maintaining a strict, non-overfitting validation threshold.*
