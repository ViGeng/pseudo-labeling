# Disagreement Classifier Examples

Classifiers that predict whether an image should be offloaded from an edge
model (MobileNet V3) to a cloud model (Swin-T) based on model disagreement.

## Problem

| Category | Description | ImageNet Val % |
|----------|-------------|----------------|
| both_correct | Both models correct | 72.3% |
| only_edge | Only MobileNet correct | 3.0% |
| only_cloud | Only Swin-T correct (**offloading helps**) | 8.8% |
| both_wrong | Neither model correct | 15.9% |

**2-class simplification:** `needs_offload` (only_cloud, 8.8%) vs `others` (91.2%).

## All Classifiers

### Post-Inference (use edge model output, decide after running weak model)

| # | Name | Input | Design Rationale |
|---|------|-------|------------------|
| 1 | ConfThreshold | edge confidence scalar | Simplest baseline: low confidence → offload (2-class only) |
| 2 | LogisticRegression | 4 metadata features | Linear decision on confidence-derived features |
| 3 | GradientBoosting | metadata + pred class ID | Trees can split on per-class difficulty patterns |
| 4 | MetadataMLP | metadata + class embedding | Learned class embeddings capture per-class offloading tendency |
| 5 | FeatureMLP | MobileNet penultimate (960-d) | Visual features from the edge model's own representation |
| 6 | FeatureMetaFusion | visual + metadata + class emb | Full fusion of all post-inference signals |

### Pre-Inference (raw image only, decide before running weak model)

| # | Name | Input | Design Rationale |
|---|------|-------|------------------|
| 7 | ImageStatsGB | 12 hand-crafted stats | Zero-model baseline: blur/contrast/entropy predict difficulty |
| 8 | FrozenBackboneLR | SqueezeNet features (512-d) | Lightweight frozen backbone + linear head; fast, no training of backbone |
| 9 | FrozenBackboneMLP | SqueezeNet features (512-d) | Same features but nonlinear MLP captures complex boundaries |
| 10 | HybridStatsBackbone | stats (12-d) + backbone (512-d) | Low-level stats + semantics cover different failure modes |
| 11 | FinetunedResNet18 | raw images (224×224×3) | End-to-end learned: discovers visual patterns that cause disagreement |

### Input/Output Summary

| # | Classifier | Stage | Input | Output |
|---|-----------|-------|-------|--------|
| 1 | ConfThreshold | post | `confidence_edge` (1 scalar) | 2-class label |
| 2 | LogisticRegression | post | `metadata` (4 floats: conf, log_conf, conf², 1-conf) | 2/4-class label + probabilities |
| 3 | GradientBoosting | post | `metadata` (4) + `pred_class` (1 int) | 2/4-class label + probabilities |
| 4 | MetadataMLP | post | `metadata` (4) + `pred_class` → 32-d embedding | 2/4-class label + probabilities |
| 5 | FeatureMLP | post | MobileNet avgpool features (960-d) | 2/4-class label + probabilities |
| 6 | FeatureMetaFusion | post | features (960-d) + metadata (4) + class emb (32-d) | 2/4-class label + probabilities |
| 7 | ImageStatsGB | pre | 12 image stats (brightness, contrast, edges, ...) | 2/4-class label + probabilities |
| 8 | FrozenBackboneLR | pre | SqueezeNet1_0 GAP features (512-d) | 2/4-class label + probabilities |
| 9 | FrozenBackboneMLP | pre | SqueezeNet1_0 GAP features (512-d) | 2/4-class label + probabilities |
| 10 | HybridStatsBackbone | pre | image stats (12) + backbone (512) = 524-d | 2/4-class label + probabilities |
| 11 | FinetunedResNet18 | pre | raw image tensor (3×224×224) | 2/4-class label + probabilities |

## Quick Start

```bash
# 1. Post-inference classifiers only (fast, metadata from CSVs):
python examples/run_experiment.py

# 2. Extract pre-inference features (image stats + frozen SqueezeNet):
python examples/extract_pre_inference_features.py --device cuda

# 3. Run both pre- and post-inference classifiers:
python examples/run_experiment.py \
    --pre-features features/imagenet_val_pre_inference.npz

# 4. (Optional) Also extract MobileNet features for classifiers 5-6:
python examples/extract_features.py --device cuda
python examples/run_experiment.py \
    --pre-features features/imagenet_val_pre_inference.npz \
    --post-features features/imagenet_val_mobilenet_features.npz

# 5. (Optional) Include end-to-end ResNet-18 (slow, needs GPU + streaming):
python examples/run_experiment.py \
    --pre-features features/imagenet_val_pre_inference.npz \
    --extract-images --max-images 10000
```

## File Structure

```
examples/
├── README.md                           # This file
├── disagreement_dataset.py             # CSV merging, label computation, feature prep
├── classifiers.py                      # Post-inference classifiers (1-6)
├── pre_inference_classifiers.py        # Pre-inference classifiers (7-11)
├── extract_features.py                 # MobileNet penultimate feature extraction
├── extract_pre_inference_features.py   # Image stats + frozen SqueezeNet extraction
└── run_experiment.py                   # Main experiment: train, eval, compare all
```
