"""Extract pre-inference features from raw images (no weak model needed).

Extracts two types of features:
  1. Image statistics  - Hand-crafted, no model (brightness, contrast, etc.)
  2. Frozen backbone   - SqueezeNet1_0 GAP features (lightweight backbone)

Usage:
    python examples/extract_pre_inference_features.py --device cuda
    python examples/extract_pre_inference_features.py --max-samples 1000
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.datasets.wrappers import (IMAGENET_MEAN, IMAGENET_STD,
                                   StreamingImageNetWrapper)

# =============================================================================
# Image Statistics (no model, CPU-only)
# =============================================================================

def compute_image_stats(images: torch.Tensor) -> np.ndarray:
    """Compute hand-crafted image statistics from a batch of normalized tensors.

    Fully vectorized over the batch dimension for speed.

    Returns (B, 12) float32 array with:
      [brightness_mean, brightness_std, contrast,
       channel_r_mean, channel_g_mean, channel_b_mean,
       channel_r_std, channel_g_std, channel_b_std,
       edge_density (laplacian), spatial_freq_energy, color_entropy]
    """
    # Denormalize to [0, 1]
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    imgs = (images * std + mean).clamp(0, 1)

    B = imgs.shape[0]
    stats = np.zeros((B, 12), dtype=np.float32)

    # Grayscale: (B, H, W)
    gray = 0.299 * imgs[:, 0] + 0.587 * imgs[:, 1] + 0.114 * imgs[:, 2]

    # Brightness mean & std
    stats[:, 0] = gray.mean(dim=(1, 2)).numpy()
    stats[:, 1] = gray.std(dim=(1, 2)).numpy()

    # Contrast (max - min per image)
    stats[:, 2] = (gray.amax(dim=(1, 2)) - gray.amin(dim=(1, 2))).numpy()

    # Per-channel mean & std
    for c in range(3):
        stats[:, 3 + c] = imgs[:, c].mean(dim=(1, 2)).numpy()
        stats[:, 6 + c] = imgs[:, c].std(dim=(1, 2)).numpy()

    # Edge density via Laplacian
    lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                              dtype=torch.float32).view(1, 1, 3, 3)
    gray_4d = gray.unsqueeze(1)  # (B, 1, H, W)
    lap = torch.nn.functional.conv2d(gray_4d, lap_kernel, padding=1)
    stats[:, 9] = lap.abs().mean(dim=(1, 2, 3)).numpy()

    # Spatial frequency energy
    dx = (gray[:, :, 1:] - gray[:, :, :-1]).pow(2).mean(dim=(1, 2))
    dy = (gray[:, 1:, :] - gray[:, :-1, :]).pow(2).mean(dim=(1, 2))
    stats[:, 10] = (dx + dy).sqrt().numpy()

    # Color entropy (per-image, using gray histogram)
    for i in range(B):
        hist = torch.histc(gray[i], bins=64, min=0, max=1)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        stats[i, 11] = -(hist * hist.log()).sum().item()

    return stats


# =============================================================================
# Frozen Backbone Feature Extraction
# =============================================================================

def build_frozen_squeezenet(device):
    """Build SqueezeNet1_0 feature extractor (frozen, ~1.2M params)."""
    backbone = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
    backbone.eval()
    # Remove classifier, keep features + adaptive avg pool
    # SqueezeNet features output: (B, 512, H', W')
    feature_extractor = nn.Sequential(
        backbone.features,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ).to(device)
    for p in feature_extractor.parameters():
        p.requires_grad = False
    return feature_extractor


def extract_all(output_file, batch_size=64, device='cuda', max_samples=None):
    """Extract image stats + frozen SqueezeNet features from ImageNet val."""
    print("Building frozen SqueezeNet1_0 feature extractor...")
    backbone = build_frozen_squeezenet(device)

    print("Loading ImageNet streaming validation set...")
    dataset = StreamingImageNetWrapper(split='val', streaming=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    all_img_stats, all_backbone_feats, all_ids = [], [], []
    count = 0

    print("Extracting pre-inference features...")
    with torch.no_grad():
        for sample_ids, images, targets in tqdm(loader, desc="Extracting"):
            # Image statistics (CPU)
            img_stats = compute_image_stats(images)
            all_img_stats.append(img_stats)

            # Frozen backbone features (GPU)
            backbone_feats = backbone(images.to(device)).cpu().numpy()
            all_backbone_feats.append(backbone_feats)

            all_ids.extend(sample_ids)
            count += len(sample_ids)
            if max_samples and count >= max_samples:
                break

    img_stats = np.concatenate(all_img_stats, axis=0)
    backbone_feats = np.concatenate(all_backbone_feats, axis=0)

    if max_samples:
        img_stats = img_stats[:max_samples]
        backbone_feats = backbone_feats[:max_samples]
        all_ids = all_ids[:max_samples]

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    np.savez(
        output_file,
        image_stats=img_stats,
        backbone_features=backbone_feats,
        sample_ids=np.array(all_ids),
    )
    print(f"Saved {len(all_ids)} samples to {output_file}")
    print(f"  image_stats:      {img_stats.shape}")
    print(f"  backbone_features: {backbone_feats.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract pre-inference features')
    default_out = os.path.join(PROJECT_ROOT, 'features',
                               'imagenet_val_pre_inference.npz')
    parser.add_argument('--output', default=default_out)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples for quick testing')
    args = parser.parse_args()

    extract_all(args.output, args.batch_size, args.device, args.max_samples)
