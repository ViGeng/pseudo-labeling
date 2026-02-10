"""Extract MobileNet V3 penultimate features from ImageNet streaming dataset.

Saves features + sample_ids to a .npz file for use with feature-based
classifiers (FeatureMLP, FeatureMetaFusion).

Usage:
    python examples/extract_features.py --device cuda
    python examples/extract_features.py --max-samples 1000  # quick test
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.datasets.wrappers import StreamingImageNetWrapper
from src.models.loader import load_model


def extract_features(output_file, batch_size=64, device='cuda', max_samples=None):
    """Extract penultimate (avgpool) features from MobileNet V3 Large."""

    print("Loading MobileNet V3 Large...")
    adapter = load_model('mobilenet_v3_large')
    adapter.to(device)
    adapter.eval()

    # Hook into avgpool to capture 960-dim features
    feat_store = {}

    def hook_fn(module, inp, out):
        feat_store['feat'] = out.flatten(1).detach()

    handle = adapter.model.avgpool.register_forward_hook(hook_fn)

    print("Loading ImageNet streaming validation set...")
    dataset = StreamingImageNetWrapper(split='val', streaming=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    all_features, all_ids = [], []
    count = 0

    print("Extracting features...")
    with torch.no_grad():
        for sample_ids, images, targets in tqdm(loader, desc="Extracting"):
            images = images.to(device)
            _ = adapter(images)
            all_features.append(feat_store['feat'].cpu().numpy())
            all_ids.extend(sample_ids)
            count += len(sample_ids)
            if max_samples and count >= max_samples:
                break

    handle.remove()

    features = np.concatenate(all_features, axis=0)
    if max_samples:
        features = features[:max_samples]
        all_ids = all_ids[:max_samples]

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    np.savez(output_file, features=features, sample_ids=np.array(all_ids))
    print(f"Saved {len(all_ids)} feature vectors ({features.shape[1]}-dim) to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract MobileNet features')
    default_out = os.path.join(PROJECT_ROOT, 'features',
                               'imagenet_val_mobilenet_features.npz')
    parser.add_argument('--output', default=default_out)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples for quick testing')
    args = parser.parse_args()

    extract_features(args.output, args.batch_size, args.device, args.max_samples)
