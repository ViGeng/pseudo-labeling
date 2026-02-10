"""Train and evaluate ALL disagreement classifiers (pre + post inference).

Post-inference classifiers (1-6): use edge model outputs (confidence, etc.)
Pre-inference classifiers (7-11): use raw image features (no weak model)

Usage:
    # Post-inference only (fast, no image data needed):
    python examples/run_experiment.py

    # Post-inference with visual features:
    python examples/run_experiment.py \
        --post-features features/imagenet_val_mobilenet_features.npz

    # Pre-inference (requires extract_pre_inference_features.py first):
    python examples/run_experiment.py \
        --pre-features features/imagenet_val_pre_inference.npz

    # All classifiers including end-to-end fine-tuned ResNet-18:
    python examples/run_experiment.py \
        --pre-features features/imagenet_val_pre_inference.npz \
        --post-features features/imagenet_val_mobilenet_features.npz \
        --extract-images --max-images 10000

    # Full run with everything:
    python examples/run_experiment.py \
        --pre-features features/imagenet_val_pre_inference.npz \
        --post-features features/imagenet_val_mobilenet_features.npz
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from classifiers import (ConfidenceThresholdClassifier,
                         FeatureMetaFusionClassifier, FeatureMLPClassifier,
                         GBClassifier, LRClassifier, MetadataMLPClassifier)
from disagreement_dataset import (LABEL_NAMES_2, LABEL_NAMES_4,
                                  load_disagreement_data, prepare_data,
                                  print_distribution)
from pre_inference_classifiers import (FinetunedMobileNetV2Classifier,
                                       FinetunedMobileNetV3SmallClassifier,
                                       FinetunedResNet18Classifier,
                                       FinetunedShuffleNetV2Classifier,
                                       FinetunedSqueezeNetClassifier,
                                       FrozenBackboneLRClassifier,
                                       FrozenBackboneMLPClassifier,
                                       HybridStatsBackboneClassifier,
                                       ImageStatsGBClassifier,
                                       SimpleCNNClassifier)


def split_data(data_dict, idx):
    """Slice every array in data dict by index."""
    return {k: v[idx] for k, v in data_dict.items()}


def load_pre_inference_features(features_file, n_data):
    """Load pre-extracted image stats + backbone features."""
    data = {}
    feat = np.load(features_file, allow_pickle=True)

    for key in ['image_stats', 'backbone_features']:
        arr = feat[key]
        n = min(len(arr), n_data)
        padded = np.zeros((n_data, arr.shape[1]), dtype=np.float32)
        padded[:n] = arr[:n]
        data[key] = padded

    print(f"Loaded pre-inference features: "
          f"image_stats={data['image_stats'].shape}, "
          f"backbone={data['backbone_features'].shape}")
    return data


def load_images_from_stream(n_samples, max_images=None):
    """Load raw image tensors from HF streaming dataset."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.datasets.wrappers import StreamingImageNetWrapper

    limit = max_images or n_samples
    print(f"Loading {limit} raw images from ImageNet streaming...")
    dataset = StreamingImageNetWrapper(split='val', streaming=True)
    loader = DataLoader(dataset, batch_size=64, num_workers=0)

    all_imgs = []
    count = 0
    for _, images, _ in tqdm(loader, desc="Loading images", total=limit // 64):
        all_imgs.append(images.numpy())
        count += images.shape[0]
        if count >= limit:
            break

    imgs = np.concatenate(all_imgs, axis=0)[:limit]
    # Pad to n_samples if needed
    if len(imgs) < n_samples:
        padded = np.zeros((n_samples, *imgs.shape[1:]), dtype=np.float32)
        padded[:len(imgs)] = imgs
        imgs = padded
    return imgs


def _can_run(clf, data):
    """Check if classifier's data requirements are met."""
    if getattr(clf, 'requires_features', False) and 'features' not in data:
        return False
    if getattr(clf, 'requires_image_stats', False) and 'image_stats' not in data:
        return False
    if getattr(clf, 'requires_backbone_features', False) and 'backbone_features' not in data:
        return False
    if getattr(clf, 'requires_images', False) and 'images' not in data:
        return False
    if getattr(clf, 'requires_images', False) and 'images' not in data:
        return False
    return True


def measure_latency(clf, data, n_warmup=5, n_runs=20, device='cuda'):
    """Measure inference latency (ms) per sample (batch size 1)."""
    if not _can_run(clf, data):
        return 0.0

    # Create batch-1 data
    single_data = {k: v[:1] for k, v in data.items()}
    # Ensure torch models are on device if not already handled by predict
    # The predict method handles device movement internally usually, 
    # but input tensors might need to be on CPU if predict moves them.
    # Our predict methods take numpy/torch and handle it.
    
    try:
        # Warmup
        for _ in range(n_warmup):
            clf.predict(single_data)
            
        start = time.time()
        for _ in range(n_runs):
            clf.predict(single_data)
        end = time.time()
        
        return ((end - start) / n_runs) * 1000  # ms
    except Exception:
        return 0.0


def run_experiment(edge_csv, cloud_csv, post_features_file=None,
                   pre_features_file=None, extract_images=False,
                   max_images=None, test_size=0.2, seed=42):
    # ---- Load & prepare data ------------------------------------------------
    print("=" * 60)
    print("  Loading Data")
    print("=" * 60)
    df = load_disagreement_data(edge_csv, cloud_csv)
    if max_images:
        print(f"Subsampling dataframe to {max_images} samples for quick testing.")
        df = df.iloc[:max_images]
    n_samples = len(df)
    print(f"Total merged samples: {n_samples}")

    # Post-inference features (edge model outputs)
    data = prepare_data(df, post_features_file)

    # Pre-inference features (raw image)
    if pre_features_file:
        pre_data = load_pre_inference_features(pre_features_file, n_samples)
        data.update(pre_data)

    # Raw images for end-to-end fine-tuning
    if extract_images:
        data['images'] = load_images_from_stream(n_samples, max_images)
        print(f"Loaded image tensors: {data['images'].shape}")

    # ---- Train/Test split ---------------------------------------------------
    indices = np.arange(n_samples)
    stratify = df['label_4cls'].values
    if n_samples < 50:
        print("Warning: Small dataset, disabling stratification.")
        stratify = None
        
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed,
        stratify=stratify,
    )

    # Collect all results across both tasks for final combined summary
    all_results_list = []  # List of dicts for DataFrame
    all_results = {}  # key: (num_classes, clf_name) → metrics

    # ---- Run experiments for 2-class and 4-class ----------------------------
    for num_classes, label_names, label_col in [
        (2, LABEL_NAMES_2, 'label_2cls'),
        (4, LABEL_NAMES_4, 'label_4cls'),
    ]:
        print(f"\n{'=' * 60}")
        print(f"  {num_classes}-CLASS EXPERIMENT")
        print(f"{'=' * 60}")

        y_train = df[label_col].values[train_idx]
        y_test = df[label_col].values[test_idx]

        print("\nTrain distribution:")
        print_distribution(y_train, label_names)
        print("\nTest distribution:")
        print_distribution(y_test, label_names)

        train_data = split_data(data, train_idx)
        test_data = split_data(data, test_idx)

        # ---- Post-inference classifiers (1-6) ----
        post_clfs = []
        if num_classes == 2:
            post_clfs.append(ConfidenceThresholdClassifier())
        post_clfs.extend([
            LRClassifier(num_classes),
            GBClassifier(num_classes),
            MetadataMLPClassifier(num_classes),
        ])
        if 'features' in data:
            post_clfs.extend([
                FeatureMLPClassifier(num_classes),
                FeatureMetaFusionClassifier(num_classes),
            ])

        # ---- Pre-inference classifiers (7-11) ----
        pre_clfs = [
            ImageStatsGBClassifier(num_classes),
            FrozenBackboneLRClassifier(num_classes),
            FrozenBackboneMLPClassifier(num_classes),
            HybridStatsBackboneClassifier(num_classes),
            FinetunedResNet18Classifier(num_classes),
            FinetunedMobileNetV3SmallClassifier(num_classes),
            FinetunedShuffleNetV2Classifier(num_classes),
            FinetunedSqueezeNetClassifier(num_classes),
            FinetunedMobileNetV2Classifier(num_classes),
            SimpleCNNClassifier(num_classes),
        ]

        # Collect results for this task
        results = []

        # --- Run post-inference ---
        runnable_post = [c for c in post_clfs
                         if (c.supports_4cls or num_classes != 4)
                         and _can_run(c, data)]
        if runnable_post:
            print(f"\n{'─' * 50}")
            print(f"  POST-INFERENCE CLASSIFIERS (use edge model output)")
            print(f"{'─' * 50}")
            for clf in runnable_post:
                print(f"\n--- [POST] {clf.name} ---")
                print("  Training...")
                try:
                    clf.fit(train_data, y_train)
                except Exception as e:
                    print(f"  FAILED to train {clf.name}: {e}")
                    continue
                print("  Evaluating...")
                print("  Evaluating...")
                metrics, report = clf.evaluate(test_data, y_test, label_names)
                metrics['stage'] = 'post'
                metrics['classifier'] = clf.name
                metrics['num_classes'] = num_classes
                
                # Computing metrics
                metrics['latency_ms'] = measure_latency(clf, test_data)
                metrics['model_size_mb'] = clf.get_model_size_mb()
                metrics['params'] = clf.get_param_count()
                metrics['gflops'] = clf.get_gflops()
                
                results.append((clf.name, metrics))
                all_results[(num_classes, clf.name)] = metrics
                all_results_list.append(metrics)
                
                for k, v in metrics.items():
                    if k not in ['stage', 'classifier', 'num_classes']:
                        val = v if isinstance(v, (int, float)) else str(v)
                        print(f"    {k}: {val}")
                print(f"\n{report}")

        # --- Run pre-inference ---
        runnable_pre = [c for c in pre_clfs
                        if (c.supports_4cls or num_classes != 4)
                        and _can_run(c, data)]
        if runnable_pre:
            print(f"\n{'─' * 50}")
            print(f"  PRE-INFERENCE CLASSIFIERS (raw image only)")
            print(f"{'─' * 50}")
            for clf in runnable_pre:
                print(f"\n--- [PRE] {clf.name} ---")
                print("  Training...")
                try:
                    clf.fit(train_data, y_train)
                except Exception as e:
                    print(f"  FAILED to train {clf.name}: {e}")
                    continue
                print("  Evaluating...")
                print("  Evaluating...")
                metrics, report = clf.evaluate(test_data, y_test, label_names)
                metrics['stage'] = 'pre'
                metrics['classifier'] = clf.name
                metrics['num_classes'] = num_classes
                
                # Computing metrics
                metrics['latency_ms'] = measure_latency(clf, test_data)
                metrics['model_size_mb'] = clf.get_model_size_mb()
                metrics['params'] = clf.get_param_count()
                metrics['gflops'] = clf.get_gflops()
                
                results.append((clf.name, metrics))
                all_results[(num_classes, clf.name)] = metrics
                all_results_list.append(metrics)

                for k, v in metrics.items():
                    if k not in ['stage', 'classifier', 'num_classes']:
                        val = v if isinstance(v, (int, float)) else str(v)
                        print(f"    {k}: {val}")
                print(f"\n{report}")

        # Per-task summary
        _print_summary(results, num_classes)

        # Note skipped classifiers
        skipped_post = [c for c in post_clfs if not _can_run(c, data)]
        skipped_pre = [c for c in pre_clfs if not _can_run(c, data)]
        if skipped_post or skipped_pre:
            print("\n  [SKIPPED - missing features]:")
            for c in skipped_post + skipped_pre:
                print(f"    - {c.name}")

    # ---- Combined summary table ----
    _print_combined_summary(all_results)
    
    # ---- Save Results ----
    out_dir = Path(SCRIPT_DIR) / 'outputs'
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / 'experiment_results.csv'
    pd.DataFrame(all_results_list).to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")


def _print_summary(results, num_classes):
    """Print per-task summary table."""
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY ({num_classes}-class)")
    print(f"{'=' * 70}")
    header = (f"{'Stage':<6} {'Classifier':<22} {'Acc':>7} "
              f"{'BalAcc':>7} {'F1_M':>7} {'F1_W':>7}")
    if num_classes == 2:
        header += f" {'AUC':>7}"
    print(header)
    print('-' * len(header))
    for name, m in results:
        stage = m.get('stage', '?')[:4]
        row = (f"{stage:<6} {name:<22} {m['accuracy']:>7.4f} "
               f"{m['balanced_acc']:>7.4f} {m['f1_macro']:>7.4f} "
               f"{m['f1_weighted']:>7.4f}")
        if num_classes == 2 and 'auc_roc' in m:
            row += f" {m['auc_roc']:>7.4f}"
        print(row)


def _print_combined_summary(all_results):
    """Print final combined summary across both tasks."""
    print(f"\n{'=' * 80}")
    print(f"  COMBINED SUMMARY: PRE-INFERENCE vs POST-INFERENCE")
    print(f"{'=' * 80}")
    print(f"{'Stage':<6} {'Classifier':<22} {'2cls_Acc':>8} "
          f"{'2cls_F1M':>8} {'2cls_AUC':>8} {'4cls_Acc':>8} {'4cls_F1M':>8}")
    print('-' * 80)

    # Gather unique classifier names preserving order
    seen = set()
    ordered_names = []
    for (nc, name) in all_results:
        if name not in seen:
            seen.add(name)
            ordered_names.append(name)

    for name in ordered_names:
        m2 = all_results.get((2, name), {})
        m4 = all_results.get((4, name), {})
        stage = (m2 or m4).get('stage', '?')[:4]

        c2_acc = f"{m2['accuracy']:.4f}" if m2 else '   -   '
        c2_f1 = f"{m2['f1_macro']:.4f}" if m2 else '   -   '
        c2_auc = f"{m2['auc_roc']:.4f}" if m2 and 'auc_roc' in m2 else '   -   '
        c4_acc = f"{m4['accuracy']:.4f}" if m4 else '   -   '
        c4_f1 = f"{m4['f1_macro']:.4f}" if m4 else '   -   '

        print(f"{stage:<6} {name:<22} {c2_acc:>8} "
              f"{c2_f1:>8} {c2_auc:>8} {c4_acc:>8} {c4_f1:>8}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Disagreement classifier experiments (pre + post inference)'
    )
    parser.add_argument(
        '--edge-csv',
        default=os.path.join(PROJECT_ROOT, 'outputs',
                             'imagenet_streaming-mobilenet_v3_large-val.csv'),
    )
    parser.add_argument(
        '--cloud-csv',
        default=os.path.join(PROJECT_ROOT, 'outputs',
                             'imagenet_streaming-swin_t-val.csv'),
    )
    parser.add_argument('--post-features', default=None,
                        help='Post-inference MobileNet features (.npz)')
    parser.add_argument('--pre-features', default=None,
                        help='Pre-inference image stats + backbone (.npz)')
    parser.add_argument('--extract-images', action='store_true',
                        help='Load raw images for FinetunedResNet18')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Max images to load (for quick test)')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        args.edge_csv, args.cloud_csv,
        post_features_file=args.post_features,
        pre_features_file=args.pre_features,
        extract_images=args.extract_images,
        max_images=args.max_images,
        test_size=args.test_size,
        seed=args.seed,
    )
