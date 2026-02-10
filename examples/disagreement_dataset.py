"""Merge model predictions and prepare data for disagreement classification.

4-class targets:
  0: both_correct  - Both models predict correctly
  1: only_edge     - Only edge model correct
  2: only_cloud    - Only cloud model correct (offloading candidate)
  3: both_wrong    - Neither model correct

2-class targets:
  0: others        - No offloading benefit
  1: needs_offload - Only cloud correct (offloading will help)
"""

import numpy as np
import pandas as pd

LABEL_NAMES_4 = ['both_correct', 'only_edge', 'only_cloud', 'both_wrong']
LABEL_NAMES_2 = ['others', 'needs_offload']


def load_disagreement_data(edge_csv: str, cloud_csv: str) -> pd.DataFrame:
    """Merge edge/cloud prediction CSVs and compute disagreement labels.

    Merges by row position (both CSVs share the same streaming order).
    """
    edge = pd.read_csv(edge_csv)
    cloud = pd.read_csv(cloud_csv)

    # sample_ids are not unique (encode class, not position), so merge by row index
    edge = edge.reset_index(drop=True)
    cloud = cloud.reset_index(drop=True)
    merged = edge.rename(columns={'prediction': 'prediction_edge',
                                  'confidence': 'confidence_edge'})
    merged['prediction_cloud'] = cloud['prediction']
    merged['confidence_cloud'] = cloud['confidence']

    ec = merged['prediction_edge'] == merged['ground_truth']
    cc = merged['prediction_cloud'] == merged['ground_truth']

    labels_4 = np.where(
        ec & cc, 0, np.where(ec & ~cc, 1, np.where(~ec & cc, 2, 3))
    )
    merged['label_4cls'] = labels_4
    merged['label_2cls'] = (labels_4 == 2).astype(int)
    return merged


def prepare_data(df: pd.DataFrame, features_file: str = None) -> dict:
    """Prepare edge-side features for classification.

    Only uses information available on the edge device at inference time.

    Returns dict with keys:
      'metadata': (N, 4) float  - confidence-derived features
      'pred_class': (N,) int    - edge predicted class IDs
      'features': (N, F) float  - visual features (if features_file provided)
    """
    conf = df['confidence_edge'].values.astype(np.float32)
    metadata = np.column_stack([
        conf,
        np.log(conf + 1e-8),
        conf ** 2,
        1.0 - conf,
    ]).astype(np.float32)

    data = {
        'metadata': metadata,
        'pred_class': df['prediction_edge'].values.astype(np.int64),
    }

    if features_file:
        feat_data = np.load(features_file, allow_pickle=True)
        feat_vecs = feat_data['features']
        n_feats = len(feat_vecs)
        n_data = len(df)
        n = min(n_feats, n_data)
        # Match by row position (same streaming order)
        features = np.zeros((n_data, feat_vecs.shape[1]), dtype=np.float32)
        features[:n] = feat_vecs[:n]
        print(f"Loaded {n}/{n_data} visual features ({feat_vecs.shape[1]}-dim)")
        data['features'] = features

    return data


def print_distribution(labels, label_names):
    """Print label distribution."""
    total = len(labels)
    print(f"  {'Label':<20} {'Count':>8} {'Pct':>8}")
    print(f"  {'-' * 38}")
    for i, name in enumerate(label_names):
        count = int((labels == i).sum())
        print(f"  {name:<20} {count:>8} {count / total * 100:>7.1f}%")
    print(f"  {'Total':<20} {total:>8}")
