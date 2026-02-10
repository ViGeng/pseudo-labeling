"""Disagreement classifiers for edge-to-cloud offloading decisions.

All classifiers use ONLY edge model outputs (confidence, prediction, features)
since the offloading decision must be made before cloud inference.

Classifiers:
  1. ConfidenceThreshold  - Threshold sweep on edge confidence (2-class only)
  2. LogisticRegression   - Sklearn LR on metadata features
  3. GradientBoosting     - Sklearn GB on metadata + predicted class
  4. MetadataMLP          - PyTorch MLP with class embedding
  5. FeatureMLP           - PyTorch MLP on visual features
  6. FeatureMetaFusion    - Visual features + metadata fusion MLP
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# Base class
# =============================================================================

class BaseClassifier(ABC):
    name = "base"
    requires_features = False
    supports_4cls = True

    @abstractmethod
    def fit(self, data: dict, y: np.ndarray):
        """Train the classifier. data keys: 'metadata', 'pred_class', 'features'."""

    @abstractmethod
    def predict(self, data: dict) -> np.ndarray:
        """Return predicted class labels."""

    def get_model_size_mb(self) -> float:
        """Return estimated model size in MB."""
        return 0.0

    def get_param_count(self) -> int:
        """Return number of parameters."""
        return 0

    def get_gflops(self) -> float:
        """Return estimated GFLOPs for a single inference."""
        return 0.0

    def predict_proba(self, data: dict) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, data: dict, y: np.ndarray, label_names: list) -> tuple:
        """Evaluate and return (metrics_dict, classification_report_str)."""
        preds = self.predict(data)
        metrics = {
            'accuracy': accuracy_score(y, preds),
            'balanced_acc': balanced_accuracy_score(y, preds),
            'f1_macro': f1_score(y, preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(y, preds, average='weighted', zero_division=0),
        }
        if len(label_names) == 2:
            try:
                proba = self.predict_proba(data)
                if proba.ndim == 2:
                    proba = proba[:, 1]
                metrics['auc_roc'] = roc_auc_score(y, proba)
            except (NotImplementedError, Exception):
                pass
        report = classification_report(
            y, preds, labels=range(len(label_names)), target_names=label_names, zero_division=0
        )
        return metrics, report


# =============================================================================
# PyTorch training helpers (DRY)
# =============================================================================

def _compute_class_weights(y: np.ndarray, num_classes: int, device: str):
    counts = np.bincount(y, minlength=num_classes).astype(float)
    weights = len(y) / (num_classes * counts + 1e-8)
    return torch.FloatTensor(weights).to(device)


def _train_torch_model(model, tensors, y_tensor, num_classes, device,
                       epochs=30, lr=1e-3, batch_size=512):
    """Generic PyTorch training loop with balanced class weights."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weights = _compute_class_weights(y_tensor.numpy(), num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    dataset = TensorDataset(*tensors, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = [b.to(device) for b in batch]
            *inputs, targets = batch
            logits = model(*inputs)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    print(f"    Final train loss: {total_loss / len(loader):.4f}")


@torch.no_grad()
def _predict_torch(model, tensors, device, batch_size=4096):
    """Batched PyTorch prediction."""
    model.eval()
    n = tensors[0].shape[0]
    all_logits = []
    for i in range(0, n, batch_size):
        batch = [t[i:i + batch_size].to(device) for t in tensors]
        logits = model(*batch)
        all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


# =============================================================================
# 1. Confidence Threshold (2-class only)
# =============================================================================

class ConfidenceThresholdClassifier(BaseClassifier):
    """If edge confidence < threshold → predict needs_offload."""
    name = "ConfThreshold"
    supports_4cls = False

    def __init__(self):
        self.threshold = 0.5

    def fit(self, data, y):
        confs = data['metadata'][:, 0]
        best_t, best_f1 = 0.5, 0
        for t in np.arange(0.05, 0.95, 0.005):
            preds = (confs < t).astype(int)
            score = f1_score(y, preds, average='macro', zero_division=0)
            if score > best_f1:
                best_f1, best_t = score, t
        self.threshold = best_t
        print(f"    Best threshold: {best_t:.3f} (train F1_macro={best_f1:.4f})")

    def get_model_size_mb(self):
        return 0.0  # Just a float threshold

    def get_param_count(self):
        return 1

    def get_gflops(self):
        return 0.0  # Negligible (formatted as float)

    def predict(self, data):
        return (data['metadata'][:, 0] < self.threshold).astype(int)

    def predict_proba(self, data):
        p = 1.0 - data['metadata'][:, 0]
        return np.column_stack([1 - p, p])


# =============================================================================
# 2. Logistic Regression
# =============================================================================

class LRClassifier(BaseClassifier):
    name = "LogisticRegression"

    def __init__(self, num_classes=2):
        self.model = LogisticRegression(
            max_iter=1000, class_weight='balanced', C=1.0,
        )

    def fit(self, data, y):
        self.model.fit(data['metadata'], y)

    def get_model_size_mb(self):
        # Coefficients + intercept: (n_features + 1) * n_classes * 8 bytes
        if not hasattr(self.model, 'coef_'): return 0.0
        n_params = self.model.coef_.size + self.model.intercept_.size
        return (n_params * 8) / (1024 * 1024)

    def get_param_count(self):
        if not hasattr(self.model, 'coef_'): return 0
        return self.model.coef_.size + self.model.intercept_.size

    def get_gflops(self):
        # Dot product: 2 * n_features * n_classes FLOPs (approx)
        if not hasattr(self.model, 'coef_'): return 0.0
        # n_classes * n_features * 2 (mul+add)
        return (self.model.coef_.size * 2) / 1e9

    def predict(self, data):
        return self.model.predict(data['metadata'])

    def predict_proba(self, data):
        return self.model.predict_proba(data['metadata'])


# =============================================================================
# 3. Gradient Boosting
# =============================================================================

class GBClassifier(BaseClassifier):
    name = "GradientBoosting"

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8
        )

    def _combine(self, data):
        return np.column_stack([data['metadata'], data['pred_class']])

    def fit(self, data, y):
        X = self._combine(data)
        # Balanced sample weights
        counts = np.bincount(y, minlength=self.num_classes).astype(float)
        sample_w = len(y) / (self.num_classes * counts[y])
        self.model.fit(X, y, sample_weight=sample_w)

    def get_model_size_mb(self):
        # Rough estimate for sklearn GB
        if not hasattr(self.model, 'estimators_'): return 0.0
        import pickle
        return len(pickle.dumps(self.model)) / (1024 * 1024)

    def get_param_count(self):
        # Count total nodes in all trees
        if not hasattr(self.model, 'estimators_'): return 0
        return sum(tree[0].tree_.node_count for tree in self.model.estimators_)

    def get_gflops(self):
        # Comparison per node match
        if not hasattr(self.model, 'estimators_'): return 0.0
        # Approx: depth * n_estimators
        return (self.model.n_estimators * self.model.max_depth) / 1e9

    def predict(self, data):
        return self.model.predict(self._combine(data))

    def predict_proba(self, data):
        return self.model.predict_proba(self._combine(data))


# =============================================================================
# 4. Metadata MLP
# =============================================================================

class _MetadataMLPNet(nn.Module):
    def __init__(self, num_classes, num_pred_classes=1000, embed_dim=32, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(num_pred_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(4 + embed_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, meta, pred_cls):
        emb = self.embed(pred_cls)
        return self.net(torch.cat([meta, emb], dim=1))


class MetadataMLPClassifier(BaseClassifier):
    name = "MetadataMLP"

    def __init__(self, num_classes=2, epochs=30, lr=1e-3, batch_size=512):
        self.nc = num_classes
        self.epochs, self.lr, self.bs = epochs, lr, batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    def _tensors(self, data):
        return [
            torch.FloatTensor(data['metadata']),
            torch.LongTensor(data['pred_class']),
        ]

    def fit(self, data, y):
        self.model = _MetadataMLPNet(self.nc)
        _train_torch_model(
            self.model, self._tensors(data), torch.LongTensor(y),
            self.nc, self.device, self.epochs, self.lr, self.bs
        )

    def get_model_size_mb(self):
        if self.model is None: return 0.0
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024 * 1024)

    def get_param_count(self):
        if self.model is None: return 0
        return sum(p.numel() for p in self.model.parameters())

    def get_gflops(self):
        # Simple MLP FLOPs: 2 * sum(in * out)
        if self.model is None: return 0.0
        flops = 0
        # We know the structure: Linear(4+em, 128) -> Linear(128, 64) -> Linear(64, nc)
        # But easier to just iterate modules if possible, or hardcode approximation
        # Hardcoding for robustness since access to internal structure is easy
        # Input: 4 (meta) + 32 (embed) = 36
        flops += 2 * 36 * 128
        flops += 2 * 128 * 64
        flops += 2 * 64 * self.nc
        return flops / 1e9

    def predict(self, data):
        logits = _predict_torch(self.model, self._tensors(data), self.device)
        return logits.argmax(dim=1).numpy()

    def predict_proba(self, data):
        logits = _predict_torch(self.model, self._tensors(data), self.device)
        return torch.softmax(logits, dim=1).numpy()


# =============================================================================
# 5. Feature MLP
# =============================================================================

class _FeatureMLPNet(nn.Module):
    def __init__(self, feat_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, features):
        return self.net(features)


class FeatureMLPClassifier(BaseClassifier):
    name = "FeatureMLP"
    requires_features = True

    def __init__(self, num_classes=2, epochs=30, lr=1e-3, batch_size=512):
        self.nc = num_classes
        self.epochs, self.lr, self.bs = epochs, lr, batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    def _tensors(self, data):
        return [torch.FloatTensor(data['features'])]

    def fit(self, data, y):
        if 'features' not in data:
            raise ValueError("Visual features required. Run extract_features.py first.")
        feat_dim = data['features'].shape[1]
        self.model = _FeatureMLPNet(feat_dim, self.nc)
        _train_torch_model(
            self.model, self._tensors(data), torch.LongTensor(y),
            self.nc, self.device, self.epochs, self.lr, self.bs
        )

    def get_model_size_mb(self):
        if self.model is None: return 0.0
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        return param_size / (1024 * 1024)

    def get_param_count(self):
        if self.model is None: return 0
        return sum(p.numel() for p in self.model.parameters())

    def get_gflops(self):
        if self.model is None or 'features' not in locals().get('data', {}): 
             pass
        flops = 0
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                flops += 2 * m.in_features * m.out_features
        return flops / 1e9

    def predict(self, data):
        logits = _predict_torch(self.model, self._tensors(data), self.device)
        return logits.argmax(dim=1).numpy()

    def predict_proba(self, data):
        logits = _predict_torch(self.model, self._tensors(data), self.device)
        return torch.softmax(logits, dim=1).numpy()


# =============================================================================
# 6. Feature + Metadata Fusion
# =============================================================================

class _FusionMLPNet(nn.Module):
    def __init__(self, feat_dim, num_classes, num_pred_classes=1000,
                 embed_dim=32, hidden=256):
        super().__init__()
        self.embed = nn.Embedding(num_pred_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(feat_dim + 4 + embed_dim, hidden),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, features, meta, pred_cls):
        emb = self.embed(pred_cls)
        return self.net(torch.cat([features, meta, emb], dim=1))


class FeatureMetaFusionClassifier(BaseClassifier):
    name = "FeatureMetaFusion"
    requires_features = True

    def __init__(self, num_classes=2, epochs=30, lr=1e-3, batch_size=512):
        self.nc = num_classes
        self.epochs, self.lr, self.bs = epochs, lr, batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    def _tensors(self, data):
        return [
            torch.FloatTensor(data['features']),
            torch.FloatTensor(data['metadata']),
            torch.LongTensor(data['pred_class']),
        ]

    def fit(self, data, y):
        if 'features' not in data:
            raise ValueError("Visual features required. Run extract_features.py first.")
        feat_dim = data['features'].shape[1]
        self.model = _FusionMLPNet(feat_dim, self.nc)
        _train_torch_model(
            self.model, self._tensors(data), torch.LongTensor(y),
            self.nc, self.device, self.epochs, self.lr, self.bs
        )

    def get_model_size_mb(self):
        if self.model is None: return 0.0
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        return param_size / (1024 * 1024)

    def get_param_count(self):
        if self.model is None: return 0
        return sum(p.numel() for p in self.model.parameters())

    def get_gflops(self):
        if self.model is None: return 0.0
        flops = 0
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                flops += 2 * m.in_features * m.out_features
        return flops / 1e9

    def predict(self, data):
        logits = _predict_torch(self.model, self._tensors(data), self.device)
        return logits.argmax(dim=1).numpy()

    def predict_proba(self, data):
        logits = _predict_torch(self.model, self._tensors(data), self.device)
        return torch.softmax(logits, dim=1).numpy()
