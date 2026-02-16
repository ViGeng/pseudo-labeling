"""Pre-inference disagreement classifiers (raw images only, no weak model).

These classifiers predict disagreement categories BEFORE running the edge
model. The goal is to decide "should we even run the edge model, or send
directly to cloud?" by looking at raw image properties.

Classifiers:
  7.  ImageStatsGB        - Hand-crafted image stats → GradientBoosting
  8.  FrozenBackboneLR    - Frozen SqueezeNet features → LogisticRegression
  9.  FrozenBackboneMLP   - Frozen SqueezeNet features → MLP
  10. HybridStatsBackbone - Image stats + frozen features → GradientBoosting
  11. FinetunedResNet18   - End-to-end fine-tuned ResNet-18
  12. FinetunedMobileNetV3 - End-to-end fine-tuned MobileNetV3-Small
  13. FinetunedShuffleNetV2 - End-to-end fine-tuned ShuffleNetV2 0.5x
  14. FinetunedSqueezeNet   - End-to-end fine-tuned SqueezeNet 1.1
  15. FinetunedMobileNetV2 - End-to-end fine-tuned MobileNetV2
  16. SimpleCNN             - Custom efficient 3-layer CNN
  17. ImageStatsRF          - Hand-crafted image stats -> Random Forest
  18. FrozenBackboneKNN     - Frozen SqueezeNet features -> k-NN
  19. FinetunedEfficientNetB0 - End-to-end fine-tuned EfficientNet-B0
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from classifiers import (BaseClassifier, _compute_class_weights,
                         _predict_torch, _train_torch_model)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# GFLOPs Helper
# =============================================================================

def count_flops(model, input_size=(1, 3, 224, 224)):
    """Estimate GFLOPs for a PyTorch model."""
    flops = 0.0
    # Create dummy input to trace shapes (if needed) but here we iterate modules
    # For a more accurate count, we'd need to hook forward pass.
    # We will use a simplified hook approach.
    
    layer_flops = []

    def conv_hook(self, input, output):
        # (2 * Cin * K * K * Cout * H * W) / groups
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels // self.groups)
        bias_ops = 1 if self.bias is not None else 0
        ops_per_element = kernel_ops + bias_ops
        total_ops = output_channels * output_height * output_width * ops_per_element
        # Multiply by 2 for MACs approximately? Standard is simply MACs or 2*MACs.
        # We will count MACs as ~ FLOPs for simplicity or 2*MACs.
        # User asked for "GFLOPS". Usually 2 * MACs.
        layer_flops.append(2 * total_ops)

    def linear_hook(self, input, output):
        # 2 * Cin * Cout
        weight_ops = self.weight.nelement() * 2
        bias_ops = self.bias.nelement() if self.bias is not None else 0
        layer_flops.append(weight_ops + bias_ops)

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    # Run dummy pass
    device = next(model.parameters()).device
    with torch.no_grad():
        try:
            inp = torch.zeros(input_size).to(device)
            model(inp)
        except Exception:
            pass # fallback or error

    for h in hooks:
        h.remove()

    total_flops = sum(layer_flops)
    return total_flops / 1e9


# =============================================================================
# 7. Image Statistics → GradientBoosting
# =============================================================================

class ImageStatsGBClassifier(BaseClassifier):
    """Hand-crafted image statistics → GradientBoosting."""
    name = "ImageStatsGB"
    requires_image_stats = True

    def __init__(self, num_classes=2):
        self.nc = num_classes
        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
        )

    def fit(self, data, y):
        X = data['image_stats']
        counts = np.bincount(y, minlength=self.nc).astype(float)
        sw = len(y) / (self.nc * counts[y])
        self.model.fit(X, y, sample_weight=sw)

    def get_model_size_mb(self):
        import pickle
        if not hasattr(self.model, 'estimators_'): return 0.0
        return len(pickle.dumps(self.model)) / (1024 * 1024)

    def get_param_count(self):
        if not hasattr(self.model, 'estimators_'): return 0
        return sum(tree[0].tree_.node_count for tree in self.model.estimators_)

    def get_gflops(self):
        if not hasattr(self.model, 'estimators_'): return 0.0
        return (self.model.n_estimators * self.model.max_depth) / 1e9

    def predict(self, data):
        return self.model.predict(data['image_stats'])

    def predict_proba(self, data):
        return self.model.predict_proba(data['image_stats'])


# =============================================================================
# 17. Image Statistics -> Random Forest
# =============================================================================

class ImageStatsRFClassifier(BaseClassifier):
    """Hand-crafted image statistics -> Random Forest."""
    name = "ImageStatsRF"
    requires_image_stats = True

    def __init__(self, num_classes=2):
        self.nc = num_classes
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=10, n_jobs=-1
        )

    def fit(self, data, y):
        X = data['image_stats']
        # RF handles class weights internally
        self.model.fit(X, y) # sample_weight not strictly needed if balanced? actually RF has class_weight='balanced'

    def get_model_size_mb(self):
        import pickle
        if not hasattr(self.model, 'estimators_'): return 0.0
        return len(pickle.dumps(self.model)) / (1024 * 1024)

    def get_param_count(self):
        if not hasattr(self.model, 'estimators_'): return 0
        return sum(tree.tree_.node_count for tree in self.model.estimators_)

    def get_gflops(self):
        if not hasattr(self.model, 'estimators_'): return 0.0
        # Average depth * n_estimators
        return (self.model.n_estimators * 10) / 1e9 # approx max_depth

    def predict(self, data):
        return self.model.predict(data['image_stats'])

    def predict_proba(self, data):
        return self.model.predict_proba(data['image_stats'])


# =============================================================================
# 8. Frozen SqueezeNet → Logistic Regression
# =============================================================================

class FrozenBackboneLRClassifier(BaseClassifier):
    """Frozen SqueezeNet1_0 features (512-dim) → Logistic Regression."""
    name = "FrozenBackboneLR"
    requires_backbone_features = True

    def __init__(self, num_classes=2):
        self.model = LogisticRegression(
            max_iter=1000, class_weight='balanced', C=1.0,
        )

    def fit(self, data, y):
        self.model.fit(data['backbone_features'], y)

    def get_model_size_mb(self):
        if not hasattr(self.model, 'coef_'): return 0.0
        n_params = self.model.coef_.size + self.model.intercept_.size
        return (n_params * 8) / (1024 * 1024)

    def get_param_count(self):
        if not hasattr(self.model, 'coef_'): return 0
        return self.model.coef_.size + self.model.intercept_.size

    def get_gflops(self):
        if not hasattr(self.model, 'coef_'): return 0.0
        return (self.model.coef_.size * 2) / 1e9

    def predict(self, data):
        return self.model.predict(data['backbone_features'])

    def predict_proba(self, data):
        return self.model.predict_proba(data['backbone_features'])


# =============================================================================
# 18. Frozen SqueezeNet -> k-Nearest Neighbors
# =============================================================================

class FrozenBackboneKNNClassifier(BaseClassifier):
    """Frozen SqueezeNet features -> k-NN."""
    name = "FrozenBackboneKNN"
    requires_backbone_features = True

    def __init__(self, num_classes=2):
        self.model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    def fit(self, data, y):
        self.model.fit(data['backbone_features'], y)

    def get_model_size_mb(self):
        # KNN stores training data: n_samples * n_features * 4 bytes
        if not hasattr(self.model, '_fit_X'): return 0.0
        return (self.model._fit_X.size * 4) / (1024 * 1024)

    def get_param_count(self):
        return 0 # Non-parametric

    def get_gflops(self):
        # Distance calculation per sample: n_train * n_features
        if not hasattr(self.model, '_fit_X'): return 0.0
        n_train, n_feat = self.model._fit_X.shape
        return (n_train * n_feat) / 1e9

    def predict(self, data):
        return self.model.predict(data['backbone_features'])

    def predict_proba(self, data):
        return self.model.predict_proba(data['backbone_features'])


# =============================================================================
# 9. Frozen SqueezeNet → MLP
# =============================================================================

class _BackboneMLPNet(nn.Module):
    def __init__(self, feat_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, features):
        return self.net(features)


class FrozenBackboneMLPClassifier(BaseClassifier):
    """Frozen SqueezeNet1_0 features (512-dim) → MLP."""
    name = "FrozenBackboneMLP"
    requires_backbone_features = True

    def __init__(self, num_classes=2, epochs=30, lr=1e-3, batch_size=512):
        self.nc = num_classes
        self.epochs, self.lr, self.bs = epochs, lr, batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    def _tensors(self, data):
        return [torch.FloatTensor(data['backbone_features'])]

    def fit(self, data, y):
        feat_dim = data['backbone_features'].shape[1]
        self.model = _BackboneMLPNet(feat_dim, self.nc)
        _train_torch_model(
            self.model, self._tensors(data), torch.LongTensor(y),
            self.nc, self.device, self.epochs, self.lr, self.bs,
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


# =============================================================================
# 10. Image Stats + Frozen Backbone → GradientBoosting
# =============================================================================

class HybridStatsBackboneClassifier(BaseClassifier):
    """Concatenated image stats + frozen SqueezeNet features → GradientBoosting."""
    name = "HybridStatsBackbone"
    requires_image_stats = True
    requires_backbone_features = True

    def __init__(self, num_classes=2):
        self.nc = num_classes
        self.model = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8,
        )

    def _combine(self, data):
        return np.column_stack([data['image_stats'], data['backbone_features']])

    def fit(self, data, y):
        X = self._combine(data)
        counts = np.bincount(y, minlength=self.nc).astype(float)
        sw = len(y) / (self.nc * counts[y])
        self.model.fit(X, y, sample_weight=sw)

    def get_model_size_mb(self):
        import pickle
        if not hasattr(self.model, 'estimators_'): return 0.0
        return len(pickle.dumps(self.model)) / (1024 * 1024)

    def get_param_count(self):
        if not hasattr(self.model, 'estimators_'): return 0
        return sum(tree[0].tree_.node_count for tree in self.model.estimators_)

    def get_gflops(self):
        if not hasattr(self.model, 'estimators_'): return 0.0
        return (self.model.n_estimators * self.model.max_depth) / 1e9

    def predict(self, data):
        return self.model.predict(self._combine(data))

    def predict_proba(self, data):
        return self.model.predict_proba(self._combine(data))


# =============================================================================
# Base Class for Finetuned Models
# =============================================================================

class _BaseFinetunedClassifier(BaseClassifier):
    """Base class for end-to-end fine-tuning on raw images."""
    requires_images = True
    model_name_str = "base_finetuned"

    def __init__(self, num_classes=2, epochs=5, lr=1e-4, batch_size=64):
        self.nc = num_classes
        self.epochs, self.lr, self.bs = epochs, lr, batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    def _build_model(self):
        raise NotImplementedError

    def fit(self, data, y):
        if 'images' not in data:
            raise ValueError(f"Image tensors required for {self.model_name_str}.")
        self.model = self._build_model().to(self.device)
        
        # Optimize only trainable params
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
        )
        weights = _compute_class_weights(y, self.nc, self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        images_t = torch.FloatTensor(data['images'])
        labels_t = torch.LongTensor(y)
        dataset = TensorDataset(images_t, labels_t)
        loader = DataLoader(dataset, batch_size=self.bs, shuffle=True,
                            num_workers=0, pin_memory=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss, n_batches = 0, 0
            for imgs, targets in loader:
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                logits = self.model(imgs)
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}  loss={total_loss/n_batches:.4f}")

    def get_model_size_mb(self):
        if self.model is None: return 0.0
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        return param_size / (1024 * 1024)

    def get_param_count(self):
        if self.model is None: return 0
        return sum(p.numel() for p in self.model.parameters())

    def get_gflops(self):
        if self.model is None: return 0.0
        # Assuming 224x224 input
        return count_flops(self.model, input_size=(1, 3, 224, 224))

    @torch.no_grad()
    def predict(self, data):
        self.model.eval()
        images_t = torch.FloatTensor(data['images'])
        all_preds = []
        for i in range(0, len(images_t), self.bs):
            batch = images_t[i:i+self.bs].to(self.device)
            logits = self.model(batch)
            all_preds.append(logits.argmax(dim=1).cpu())
        return torch.cat(all_preds).numpy()

    @torch.no_grad()
    def predict_proba(self, data):
        self.model.eval()
        images_t = torch.FloatTensor(data['images'])
        all_probs = []
        for i in range(0, len(images_t), self.bs):
            batch = images_t[i:i+self.bs].to(self.device)
            logits = self.model(batch)
            all_probs.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(all_probs).numpy()


# =============================================================================
# 11. Finetuned ResNet-18
# =============================================================================

class FinetunedResNet18Classifier(_BaseFinetunedClassifier):
    name = "FinetunedResNet18"

    def _build_model(self):
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze early layers
        for param in base.conv1.parameters(): param.requires_grad = False
        for param in base.bn1.parameters(): param.requires_grad = False
        for param in base.layer1.parameters(): param.requires_grad = False
        
        backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool, nn.Flatten(),
        )
        head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, self.nc))
        return nn.Sequential(backbone, head)


# =============================================================================
# 12. Finetuned MobileNetV3-Small
# =============================================================================

class FinetunedMobileNetV3SmallClassifier(_BaseFinetunedClassifier):
    name = "FinetunedMobileNetV3Small"

    def _build_model(self):
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Freeze early layers logic is harder due to structure, just fine-tune classifier
        # or last few blocks. For now, freeze features[0:9]
        for param in base.features[:9].parameters():
            param.requires_grad = False
        
        # Replace classifier
        # Original: Sequential(Linear(576, 1024), Hardswish, Dropout, Linear(1024, 1000))
        # customized for our num_classes
        last_channel = 576
        base.classifier = nn.Sequential(
            nn.Linear(last_channel, 1024),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, self.nc)
        )
        return base


# =============================================================================
# 13. Finetuned ShuffleNetV2 x0.5
# =============================================================================

class FinetunedShuffleNetV2Classifier(_BaseFinetunedClassifier):
    name = "FinetunedShuffleNetV2"

    def _build_model(self):
        base = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        # Freeze conv1 and stage1
        for param in base.conv1.parameters(): param.requires_grad = False
        for param in base.stage2.parameters(): param.requires_grad = False # Named stage2 but is the first heavy block
        
        # fc is the classifier
        # in_features = 1024
        base.fc = nn.Linear(1024, self.nc)
        return base


# =============================================================================
# 14. Finetuned SqueezeNet 1.1
# =============================================================================

class FinetunedSqueezeNetClassifier(_BaseFinetunedClassifier):
    name = "FinetunedSqueezeNet1_1"

    def _build_model(self):
        base = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        # Freeze features[:8]
        for param in base.features[:8].parameters():
            param.requires_grad = False
            
        base.classifier[1] = nn.Conv2d(512, self.nc, kernel_size=(1,1), stride=(1,1))
        # Squeezenet classifier is (Dropout, Conv2d, ReLU, AvgPool)
        # We replace the Conv2d.
        return base


# =============================================================================
# 15. Finetuned MobileNetV2
# =============================================================================

class FinetunedMobileNetV2Classifier(_BaseFinetunedClassifier):
    name = "FinetunedMobileNetV2"

    def _build_model(self):
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Freeze features[:14]
        for param in base.features[:14].parameters():
            param.requires_grad = False
        
        # classifier is Sequential(Dropout, Linear)
        base.classifier[1] = nn.Linear(1280, self.nc)
        return base


# =============================================================================
# 16. SimpleCNN (Custom 3-layer)
# =============================================================================

class SimpleCNNNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 112
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 56
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 64),
            nn.ReLU(), 
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SimpleCNNClassifier(_BaseFinetunedClassifier):
    name = "SimpleCNN"

    def _build_model(self):
        return SimpleCNNNet(self.nc)


# =============================================================================
# 19. Finetuned EfficientNet-B0
# =============================================================================

class FinetunedEfficientNetB0Classifier(_BaseFinetunedClassifier):
    name = "FinetunedEfficientNetB0"

    def _build_model(self):
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Freeze features[:4] (first few blocks)
        # EfficientNet features are in base.features
        for param in base.features[:4].parameters():
            param.requires_grad = False
        
        # Classifier is in base.classifier: Sequential(Dropout, Linear)
        # In EfficientNet B0, last channel is 1280
        base.classifier[1] = nn.Linear(1280, self.nc)
        return base
