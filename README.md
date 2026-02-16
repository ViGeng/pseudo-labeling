# Pseudo-Labeling Inference Pipeline

A lightweight inference pipeline for running various models on datasets like CIFAR-10, ImageNet, etc., configured via Hydra. Now available as a installable Python package.

## Quick Start

### 1. Installation

You can install the package directly from the source code:

```bash
# Install in editable mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```

(Optional) Install development dependencies:
```bash
pip install -e .[dev]
```

### 2. CLI Usage

The package provides a `pseudo-label` command line interface.

**Basic Usage:**
 Runs default configuration (ImageNet Streaming + MobileNetV3/SwinT on validation split).
```bash
pseudo-label
```

**Custom Configuration:**
You can override any configuration parameter using Hydra syntax.

*Run specific dataset/model pair:*
```bash
pseudo-label "pairs=[{dataset:cifar10,model:resnet18,split:val}]"
```

*Change batch size or device:*
```bash
pseudo-label batch_size=32 device=cpu
```

*Run multiple pairs:*
```bash
pseudo-label "pairs=[{dataset:cifar10,model:resnet18},{dataset:cifar100,model:efficientnet_b0}]"
```

### 3. Library Usage

You can use the dataset wrappers and models in your own Python scripts:

```python
from pseudo_labeling.datasets.wrappers import ImageNetWrapper, StreamingImageNetWrapper
from pseudo_labeling.models.loader import load_model

# Initialize wrapper
dataset = StreamingImageNetWrapper(split='val', streaming=True)

# Load model
model_adapter = load_model(name='resnet18', weights_enum='DEFAULT')

# Access data
sample_id, img, target = next(iter(dataset))
print(f"Sample: {sample_id}, Label: {target}")
```

### 4. Verify Setup
Run the included example script to verify valid environment and data access:
```bash
python tests/example_usage.py
```

## Features

- **Batch Inference**: Efficiently run pytorch models on standard datasets.
- **Streaming Data**: Support for streaming huge datasets (e.g., ImageNet-1k) from Hugging Face without downloading.
- **Label Mapping**: Auto-map predictions between different label spaces (e.g., ImageNet -> CIFAR10).
- **Extensible**: Easily add new datasets or models via the `pseudo_labeling` package.

## Project Structure

```text
├── pyproject.toml      # Package configuration
├── src/
│   └── pseudo_labeling/ # Main package
│       ├── configs/    # Hydra configs (bundled)
│       ├── datasets/   # Dataset wrappers
│       ├── models/     # Model factory
│       ├── inference/  # Inference runner
│       ├── utils/      # Utilities
│       └── cli.py      # Entry point
└── tests/              # Verification scripts
```

## Output

Results are saved to the `outputs/` directory by default (configurable via `output_dir`).

**Filename Format:**
`[dataset_name]-[model_name]-[split].csv`

**CSV Structure:**
```csv
sample_id,prediction,confidence,ground_truth
cifar10_test_0,776,0.2668,3
...
```
- `sample_id`: Unique identifier for the sample.
- `prediction`: The class index predicted by the model (mapped to target dataset space if applicable).
- `confidence`: The probability/confidence score of the prediction.
- `ground_truth`: The actual class label.

## Label Mapping

To handle label discrepancies (e.g., ImageNet model on CIFAR10), the system automatically maps equivalent classes using JSON files in `configs/mappings/`.

- **Model Config**: Models define their `source_dataset` (e.g., `imagenet1k`).
- **Logic**: If `source != target`, predictions are mapped. Unmapped classes become `-1`.
- **Supported Mappings**: ImageNet1k -> Imagenette, CIFAR10 (208 classes), CIFAR100 (131 classes).
