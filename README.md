# Pseudo-Labeling Inference Pipeline

A lightweight inference pipeline for running various models on datasets like CIFAR-10, ImageNet, etc., configured via Hydra.

## Quick Start

### 1. Setup Environment
Ensure your environment is set up with the required dependencies.
```bash
pip install -r requirements.txt
# Or use the existing conda env
conda activate proxy-det
```

### 2. Run Inference
Run the main script. By default, it runs `resnet18` on `cifar10` (or whatever defaults are in `configs/config.yaml`).
```bash
python main.py
```

## Output

Results are saved to the `outputs/` directory by default.

**Filename Format:**
`[dataset_name]-[model_name]-[split].csv`

**CSV Structure:**
```csv
sample_id,prediction,confidence,ground_truth
cifar10_test_0,776,0.2668,3
...
```
- `sample_id`: Unique identifier for the sample.
- `prediction`: The class index predicted by the model.
- `confidence`: The probability/confidence score of the prediction.
- `ground_truth`: The actual class label.
