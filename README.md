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

### 3. Verify Setup & Usage
Run the included example script to understand how to interact with the dataset wrappers and verify your environment.
```bash
python src/test/example_usage.py
```

## Features

- **Batch Inference**: Efficiently run pytorch models on standard datasets.
- **Streaming Data**: Support for streaming huge datasets (e.g., ImageNet-1k) from Hugging Face without downloading.
- **Label Mapping**: Auto-map predictions between different label spaces (e.g., ImageNet -> CIFAR10).

## Project Structure

```text
├── configs/            # Hydra configs
├── src/
│   ├── datasets/       # Dataset wrappers (standard & streaming)
│   ├── models/         # Model definitions
│   ├── inference/      # Inference engine
│   ├── test/           # Tests and verification scripts
├── main.py             # Entry point
└── requirements.txt
```

## Testing & Verification

The project includes a `src/test` directory with unit tests and examples.

**Run Unit Tests:**
```bash
python src/test/test_wrappers.py
```
*Tests all wrappers including streaming and reliability edge cases.*

**Run Usage Example:**
```bash
python src/test/example_usage.py
```
*Demonstrates how to initialize wrappers (both local and streaming) and access data.*

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
- `prediction`: The class index predicted by the model (mapped to target dataset space if applicable).
- `confidence`: The probability/confidence score of the prediction.
- `ground_truth`: The actual class label.

## Label Mapping

To handle label discrepancies (e.g., ImageNet model on CIFAR10), the system automatically maps equivalent classes using JSON files in `configs/mappings/`.

- **Model Config**: Models define their `source_dataset` (e.g., `imagenet1k`).
- **Logic**: If `source != target`, predictions are mapped. Unmapped classes become `-1`.
- **Supported Mappings**: ImageNet1k -> Imagenette, CIFAR10 (208 classes), CIFAR100 (131 classes).
