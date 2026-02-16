# Project: Model Disagreement Analysis System

## 1. Background
To systematically evaluate performance discrepancies between different pre-trained classification models. By generating and comparing pseudo-labels against ground truth, the system identifies specific knowledge gaps (e.g., Model A correct vs. Model B incorrect) to inform intelligent model selection for downstream tasks.

**Project Goals:**

* **Inference:** Generate model-specific pseudo-labels for the ImageNet dataset.
* **Downstream Integration:** Provide a ready-to-use Python class compatible with existing ImageNet pipelines to facilitate seamless integration into downstream tasks.

## 2. Core Features
-   **Task Scope**: Image Classification only.
-   **Dataset Abstraction**: Wrappers around existing libraries (e.g., Torchvision, Hugging Face) to provide a unified interface (Image + Metadata).
-   **Config-Driven**: Use Hydra/YAML for reproducible experiment configurations (Model A vs. B on Dataset X).
-   **Batch Inference**: High-throughput GPU prediction pipeline with progress tracking.
-   **Incremental Output**: model-dataset-wise prediction results saved to CSV for easy inspection. this provides a suplementary model-specific predicton on dataset images.

## 3. Data Integrity & Output Schema
To ensure precise matching of inputs to model outputs, the system uses a strict CSV schema.
-   **Format**: CSV
-   **Schema**:
    -   `sample_id`: Unique identifier (e.g., filename or hash) to link back to the source image.
    -   `model_id`: Identifier for the model version/config.
    -   `prediction`: Predicted class label/index.
    -   `confidence`: Top-1 score.
    -   `logits`: (Optional) Stringified logits for deeper semantic analysis.
    -   `ground_truth`: (Optional) True label if available.

## 4. Architecture
The project follows a modular design to decouple data, model, and logic.

### Structure
```text
labeling_project/
├── configs/            # Hydra/YAML configs for Models, Datasets, and Experiments
├── src/
│   ├── datasets/       # Wrappers for standardizing external datasets, can be imported by other projects
│   ├── models/         # Model loaders and adapters, used for inferencing pseudo-labels
│   ├── inference/      # Batch inference engine (Dataset -> Model -> CSV) for speeding up the inference process
│   ├── test/           # Test scripts for verifying the pipeline and exported wrappers. and examples showcasing how to use
├── main.py             # CLI entry point
└── requirements.txt
```

### Technology Stack
-   **Core**: Python 3.8+, PyTorch, Torchvision
-   **Config**: Hydra or Omegaconf
-   **Data**: Pandas (for CSV handling), PIL
-   **Utils**: tqdm

### Environment

- Conda environment: `proxy-det`