# Project: Model Disagreement Analysis System

## 1. Goal
To systematically evaluate performance discrepancies between different pre-trained classification models. By generating and comparing pseudo-labels against ground truth, the system identifies specific knowledge gaps (e.g., Model A correct vs. Model B incorrect) to inform intelligent model selection for downstream tasks.

## 2. Core Features
-   **Task Scope**: Image Classification only.
-   **Dataset Abstraction**: Wrappers around existing libraries (e.g., Torchvision, Hugging Face) to provide a unified interface (Image + Metadata).
-   **Config-Driven**: Use Hydra/YAML for reproducible experiment configurations (Model A vs. B on Dataset X).
-   **Batch Inference**: High-throughput GPU prediction pipeline with progress tracking.
-   **Incremental Output**: Results saved to CSV for easy inspection.
-   **Analysis Playground**: Separate tools to consume inference outputs for statistical analysis and visualization.

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
│   ├── datasets/       # Wrappers for standardizing external datasets
│   ├── models/         # Model loaders and adapters
│   ├── inference/      # Batch inference engine (Dataset -> Model -> CSV)
│   └── analysis/       # Post-hoc analysis scripts (CSV -> Report)
├── notebooks/          # Jupyter notebooks for interactive exploration
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