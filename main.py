import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

import pandas as pd
import os
from src.datasets.wrappers import CIFAR10Wrapper, CIFAR100Wrapper, FashionMNISTWrapper
from src.models.loader import load_model
from src.inference.runner import InferenceRunner

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Parse lists from config (handle comma-separated strings)
    # Datasets
    if "," in cfg.dataset.name:
        dataset_names = [x.strip() for x in cfg.dataset.name.split(",")]
    else:
        dataset_names = [cfg.dataset.name]

    # Models
    if "," in cfg.model.name:
        model_names = [x.strip() for x in cfg.model.name.split(",")]
    else:
        model_names = [cfg.model.name]
        
    # Splits
    # Handle list or string (if it's a list in yaml, Hydra handles it, but user requested comma string support)
    if isinstance(cfg.dataset.split, str) and "," in cfg.dataset.split:
        splits = [x.strip() for x in cfg.dataset.split.split(",")]
    else:
        splits = [cfg.dataset.split]
    
    # --- Print Matrix Plan ---
    print("\n" + "="*40)
    print("Execution Matrix Plan")
    print("="*40)
    print(f"Datasets ({len(dataset_names)}): {', '.join(dataset_names)}")
    print(f"Splits ({len(splits)}):   {', '.join(splits)}")
    print(f"Models ({len(model_names)}):   {', '.join(model_names)}")
    print(f"Total Runs: {len(dataset_names) * len(splits) * len(model_names)}")
    print("="*40 + "\n")
    
    # --- Matrix Loop ---
    for ds_name in dataset_names:
        for split in splits:
            # 1. Setup Dataset
            print(f"Loading Dataset: {ds_name} | Split: {split}")
            if ds_name == "cifar10":
                dataset = CIFAR10Wrapper(root=cfg.dataset.root, split=split, download=cfg.dataset.download)
            elif ds_name == "cifar100":
                dataset = CIFAR100Wrapper(root=cfg.dataset.root, split=split, download=cfg.dataset.download)
            elif ds_name == "fashion_mnist":
                dataset = FashionMNISTWrapper(root=cfg.dataset.root, split=split, download=cfg.dataset.download)
            elif ds_name == "imagenette":
                from src.datasets.wrappers import ImagenetteWrapper
                dataset = ImagenetteWrapper(root=cfg.dataset.root, split=split, download=cfg.dataset.download)
            elif ds_name == "imagenet":
                from src.datasets.wrappers import ImageNetWrapper
                dataset = ImageNetWrapper(root=cfg.dataset.root, split=split, download=cfg.dataset.download)
            else:
                raise ValueError(f"Unknown dataset: {ds_name}")
            
            # --- Subsetting ---
            if cfg.dataset.subset is not None and cfg.dataset.subset > 0:
                print(f"Creating subset of size {cfg.dataset.subset}")
                indices = list(range(min(len(dataset), cfg.dataset.subset)))
                dataset = torch.utils.data.Subset(dataset, indices)
            
            dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)
            
            # Construct output path specific to dataset and split
            # e.g. "outputs/cifar10_val.csv"
            # We use the config-defined 'output_csv' as a template if possible, or force our scheme.
            # Using the schema requested: "outputs/dataset_name.csv", but user asked for "cifar10_val.csv"
            # let's construct it manually to be safe and support the feature.
            output_dir = cfg.output_dir
            output_csv_path = f"{output_dir}/{ds_name}_{split}.csv"

            for model_name in model_names:
                # Check for skip
                if os.path.exists(output_csv_path):
                    try:
                        # Only read specific columns to check model_id presence quickly
                        existing_df = pd.read_csv(output_csv_path, usecols=['model_id'])
                        if model_name in existing_df['model_id'].unique():
                            print(f"[SKIP] {model_name} already exists in {output_csv_path}")
                            continue
                    except Exception as e:
                        print(f"[WARN] Failed to check for existing results: {e}")

                print(f"Loading Model: {model_name}")
                
                # 2. Setup Model
                # We assume shared weights param (e.g. DEFAULT)
                model = load_model(name=model_name, weights_enum=cfg.model.weights)
                
                # 3. Setup Runner
                # Note: 'a' mode (append) in runner allows multiple models to write to same file
                runner = InferenceRunner(
                    model=model,
                    dataloader=dataloader,
                    device=cfg.device,
                    output_csv=output_csv_path,
                    model_name=model_name,
                    save_logits=cfg.analysis.save_logits
                )
                
                # 4. Run
                runner.run()

if __name__ == "__main__":
    main()
