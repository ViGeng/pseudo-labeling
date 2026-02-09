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
            output_dir = cfg.output_dir

            for model_name in model_names:
                # Construct output path using [dataset_name]-[model name]-[split].csv
                output_csv_path = f"{output_dir}/{ds_name}-{model_name}-{split}.csv"

                if os.path.exists(output_csv_path):
                    print(f"[SKIP] Output file already exists: {output_csv_path}")
                    continue

                print(f"Loading Model: {model_name}")
                
                # 2. Setup Model
                # Load model-specific config to get weights and source dataset
                # Try to find a config file for the model
                model_cfg_path = os.path.join(hydra.utils.get_original_cwd(), "configs", "model", f"{model_name}.yaml")
                # Handle case where model name in loop might not perfectly match file, 
                # but here we assume strict naming or existing mapping.
                # For 'mobilenet_v3_large', the file is 'mobilenet_v3.yaml' based on previous ls.
                # We need a fallback or a way to map names.
                # The user's config loop uses "mobilenet_v3_large". 
                # I see 'mobilenet_v3.yaml' has 'name: mobilenet_v3_large' inside it.
                # Let's try to map or just try standard name.
                
                # Simple mapping for known deviation
                fname = model_name
                if model_name == "mobilenet_v3_large":
                    fname = "mobilenet_v3"
                    
                model_cfg_path = os.path.join(hydra.utils.get_original_cwd(), "configs", "model", f"{fname}.yaml")
                
                if os.path.exists(model_cfg_path):
                    model_cfg = OmegaConf.load(model_cfg_path)
                    weights = model_cfg.get("weights", "DEFAULT")
                    model_source = model_cfg.get("source_dataset", "imagenet1k")
                    print(f"Loaded config for {model_name} from {fname}.yaml")
                else:
                    print(f"[WARNING] No config found for {model_name}, using defaults.")
                    weights = "DEFAULT"
                    model_source = "imagenet1k"

                model = load_model(name=model_name, weights_enum=weights)
                
                # --- Label Mapping Setup ---
                target_dataset = ds_name
                target_dataset = ds_name
                
                label_mapper = None
                if model_source != target_dataset:
                    # Construct potential mapping file path: configs/mappings/{source}_to_{target}.json
                    mapping_filename = f"{model_source}_to_{target_dataset}.json"
                    mapping_path = os.path.join(hydra.utils.get_original_cwd(), "configs", "mappings", mapping_filename)
                    
                    if os.path.exists(mapping_path):
                        print(f"Loading label mapping: {mapping_filename}")
                        from src.utils.label_mapper import LabelMapper
                        label_mapper = LabelMapper(mapping_path)
                    else:
                        print(f"[WARNING] No mapping found for {model_source} -> {target_dataset}. Predictions will remain in {model_source} label space.")

                # 3. Setup Runner
                # Note: 'a' mode (append) in runner allows multiple models to write to same file
                runner = InferenceRunner(
                    model=model,
                    dataloader=dataloader,
                    device=cfg.device,
                    output_csv=output_csv_path,
                    model_name=model_name,
                    save_logits=cfg.analysis.save_logits,
                    label_mapper=label_mapper
                )
                
                # 4. Run
                runner.run()

if __name__ == "__main__":
    main()
