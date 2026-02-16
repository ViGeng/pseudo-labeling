import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from pseudo_labeling.datasets.wrappers import (  # New wrappers for classification prototyping experiments
    CIFAR10Wrapper, CIFAR100CWrapper, CIFAR100Wrapper, CUB200Wrapper,
    DomainNetWrapper, FashionMNISTWrapper, ImageNetCWrapper, ImagenetteWrapper,
    ImageNetWrapper, OfficeHomeWrapper, StanfordCarsWrapper, StreamingImageNetWrapper,
    StylizedImageNetWrapper, TinyImageNetCWrapper, TinyImageNetWrapper)
from pseudo_labeling.inference.runner import InferenceRunner
from pseudo_labeling.models.loader import load_model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Ensure pairs is a list
    if not hasattr(cfg, "pairs") or not cfg.pairs:
        print("No pairs defined in config. Exiting.")
        return

    pairs = cfg.pairs
    
    # --- Print Pair Plan ---
    print("\n" + "="*40)
    print("Execution Pair Plan")
    print("="*40)
    for i, pair in enumerate(pairs):
        print(f"{i+1}. Dataset: {pair.dataset} | Model: {pair.model} | Split: {pair.get('split', 'val')}")
    print(f"Total Runs: {len(pairs)}")
    print("="*40 + "\n")
    
    # --- Pair Loop ---
    for pair in pairs:
        ds_name = pair.dataset
        model_name = pair.model
        split = pair.get("split", "val")

        # 1. Setup Dataset
        print(f"Loading Dataset: {ds_name} | Split: {split}")
        
        # Load dataset-specific config
        # Use package-relative path for configs
        base_config_path = os.path.join(os.path.dirname(__file__), "configs")
        ds_cfg_path = os.path.join(base_config_path, "dataset", f"{ds_name}.yaml")
        if os.path.exists(ds_cfg_path):
            ds_cfg = OmegaConf.load(ds_cfg_path)
            download = ds_cfg.get("download", False)
            ds_root = ds_cfg.get("root", cfg.dataset_root)
            print(f"Loaded config for {ds_name} (download={download})")
        else:
            print(f"[WARNING] No config found for dataset {ds_name}, using defaults.")
            download = False
            ds_root = cfg.dataset_root

        if ds_name == "cifar10":
            dataset = CIFAR10Wrapper(root=ds_root, split=split, download=download)
        elif ds_name == "cifar100":
            dataset = CIFAR100Wrapper(root=ds_root, split=split, download=download)
        elif ds_name == "fashion_mnist":
            dataset = FashionMNISTWrapper(root=ds_root, split=split, download=download)
        elif ds_name == "imagenette":
            dataset = ImagenetteWrapper(root=ds_root, split=split, download=download)
        elif ds_name == "imagenet":
            dataset = ImageNetWrapper(root=ds_root, split=split, download=download)
        # --- New datasets for classification prototyping ---
        elif ds_name == "imagenet_c":
            corruption_type = ds_cfg.get("corruption_type", "gaussian_noise")
            severity = ds_cfg.get("severity", 3)
            dataset = ImageNetCWrapper(root=ds_root, split=split, download=download,
                                        corruption_type=corruption_type, severity=severity)
        elif ds_name == "stylized_imagenet":
            dataset = StylizedImageNetWrapper(root=ds_root, split=split, download=download)
        elif ds_name == "office_home":
            domain = ds_cfg.get("domain", "Real_World")
            dataset = OfficeHomeWrapper(root=ds_root, split=split, download=download, domain=domain)
        elif ds_name == "domainnet":
            domain = ds_cfg.get("domain", "real")
            dataset = DomainNetWrapper(root=ds_root, split=split, download=download, domain=domain)
        elif ds_name == "stanford_cars":
            dataset = StanfordCarsWrapper(root=ds_root, split=split, download=download)
        elif ds_name == "cub200":
            dataset = CUB200Wrapper(root=ds_root, split=split, download=download)
        elif ds_name == "cifar100_c":
            corruption_type = ds_cfg.get("corruption_type", "gaussian_noise")
            severity = ds_cfg.get("severity", 3)
            dataset = CIFAR100CWrapper(root=ds_root, split=split, download=download,
                                        corruption_type=corruption_type, severity=severity)
        elif ds_name == "tiny_imagenet":
            dataset = TinyImageNetWrapper(root=ds_root, split=split, download=download)
        elif ds_name == "tiny_imagenet_c":
            corruption_type = ds_cfg.get("corruption_type", "gaussian_noise")
            severity = ds_cfg.get("severity", 3)
            dataset = TinyImageNetCWrapper(root=ds_root, split=split, download=download,
                                            corruption_type=corruption_type, severity=severity)
        elif ds_name == "imagenet_streaming":
            dataset = StreamingImageNetWrapper(split=split, streaming=True)
        else:
            raise ValueError(f"Unknown dataset: {ds_name}")
        
        # --- Subsetting ---
        if cfg.subset is not None and cfg.subset > 0:
            print(f"Creating subset of size {cfg.subset}")
            if isinstance(dataset, torch.utils.data.IterableDataset):
                # For streaming datasets, we can't use Subset or len()
                # We use itertools.islice in the dataloader or WRAP the dataset
                # But PyTorch DataLoader with IterableDataset doesn't support 'subset' easily directly.
                # A simple way is to wrap it.
                import itertools
                
                class IsliceWrapper(torch.utils.data.IterableDataset):
                    def __init__(self, dataset, n):
                        self.dataset = dataset
                        self.n = n
                    def __iter__(self):
                        return itertools.islice(self.dataset, self.n)
                        
                dataset = IsliceWrapper(dataset, cfg.subset)
                
                dataset = IsliceWrapper(dataset, cfg.subset)
            else:
                indices = list(range(min(len(dataset), cfg.subset)))
                dataset = torch.utils.data.Subset(dataset, indices)
        
        # Use num_workers=0 for IterableDataset to avoid duplication unless proper sharding is implemented
        # (HF Datasets streaming + PyTorch DataLoader with num_workers > 0 requires manual sharding)
        loader_num_workers = 0 if isinstance(dataset, torch.utils.data.IterableDataset) else cfg.num_workers
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=loader_num_workers, shuffle=False)
        
        # Construct output path specific to dataset and split
        output_dir = cfg.output_dir
        
        # Construct output path using [dataset_name]-[model name]-[split].csv
        output_csv_path = f"{output_dir}/{ds_name}-{model_name}-{split}.csv"

        if os.path.exists(output_csv_path):
            print(f"[SKIP] Output file already exists: {output_csv_path}")
            continue

        print(f"Loading Model: {model_name}")
        
        # 2. Setup Model
        # Load model-specific config to get weights and source dataset
        # Try to find a config file for the model
        # Simple mapping for known deviation
        fname = model_name
        if model_name == "mobilenet_v3_large":
            fname = "mobilenet_v3"
            
        model_cfg_path = os.path.join(base_config_path, "model", f"{fname}.yaml")
        
        if os.path.exists(model_cfg_path):
            model_cfg = OmegaConf.load(model_cfg_path)
            weights = model_cfg.get("weights", "DEFAULT")
            model_source = model_cfg.get("source_dataset", "imagenet1k")
            model_library = model_cfg.get("source", "torchvision")  # torchvision, timm, or open_clip
            print(f"Loaded config for {model_name} from {fname}.yaml (source: {model_library})")
        else:
            print(f"[WARNING] No config found for {model_name}, using defaults.")
            weights = "DEFAULT"
            model_source = "imagenet1k"
            model_library = "torchvision"

        model = load_model(name=model_name, weights_enum=weights, source=model_library)
        
        # --- Label Mapping Setup ---
        target_dataset = ds_name
        
        label_mapper = None
        if model_source != target_dataset:
            # Construct potential mapping file path: configs/mappings/{source}_to_{target}.json
            mapping_filename = f"{model_source}_to_{target_dataset}.json"
            mapping_path = os.path.join(base_config_path, "mappings", mapping_filename)
            
            if os.path.exists(mapping_path):
                print(f"Loading label mapping: {mapping_filename}")
                from pseudo_labeling.utils.label_mapper import LabelMapper
                label_mapper = LabelMapper(mapping_path)
            else:
                print(f"[WARNING] No mapping found for {model_source} -> {target_dataset}. Predictions will remain in {model_source} label space.")

        # 3. Setup Runner
        runner = InferenceRunner(
            model=model,
            dataloader=dataloader,
            device=cfg.device,
            output_csv=output_csv_path,
            model_name=model_name,
            save_logits=cfg.save_logits,
            label_mapper=label_mapper
        )
        
        # 4. Run
        runner.run()

if __name__ == "__main__":
    main()
