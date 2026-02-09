import torch
import pandas as pd
from typing import Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json

from ..models.adapters import ClassificationAdapter

class InferenceRunner:
    def __init__(self, model: ClassificationAdapter, dataloader: DataLoader, device: str, output_csv: str, model_name: str, save_logits: bool = False):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.output_csv = output_csv
        self.model_name = model_name
        self.save_logits = save_logits
        
        self.model.to(self.device)

    def run(self):
        # Create directory for output if not exists
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        
        # Check if file exists to determine if we need header
        file_exists = os.path.isfile(self.output_csv)
        
        print(f"Starting inference... Output to {self.output_csv}")
        
        with open(self.output_csv, 'a') as f:
            if not file_exists:
                # Write header
                header = "sample_id,prediction,confidence"
                if self.save_logits:
                    header += ",logits"
                header += ",ground_truth\n"
                f.write(header)

            for batch in tqdm(self.dataloader, desc="Inferring"):
                sample_ids, images, targets = batch
                images = images.to(self.device)
                
                outputs = self.model(images)
                
                # Move to CPU for saving
                logits = outputs['logits'].cpu().numpy()
                preds = outputs['predictions'].cpu().numpy()
                confs = outputs['confidences'].cpu().numpy()
                targets = targets.numpy()
                
                # Iterate through batch and write rows
                for i in range(len(sample_ids)):
                    row = f"{sample_ids[i]}," \
                          f"{preds[i]}," \
                          f"{confs[i]:.4f},"

                    if self.save_logits:
                        logit_str = json.dumps(logits[i].tolist())
                        row += f"\"{logit_str}\","

                    row += f"{targets[i]}\n"
                    f.write(row)
                    
        print("Inference complete.")
