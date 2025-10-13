import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import os
from src.datamodules.dataset.MAPPODataset import MAPPODataset, custom_collate_fn


class MAPPODatamodule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, test_batch_size, num_workers, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: str = None):
        # `setup` is called on every GPU in DDP, so load data here once
        pre_data_path = os.path.join(self.data_dir, 'pre_data_with_index.pkl')
        if not os.path.exists(pre_data_path):
            raise FileNotFoundError(f"Data file not found at {pre_data_path}")
            
        pre_data = torch.load(pre_data_path, weights_only=False)
        train_data, test_data = pre_data[0], pre_data[1]
        
        
        # Create a validation set from a subset of training data
        val_data = [d[:len(test_data[0])] for d in train_data]

        if stage == 'fit' or stage is None:
            self.train_dataset = MAPPODataset(train_data)
            self.val_dataset = MAPPODataset(val_data)
        if stage == 'test' or stage is None:
            self.test_dataset = MAPPODataset(test_data)

    def train_dataloader(self):
        import random
        import numpy as np
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0,  
            pin_memory=True, 
            drop_last=True,
            collate_fn=custom_collate_fn,  
            generator=torch.Generator().manual_seed(self.seed)
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)
