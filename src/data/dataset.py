import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class EEGDataset(Dataset):
    """PyTorch Dataset for preprocessed EEG trials."""
    
    def __init__(self, data_dir, subject_ids=None):
        self.data_dir = Path(data_dir)
        self.trials, self.labels = [], []
        
        data_files = list(self.data_dir.glob("*_preprocessed.npz"))
        if not data_files:
            raise FileNotFoundError(f"No preprocessed data in {data_dir}")
        
        for file_path in data_files:
            subj_id = file_path.stem.replace("_preprocessed", "")
            if subject_ids and subj_id not in subject_ids:
                continue
            
            data = np.load(file_path)
            self.trials.append(data['X'])
            self.labels.append(data['y'])
        
        self.trials = torch.FloatTensor(np.concatenate(self.trials, axis=0))
        self.labels = torch.LongTensor(np.concatenate(self.labels, axis=0))
        
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        return self.trials[idx], self.labels[idx]
