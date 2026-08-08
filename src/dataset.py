import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List

class SleepSequenceDataset(Dataset):
    def __init__(self, X_list: List[np.ndarray], y_list: List[np.ndarray], 
                 sequence_length: int = 20, stride: int = 20, is_train: bool = True):
        self.sequence_length = sequence_length
        self.stride = stride
        self.is_train = is_train
        
        # Convert original data to shared PyTorch tensors in-place to prevent multiprocessing leaks
        # We explicitly delete the numpy arrays from the original lists to free RAM instantly
        self.X_list = []
        self.y_list = []
        for i in range(len(X_list)):
            self.X_list.append(torch.tensor(X_list[i], dtype=torch.float32))
            self.y_list.append(torch.tensor(y_list[i], dtype=torch.long))
            X_list[i] = None # Free RAM!
            y_list[i] = None # Free RAM!
        
        # Only store the indices of the sequences
        self.valid_indices = []
        for rec_idx, y in enumerate(self.y_list):
            num_epochs = len(y)
            if num_epochs < sequence_length:
                continue
            for start_idx in range(0, num_epochs - sequence_length + 1, stride):
                self.valid_indices.append((rec_idx, start_idx))
                
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        rec_idx, start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Dynamically slice the sequence (clone so we don't mutate shared memory during aug)
        x = self.X_list[rec_idx][start_idx:end_idx].clone()
        y = self.y_list[rec_idx][start_idx:end_idx]
        
        # Apply N1 augmentation if training
        if self.is_train:
            n1_indices = torch.where(y == 1)[0]
            for i in n1_indices:
                epoch_data = x[i]
                # Amplitude Scaling (±10%)
                # This preserves real brainwave structures while acting as augmentation
                scale_factor = torch.empty(1).uniform_(0.9, 1.1).item()
                x[i] = epoch_data * scale_factor
                
        return x, y

def compute_class_weights(y_list: List[np.ndarray], num_classes: int = 5) -> torch.Tensor:
    """
    Computes class weights using smoothed inverse frequency: w_c = sqrt(N / N_c)
    Normalized so they sum to num_classes.
    """
    all_y = np.concatenate(y_list)
    total_epochs = len(all_y)
    
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        n_c = np.sum(all_y == c)
        if n_c > 0:
            weights[c] = np.sqrt(total_epochs / n_c)
        else:
            weights[c] = 0.0 # Or 1.0 depending on edge case handling
            
    # Normalize weights so they sum to num_classes
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = (weights / weight_sum) * num_classes
        
    return torch.tensor(weights, dtype=torch.float32)
