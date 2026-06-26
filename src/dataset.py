import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List

class SleepSequenceDataset(Dataset):
    def __init__(self, X_list: List[np.ndarray], y_list: List[np.ndarray], 
                 sequence_length: int = 20, stride: int = 20, is_train: bool = True):
        """
        Dataset for Sleep Epoch Sequences.
        
        Args:
            X_list (List[np.ndarray]): List of numpy arrays, each of shape (N_i, 3, 3000) for recording i.
            y_list (List[np.ndarray]): List of numpy arrays, each of shape (N_i,) for recording i.
            sequence_length (int): Length of the sequence (default 20).
            stride (int): Stride for extracting sequences. 20 for non-overlapping (train), 1 for overlapping (inference).
            is_train (bool): Whether to apply N1 augmentation.
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.is_train = is_train
        
        self.sequences_x = []
        self.sequences_y = []
        
        # Build sequences per recording
        for X, y in zip(X_list, y_list):
            num_epochs = len(y)
            if num_epochs < sequence_length:
                continue
                
            for start_idx in range(0, num_epochs - sequence_length + 1, stride):
                end_idx = start_idx + sequence_length
                seq_x = X[start_idx:end_idx] # (20, 3, 3000)
                seq_y = y[start_idx:end_idx] # (20,)
                
                self.sequences_x.append(seq_x)
                self.sequences_y.append(seq_y)
                
    def __len__(self):
        return len(self.sequences_y)
        
    def __getitem__(self, idx):
        # Shape: (20, 3, 3000)
        x = np.copy(self.sequences_x[idx])
        y = self.sequences_y[idx]
        
        # Apply N1 augmentation if training
        if self.is_train:
            # 1 corresponds to N1 sleep stage
            n1_indices = np.where(y == 1)[0]
            for i in n1_indices:
                # Additive Gaussian noise strictly on N1
                # Standard Deviation = 5% of the epoch's RMS amplitude
                epoch_data = x[i] # (3, 3000)
                # RMS over channels and time
                rms = np.sqrt(np.mean(epoch_data**2))
                noise_std = 0.05 * rms
                noise = np.random.normal(0, noise_std, size=epoch_data.shape)
                x[i] = epoch_data + noise
                
        # Convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        return x_tensor, y_tensor

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
