import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, classification_report
from tqdm import tqdm

from dataset import SleepSequenceDataset
from model import NeuroSleepModel
from train import load_real_data, evaluate

def main():
    parser = argparse.ArgumentParser(description="NeuroSleep Evaluation Pipeline")
    parser.add_argument('--model-path', type=str, default="best_model.pth", help="Path to trained model weights")
    parser.add_argument('--cache-dir', type=str, default="./data_cache", help="Directory with cached data")
    args = parser.parse_args()
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Test subject based on train.py logic (Subject 40 since 39 is missing)
    test_subjects = [40]
    print(f"Loading test data for subject(s) {test_subjects}...")
    
    X_test, y_test = load_real_data(test_subjects, cache_dir=args.cache_dir)
    
    if len(X_test) == 0:
        raise ValueError("No test data loaded. Check cache directory or dataset.")
        
    # Setup test dataset and loader with stride=1 for inference
    test_dataset = SleepSequenceDataset(X_test, y_test, sequence_length=30, stride=1, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("Loading model...")
    model = NeuroSleepModel().to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded weights from {args.model_path}")
    else:
        print(f"Warning: Model path {args.model_path} not found. Using untrained model.")
        
    # Dummy criterion for evaluate function
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting evaluation...")
    _, macro_f1, kappa, acc, targets, preds = evaluate(model, test_loader, criterion, device)
    
    print("\n--- Evaluation Results ---")
    print(f"Overall Accuracy:   {acc:.4f}")
    print(f"Macro-average F1:   {macro_f1:.4f}")
    print(f"Cohen's Kappa:      {kappa:.4f}")
    
    print("\nPer-class Metrics:")
    print(classification_report(targets, preds, target_names=['Wake', 'N1', 'N2', 'N3', 'REM'], zero_division=0))
    
    # Target from context document:
    # 1. Macro-average F1 Score (Target: 0.77 - 0.80)
    # 2. Cohen's Kappa (Target: 0.73 - 0.78)
    
if __name__ == "__main__":
    main()
