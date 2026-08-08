import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SleepSequenceDataset, compute_class_weights
from model import NeuroSleepModel
from train import load_real_data, train_one_epoch, evaluate, FocalLoss

def main():
    parser = argparse.ArgumentParser(description="NeuroSleep Hyperparameter Tuning")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs per config")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers")
    parser.add_argument('--test-mode', action='store_true', help="Run a quick test with dummy data")
    parser.add_argument('--limit-subjects', type=int, default=None, help="Limit number of subjects loaded")
    parser.add_argument('--cache-dir', type=str, default="./data_cache", help="Directory to cache preprocessed subject data")
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    if args.test_mode:
        print("Running in test mode with dummy data...")
        X_train = [np.random.randn(100, 3, 3000) for _ in range(2)]
        y_train = [np.random.randint(0, 5, size=(100,)) for _ in range(2)]
        X_val = [np.random.randn(40, 3, 3000)]
        y_val = [np.random.randint(0, 5, size=(40,))]
        class_weights = compute_class_weights(y_train)
    else:
        train_subjects = [i for i in range(1, 33) if i != 39]
        val_subjects = [i for i in range(33, 39) if i != 39]
        
        if args.limit_subjects:
            train_subjects = train_subjects[:args.limit_subjects]
            val_subjects = val_subjects[:max(1, args.limit_subjects // 2)]
            
        print("Loading training data...")
        X_train, y_train = load_real_data(train_subjects, cache_dir=args.cache_dir)
        print("Loading validation data...")
        X_val, y_val = load_real_data(val_subjects, cache_dir=args.cache_dir)
        class_weights = compute_class_weights(y_train)

    train_dataset = SleepSequenceDataset(X_train, y_train, sequence_length=30, stride=30, is_train=True)
    val_dataset = SleepSequenceDataset(X_val, y_val, sequence_length=30, stride=1, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Search Space
    lr_list = [5e-4, 1e-4, 5e-5]
    wd_list = [1e-2, 1e-3, 1e-4]
    gamma_list = [1.5, 2.0, 2.5]
    
    if args.test_mode:
        lr_list = [1e-4]
        wd_list = [1e-2]
        gamma_list = [2.0]
        args.epochs = 1
        
    best_overall_f1 = 0.0
    
    # Setup CSV logging
    with open("tuning_results.csv", mode="w", newline="") as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(["LR", "Weight_Decay", "Gamma", "Best_Epoch", "Best_Val_Macro_F1", "Val_Kappa"])
        
        config_idx = 1
        total_configs = len(lr_list) * len(wd_list) * len(gamma_list)
        
        for lr in lr_list:
            for wd in wd_list:
                for gamma in gamma_list:
                    print(f"\n[{config_idx}/{total_configs}] Testing Config: LR={lr}, WD={wd}, Gamma={gamma}")
                    
                    # Initialize fresh model for each configuration
                    model = NeuroSleepModel().to(device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                    criterion = FocalLoss(weight=class_weights.to(device), gamma=gamma)
                    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
                    
                    best_config_f1 = 0.0
                    best_config_kappa = 0.0
                    best_epoch = 0
                    
                    for epoch in range(1, args.epochs + 1):
                        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
                        val_loss, macro_f1, kappa, acc, targets, preds = evaluate(model, val_loader, criterion, device)
                        scheduler.step()
                        
                        if macro_f1 > best_config_f1:
                            best_config_f1 = macro_f1
                            best_config_kappa = kappa
                            best_epoch = epoch
                            
                            # If this config produced the absolute best model overall, save it
                            if macro_f1 > best_overall_f1:
                                best_overall_f1 = macro_f1
                                torch.save(model.state_dict(), "best_tuned_model.pth")
                                print(f"  -> [NEW GLOBAL BEST!] Macro F1: {best_overall_f1:.4f} (Saved to best_tuned_model.pth)")
                                
                    print(f"  Config Finished! Best F1 for this config: {best_config_f1:.4f} (Epoch {best_epoch})")
                    csv_writer.writerow([lr, wd, gamma, best_epoch, best_config_f1, best_config_kappa])
                    log_file.flush()
                    config_idx += 1
                    
    print(f"\nHyperparameter tuning complete! Best overall Macro F1: {best_overall_f1:.4f}")

if __name__ == "__main__":
    main()
