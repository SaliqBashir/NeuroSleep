import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, classification_report
from tqdm import tqdm

from dataset import SleepSequenceDataset, compute_class_weights
from model import NeuroSleepModel
from preprocess import SleepEDFPreprocessor

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.weight is not None:
            focal_loss = focal_loss * self.weight[targets]
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def _process_subject_night(sid, n, psg_path, hyp_path, cache_dir):
    from preprocess import SleepEDFPreprocessor
    import os
    import numpy as np
    preprocessor = SleepEDFPreprocessor()
    X, y = preprocessor.preprocess_recording(psg_path, hyp_path)
    x_path = os.path.join(cache_dir, f"subject_{sid}_night_{n}_X.npy")
    y_path = os.path.join(cache_dir, f"subject_{sid}_night_{n}_y.npy")
    np.save(x_path, X)
    np.save(y_path, y)

def load_real_data(subject_ids, cache_dir="./data_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    from preprocess import SleepEDFPreprocessor
    preprocessor = SleepEDFPreprocessor()
    X_list = []
    y_list = []
    
    import multiprocessing as mp
    
    for sid in tqdm(subject_ids, desc="Loading/Preprocessing subjects"):
        night_idx = 0
        while True:
            x_path = os.path.join(cache_dir, f"subject_{sid}_night_{night_idx}_X.npy")
            y_path = os.path.join(cache_dir, f"subject_{sid}_night_{night_idx}_y.npy")
            if os.path.exists(x_path) and os.path.exists(y_path):
                # Load with mmap to use ZERO RAM!
                X_list.append(np.load(x_path, mmap_mode='r'))
                y_list.append(np.load(y_path, mmap_mode='r'))
                night_idx += 1
            else:
                break
                
        if night_idx == 0:
            # Clean up old .npz if they exist to force rebuild
            old_npz = os.path.join(cache_dir, f"subject_{sid}.npz")
            if os.path.exists(old_npz):
                os.remove(old_npz)
                
            # Fetch files and preprocess
            paths = preprocessor.fetch_subject(sid)
            if not paths:
                continue
            
            for n, (psg_path, hyp_path) in enumerate(paths):
                try:
                    p = mp.Process(target=_process_subject_night, args=(sid, n, psg_path, hyp_path, cache_dir))
                    p.start()
                    p.join()
                    
                    if p.exitcode != 0:
                        raise RuntimeError(f"Preprocessing subprocess failed with exit code {p.exitcode}")
                    
                    x_path = os.path.join(cache_dir, f"subject_{sid}_night_{n}_X.npy")
                    y_path = os.path.join(cache_dir, f"subject_{sid}_night_{n}_y.npy")
                    
                    # Immediately load them back mapped
                    X_list.append(np.load(x_path, mmap_mode='r'))
                    y_list.append(np.load(y_path, mmap_mode='r'))
                except Exception as e:
                    print(f"\nCRASHED ON SUBJECT {sid} NIGHT {n} WITH ERROR:")
                    import traceback
                    traceback.print_exc()
                    raise
                
    return X_list, y_list

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x) # (B, 20, 5)
                logits_flat = logits.view(-1, 5)
                y_flat = y.view(-1)
                loss = criterion(logits_flat, y_flat)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x) # (B, 20, 5)
            
            # CrossEntropyLoss expects (B, C, d1, d2, ...) or flattened
            # Reshape to (B * 20, 5) and y to (B * 20)
            logits_flat = logits.view(-1, 5)
            y_flat = y.view(-1)
            
            loss = criterion(logits_flat, y_flat)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            if device.type == 'cuda':
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(x)
                    logits_flat = logits.view(-1, 5)
                    y_flat = y.view(-1)
                    loss = criterion(logits_flat, y_flat)
            else:
                logits = model(x)
                logits_flat = logits.view(-1, 5)
                y_flat = y.view(-1)
                loss = criterion(logits_flat, y_flat)
                
            total_loss += loss.item()
            
            preds = torch.argmax(logits_flat, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_flat.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)
    
    return total_loss / len(dataloader), macro_f1, kappa, acc, all_targets, all_preds

def main():
    parser = argparse.ArgumentParser(description="NeuroSleep Training Pipeline")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers")
    parser.add_argument('--test-mode', action='store_true', help="Run a quick test with dummy data")
    parser.add_argument('--limit-subjects', type=int, default=None, help="Limit number of subjects loaded (for fast testing)")
    parser.add_argument('--cache-dir', type=str, default="./data_cache", help="Directory to cache preprocessed subject data")
    args = parser.parse_args()
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        # Enable TF32 for Ampere (RTX 30 series) and newer
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    if args.test_mode:
        print("Running in test mode with dummy data...")
        # 2 recordings, 100 epochs each for train
        X_train = [np.random.randn(100, 3, 3000) for _ in range(2)]
        y_train = [np.random.randint(0, 5, size=(100,)) for _ in range(2)]
        
        # 1 recording, 40 epochs for val
        X_val = [np.random.randn(40, 3, 3000)]
        y_val = [np.random.randint(0, 5, size=(40,))]
        
        # Make dummy weights
        class_weights = compute_class_weights(y_train)
    else:
        # Define subject splits
        # Train: Subjects 1-32. Val: Subjects 33-38. Test: Subject 40 (since 39 is missing from PhysioNet)
        train_subjects = [i for i in range(1, 33) if i != 39]
        val_subjects = [i for i in range(33, 39) if i != 39]
        test_subjects = [40]
        
        if args.limit_subjects:
            print(f"Limiting dataset load to {args.limit_subjects} train subjects...")
            train_subjects = train_subjects[:args.limit_subjects]
            val_subjects = val_subjects[:max(1, args.limit_subjects // 2)]
            test_subjects = test_subjects[:1]
            
        print("Loading real training data...")
        X_train, y_train = load_real_data(train_subjects, cache_dir=args.cache_dir)
        print("Loading real validation data...")
        X_val, y_val = load_real_data(val_subjects, cache_dir=args.cache_dir)
        
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("No data loaded. Check connection to PhysioNet or cache directory contents.")
            
        class_weights = compute_class_weights(y_train)
        print(f"Computed Class Weights: {class_weights.numpy()}")

    # Setup datasets & loaders
    train_dataset = SleepSequenceDataset(X_train, y_train, sequence_length=30, stride=30, is_train=True)
    val_dataset = SleepSequenceDataset(X_val, y_val, sequence_length=30, stride=1, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    
    model = NeuroSleepModel().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = FocalLoss(weight=class_weights.to(device), gamma=2.0)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_f1 = 0.0
    
    # Setup CSV logging
    log_file = open("training_history.csv", mode="w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_Macro_F1", "Val_Kappa", "Val_Acc"])
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, macro_f1, kappa, acc, targets, preds = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Macro F1: {macro_f1:.4f} | Val Kappa: {kappa:.4f} | Val Acc: {acc:.4f}")
        
        # Log to CSV
        csv_writer.writerow([epoch, train_loss, val_loss, macro_f1, kappa, acc])
        log_file.flush()
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), "best_model.pth")
            print("=> Saved new best model")
            
    log_file.close()
            
    # Final evaluation printout
    print("\nTraining completed. Final Classification Report on last validation pass:")
    report = classification_report(targets, preds, target_names=['Wake', 'N1', 'N2', 'N3', 'REM'], zero_division=0)
    print(report)
    
    with open("classification_report.txt", "w") as f:
        f.write("Final Classification Report on last validation pass:\n\n")
        f.write(report)
    
if __name__ == "__main__":
    main()

