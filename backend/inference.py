from src.preprocess import SleepEDFPreprocessor
from src.model import NeuroSleepModel
import os
import torch
import numpy as np
import sys

# Ensure src is in path so we can import from it when running from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 0: Wake, 1: N1, 2: N2, 3: N3, 4: REM
STAGE_MAP = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "best_model.pth"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = NeuroSleepModel().to(device)
        if os.path.exists(model_path):
            _model.load_state_dict(torch.load(model_path, map_location=device))
            _model.eval()
        else:
            print(f"Warning: Model weights not found at {model_path}.")
    return _model


def predict_sleep_stages(edf_file_path: str):
    preprocessor = SleepEDFPreprocessor()

    # 1. Preprocess the file
    X = preprocessor.preprocess_inference_recording(edf_file_path)

    # 2. Run inference
    model = get_model()

    # X shape is (N, C, T) e.g., (1000, 3, 3000)
    # The model expects (B, SeqLen, C, T) where SeqLen = 30.
    seq_len = 30
    num_epochs = X.shape[0]

    # Pad to nearest multiple of seq_len if necessary
    remainder = num_epochs % seq_len
    if remainder != 0:
        pad_size = seq_len - remainder
        pad_data = np.zeros(
            (pad_size, X.shape[1], X.shape[2]), dtype=np.float32)
        X = np.concatenate([X, pad_data], axis=0)

    B = X.shape[0] // seq_len
    X_seq = X.reshape(B, seq_len, X.shape[1], X.shape[2])

    all_preds = []

    with torch.no_grad():
        for i in range(B):
            # Shape: (1, 30, 3, 3000)
            batch = torch.tensor(X_seq[i:i+1]).to(device)
            logits = model(batch)  # (1, 30, 5)
            preds = torch.argmax(logits.view(-1, 5), dim=1)  # (30,)
            all_preds.extend(preds.cpu().numpy().tolist())

    # Truncate padded predictions
    all_preds = all_preds[:num_epochs]

    # Map predictions to human readable strings
    return [STAGE_MAP.get(p, "Unknown") for p in all_preds]
