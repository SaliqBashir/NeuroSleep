import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import DualBranchCNN, SequenceLSTM, NeuroSleepModel

def test_dual_branch_cnn():
    model = DualBranchCNN(in_channels=3, emb_dim=256)
    model.eval()
    
    # Dummy data: Batch=4, Channels=3, Length=3000
    x = torch.randn(4, 3, 3000)
    
    with torch.no_grad():
        out = model(x)
        
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"
    print("DualBranchCNN test passed.")

def test_sequence_lstm():
    model = SequenceLSTM(input_dim=256, hidden_dim=512, num_layers=2, num_classes=5)
    model.eval()
    
    # Dummy data: Batch=2, SeqLen=20, Features=256
    x = torch.randn(2, 20, 256)
    
    with torch.no_grad():
        out = model(x)
        
    assert out.shape == (2, 20, 5), f"Expected (2, 20, 5), got {out.shape}"
    print("SequenceLSTM test passed.")

def test_neurosleep_model():
    model = NeuroSleepModel(in_channels=3, cnn_emb_dim=256, lstm_hidden=512, num_classes=5)
    model.eval()
    
    # Dummy data: Batch=2, SeqLen=20, Channels=3, Length=3000
    x = torch.randn(2, 20, 3, 3000)
    
    with torch.no_grad():
        out = model(x)
        
    assert out.shape == (2, 20, 5), f"Expected (2, 20, 5), got {out.shape}"
    print("NeuroSleepModel test passed.")

if __name__ == "__main__":
    test_dual_branch_cnn()
    test_sequence_lstm()
    test_neurosleep_model()
