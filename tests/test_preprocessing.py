import os
import sys
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocess import SleepEDFPreprocessor

def test_preprocessing():
    # Initialize preprocessor
    preprocessor = SleepEDFPreprocessor()
    
    # Fetch subject 0, recording 1
    paths = preprocessor.fetch_subject(subject_id=0, recording=[1])
    
    # Assert paths were downloaded and we got one recording
    assert len(paths) > 0
    psg_path, hyp_path = paths[0]
    assert os.path.exists(psg_path)
    assert os.path.exists(hyp_path)
    
    # Run preprocessing
    X, y = preprocessor.preprocess_recording(psg_path, hyp_path)
    
    # Test output types
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    
    # Test output shapes
    assert X.ndim == 3
    N_epochs = X.shape[0]
    assert X.shape[1] == 3 # 3 channels
    assert X.shape[2] == 3000 # 30s at 100Hz
    assert y.shape == (N_epochs,)
    
    # Test labels are integers and within expected range (0-4)
    unique_labels = np.unique(y)
    for label in unique_labels:
        assert label in [0, 1, 2, 3, 4]
        
    # Test normalization (Z-score)
    # Mean of each channel across the whole recording should be ~0 and std ~1
    for c in range(3):
        ch_mean = np.mean(X[:, c, :])
        ch_std = np.std(X[:, c, :])
        assert np.isclose(ch_mean, 0, atol=1e-3)
        assert np.isclose(ch_std, 1, atol=1e-3)
        
    print(f"Test passed successfully! Generated {N_epochs} epochs. Labels distribution: {np.bincount(y)}")

if __name__ == "__main__":
    test_preprocessing()
