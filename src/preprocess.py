import os
import numpy as np
import mne
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
import glob

# Mapping Sleep-EDF annotations to integer labels
# 0: Wake, 1: N1, 2: N2, 3: N3, 4: REM
# Note: N4 is merged with N3 in the AASM standard. 
ANNOTATION_MAP = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3, # Merge N4 into N3
    'Sleep stage R': 4
}
IGNORE_LABELS = ['Movement time', 'Sleep stage ?']

class SleepEDFPreprocessor:
    def __init__(self, data_path: str = None):
        """
        Initializes the preprocessor for NeuroSleep pipeline.
        
        Args:
            data_path (str, optional): Path to save/load dataset. Defaults to mne_data.
        """
        self.data_path = data_path
        self.ch_names = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
        self.sfreq = 100
        self.epoch_duration = 30.0 # seconds
        
    def fetch_subject(self, subject_id: int, recording: List[int] = [1, 2]) -> List[List[str]]:
        """
        Fetches the PSG and hypnogram file paths for a given subject.
        """
        base_path = os.path.expanduser(self.data_path or '~/mne_data')
        dataset_path = os.path.join(base_path, 'physionet-sleep-data')
        subject_str = f"SC4{subject_id:02d}"
        
        # Check if files exist, if not raise error instead of downloading
        if not glob.glob(os.path.join(dataset_path, f"{subject_str}*")):
            raise FileNotFoundError(f"Data for subject {subject_id} not found in {dataset_path}. Automatic download is disabled.")
            
        try:
            paths = mne.datasets.sleep_physionet.age.fetch_data(
                subjects=[subject_id], recording=recording, path=self.data_path
            )
            return paths
        except Exception as e:
            print(f"Error fetching subject {subject_id}: {e}")
            return []

    def preprocess_recording(self, psg_path: str, hyp_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses a single recording (one night).
        
        Args:
            psg_path (str): Path to the raw PSG .edf file.
            hyp_path (str): Path to the hypnogram .edf file.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (N, 3, 3000) and y (N,)
        """
        # 1. Load data
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        annot = mne.read_annotations(hyp_path)
        raw.set_annotations(annot, emit_warning=False)
        
        # 2. Select channels
        raw.pick(self.ch_names)
        
        # 3. Filtering
        # EEG channels (0.5 - 35 Hz)
        raw.filter(l_freq=0.5, h_freq=35.0, picks=['EEG Fpz-Cz', 'EEG Pz-Oz'], 
                   method='iir', iir_params={'order': 2, 'ftype': 'butter'}, verbose=False)
        # EOG channel (0.3 - 10 Hz)
        raw.filter(l_freq=0.3, h_freq=10.0, picks=['EOG horizontal'], 
                   method='iir', iir_params={'order': 2, 'ftype': 'butter'}, verbose=False)
                   
        # 4. Epoching and Annotation alignment
        events, event_id = mne.events_from_annotations(
            raw, event_id=ANNOTATION_MAP,
            chunk_duration=self.epoch_duration, verbose=False
        )
        
        # Create Epochs
        tmax = self.epoch_duration - 1. / raw.info['sfreq']
        epochs = mne.Epochs(
            raw, events, event_id=event_id, tmin=0., tmax=tmax, 
            baseline=None, preload=True, verbose=False
        )
        
        # Get labels
        mapped_labels = epochs.events[:, 2]
            
        # Get data array shape (N, channels, times)
        X = epochs.get_data()
        
        # Ensure channel order matches self.ch_names
        # MNE might reorder, so we explicitly select indices
        ch_indices = [epochs.ch_names.index(ch) for ch in self.ch_names]
        X = X[:, ch_indices, :]
        
        # 6. Z-score normalization per channel
        # We need to normalize across the *entire recording* (which is the combined data in X)
        # X shape is (N_epochs, 3, 3000)
        # We compute mean and std for each channel over all epochs and timepoints
        for c in range(X.shape[1]):
            ch_data = X[:, c, :]
            mean_val = np.mean(ch_data)
            std_val = np.std(ch_data)
            if std_val > 0:
                X[:, c, :] = (ch_data - mean_val) / std_val
                
        return X, mapped_labels
