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
        
        # Check if files exist, if not return empty list and warn
        if not glob.glob(os.path.join(dataset_path, f"{subject_str}*")):
            print(f"Warning: Data for subject {subject_id} not found in {dataset_path}. Skipping.")
            return []
            
        try:
            paths = mne.datasets.sleep_physionet.age.fetch_data(
                subjects=[subject_id], recording=recording, path=self.data_path
            )
            return paths
        except Exception as e:
            print(f"Error fetching subject {subject_id}: {e}")
            return []

    def preprocess_inference_recording(self, psg_path: str) -> np.ndarray:
        print(f"DEBUG: Inference Processing {psg_path}")
        # 1. Load data metadata only
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
        
        # 2. Select channels FIRST to save memory
        available_channels = raw.ch_names
        picks = [ch for ch in self.ch_names if ch in available_channels]
        if len(picks) < len(self.ch_names):
            print(f"Warning: Expected channels {self.ch_names}, but only found {picks}.")
            if len(picks) == 0:
                raise ValueError("None of the required channels were found in the EDF file.")
        
        raw.pick(picks)
        
        print("DEBUG: Loading picked channels into RAM...")
        raw.load_data()
        
        print("DEBUG: Filtering...")
        eeg_picks = [ch for ch in ['EEG Fpz-Cz', 'EEG Pz-Oz'] if ch in picks]
        eog_picks = [ch for ch in ['EOG horizontal'] if ch in picks]
        
        if eeg_picks:
            raw.filter(l_freq=0.5, h_freq=35.0, picks=eeg_picks, 
                       method='iir', iir_params={'order': 2, 'ftype': 'butter'}, verbose=False)
        if eog_picks:
            raw.filter(l_freq=0.3, h_freq=10.0, picks=eog_picks, 
                       method='iir', iir_params={'order': 2, 'ftype': 'butter'}, verbose=False)
                   
        print("DEBUG: Epoching...")
        events = mne.make_fixed_length_events(raw, duration=self.epoch_duration)
        
        tmax = self.epoch_duration - 1. / raw.info['sfreq']
        epochs = mne.Epochs(
            raw, events, tmin=0., tmax=tmax, 
            baseline=None, preload=True, verbose=False
        )
            
        print("DEBUG: Extracting data (this might use RAM)...")
        X = epochs.get_data()
        
        print("DEBUG: Casting to float32...")
        ch_indices = [epochs.ch_names.index(ch) for ch in picks]
        X = X[:, ch_indices, :].astype(np.float32)
        
        if len(picks) < len(self.ch_names):
            full_X = np.zeros((X.shape[0], len(self.ch_names), X.shape[2]), dtype=np.float32)
            for i, expected_ch in enumerate(self.ch_names):
                if expected_ch in picks:
                    idx = picks.index(expected_ch)
                    full_X[:, i, :] = X[:, idx, :]
            X = full_X
        
        print("DEBUG: Freeing raw memory...")
        del raw
        del epochs
        import gc
        gc.collect()
        
        print("DEBUG: Normalizing...")
        for c in range(X.shape[1]):
            ch_data = X[:, c, :]
            mean_val = np.mean(ch_data)
            std_val = np.std(ch_data)
            if std_val > 0:
                X[:, c, :] = (ch_data - mean_val) / std_val
                
        return X

    def preprocess_recording(self, psg_path: str, hyp_path: str) -> Tuple[np.ndarray, np.ndarray]:
        print(f"DEBUG: Processing {psg_path}")
        # 1. Load data metadata only
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
        annot = mne.read_annotations(hyp_path)
        raw.set_annotations(annot, emit_warning=False)
        
        # 2. Select channels FIRST to save memory
        raw.pick(self.ch_names)
        
        print("DEBUG: Loading picked channels into RAM...")
        raw.load_data()
        
        print("DEBUG: Filtering...")
        # 3. Filtering
        # EEG channels (0.5 - 35 Hz)
        raw.filter(l_freq=0.5, h_freq=35.0, picks=['EEG Fpz-Cz', 'EEG Pz-Oz'], 
                   method='iir', iir_params={'order': 2, 'ftype': 'butter'}, verbose=False)
        # EOG channel (0.3 - 10 Hz)
        raw.filter(l_freq=0.3, h_freq=10.0, picks=['EOG horizontal'], 
                   method='iir', iir_params={'order': 2, 'ftype': 'butter'}, verbose=False)
                   
        print("DEBUG: Epoching...")
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
        
        print("DEBUG: Extracting labels...")
        # Get labels
        mapped_labels = epochs.events[:, 2]
            
        print("DEBUG: Extracting data (this might use RAM)...")
        # Get data array shape (N, channels, times)
        X = epochs.get_data()
        
        print("DEBUG: Casting to float32...")
        # Ensure channel order matches self.ch_names
        # MNE might reorder, so we explicitly select indices
        ch_indices = [epochs.ch_names.index(ch) for ch in self.ch_names]
        X = X[:, ch_indices, :].astype(np.float32)
        
        print("DEBUG: Freeing raw memory...")
        # Free MNE objects from RAM immediately
        del raw
        del epochs
        import gc
        gc.collect()
        
        print("DEBUG: Normalizing...")
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
