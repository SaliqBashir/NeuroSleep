# NeuroSleep: Automated Sleep Stage Classification

## Objective
Implement a fully reproducible, end-to-end deep learning pipeline to automatically classify sleep stages from raw polysomnographic (PSG) data, matching expert inter-scorer agreement levels.

## Core Problem Statement & Constraints
1. **Manual Bottleneck:** Human scoring takes 60-90 minutes per recording. Inter-scorer reliability (Cohen's kappa) is around 0.70 - 0.80. The system must hit this range.
2. **Class Imbalance:** Sleep data is inherently skewed:
   - **Wake** (~8.9%)
   - **N1 / Light Sleep** (~6.8%) -> **CRITICAL BOTTLENECK:** High misclassification risk due to underrepresentation.
   - **N2 / Intermediate** (~51.2%) -> Majority class.
   - **N3 / Deep Sleep** (~14.1%)
   - **REM** (~19.0%)
3. **Temporal Context Dependency:** A single 30-second epoch is insufficient for stage transitions (e.g., N1 to N2). The model uses sequence modeling (past and future epochs).
4. **No Hand-Engineered Features:** The system learns directly from raw signals. We do not use FFTs or PSDs as manual inputs to the network.

## Dataset Specifications
*   **Source:** `Sleep-EDF Cassette` subset from PhysioNet.
*   **Data Volume:** 78 whole-night Holter recordings from 39 healthy Caucasian subjects (2 nights per subject).
*   **Channels Used (3 total):** `EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal`.
*   **Sampling Rate:** 100 Hz.
*   **Splitting Strategy (Subject-Level strictly to prevent leakage):**
    *   **Train:** Subjects 1-32 (64 recordings)
    *   **Validation:** Subjects 33-38 (12 recordings)
    *   **Test:** Subject 40 (Subject 39 is omitted from the dataset)

## System Architecture (The 4-Stage Pipeline)
### Stage 1: Signal Preprocessing
*   **Library:** `MNE-Python`.
*   **Filtering:** 2nd-order Butterworth bandpass filter (EEG: 0.5 to 35 Hz, EOG: 0.3 to 10 Hz).
*   **Epoching & Normalization:** Split continuous signals into 30-second non-overlapping windows (3000 samples). Z-score normalization applied per channel.

### Stage 2: CNN Epoch Feature Extractor (Intra-Epoch)
A dual-branch Convolutional Neural Network (CNN) designed to capture both fast and slow frequency events without spectrograms. Includes `ResidualBlock1D` for deeper extraction.
*   **Branch 1 (Small Filter):** Captures high-frequency events (e.g., sleep spindles).
*   **Branch 2 (Large Filter):** Captures low-frequency events (e.g., delta waves).
*   **Output:** 256-dimensional embedding vector per epoch.

### Stage 3: Bidirectional LSTM Sequence Model (Inter-Epoch)
Models temporal transitions using a sliding window.
*   **Input Sequence Length:** 20 consecutive epoch embeddings (10 minutes of chronological context).
*   **Architecture:** 2-layer Bidirectional LSTM (512 hidden units per direction).

### Stage 4: Training, Imbalance Countermeasures, & Inference
*   **Optimizer:** `AdamW` (LR = 1e-4) with Cosine Annealing.
*   **Countermeasure 1 (Loss):** Custom `FocalLoss` with smoothed inverse frequency weights to focus heavily on hard, rare classes like N1.
*   **Countermeasure 2 (Augmentation):** Additive Gaussian noise injected strictly on minority `N1` training epochs (Std Dev = 5% of epoch's RMS amplitude).

## Hardware & Software Requirements
*   **Framework:** PyTorch 2.1+ (CUDA 12.1 or Apple Silicon MPS support).
*   **Libraries:** `mne >= 1.6`, `numpy >= 1.25`, `scipy >= 1.11`, `scikit-learn >= 1.3`, `matplotlib`, `seaborn`.
*   **Hardware:** CPU viable for inference. GPU/MPS highly recommended for training.

## Evaluation Metrics
The system is evaluated based on:
1.  **Macro-average F1 Score** (Target: 0.77 - 0.80)
2.  **Cohen's Kappa** (Target: 0.73 - 0.78)
3.  **Overall Accuracy**
4.  **Per-class F1 Score** (Specifically to audit N1 and N3 performance).

## File Structure & What Each File Does
*   `src/preprocess.py`: Handles checking for data via `mne`, bandpass filtering, epoch segmentation (30s), label mapping, and Z-score normalization. Blocks automatic downloading.
*   `src/dataset.py`: Defines the PyTorch Dataset class. Groups epochs into sequences of 20, applies Gaussian noise augmentation to N1 classes, and computes smoothed class weights.
*   `src/model.py`: Defines the neural network architecture (`DualBranchCNN` with residual blocks and `SequenceLSTM`).
*   `src/train.py`: The training loop. Implements `FocalLoss`, configures `AdamW` and schedulers, tracks validation metrics, logs history to CSV, and saves `best_model.pth`.
*   `src/evaluate.py`: The evaluation script. Runs sliding-window inference (`stride=1`) on the held-out test subject using `best_model.pth` and reports final classification metrics.
*   `tests/`: Contains pytest unit tests for the preprocessor and model architecture to ensure code integrity.

## How to Run the Program
Ensure you have the required dependencies installed and your virtual environment activated:
```bash
source venv/bin/activate
```

**1. Training the Model**
```bash
python src/train.py
```
*Optional flags:*
*   `--epochs 50`: Set epochs (default: 100)
*   `--batch_size 16`: Set batch size (default: 32)
*   `--limit-subjects 10`: Load fewer subjects for a faster test run.
*   `--test-mode`: Run 1 quick epoch with random dummy data to verify the script compiles.

**2. Evaluating the Model**
After training is complete (or if you already have a `best_model.pth`), evaluate it on the unseen test set:
```bash
python src/evaluate.py
```

## How to Test
The project includes unit tests for the core logic. To run them, execute:
```bash
pytest
```
