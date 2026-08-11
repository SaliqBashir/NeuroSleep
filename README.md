# NeuroSleep: Automated Sleep Stage Classification

## Overview
NeuroSleep is a comprehensive, end-to-end deep learning system designed to automatically classify sleep stages from raw polysomnographic (PSG) data. The primary objective is to match expert inter-scorer agreement levels, effectively mitigating the manual bottleneck of traditional sleep scoring which typically requires 60-90 minutes per recording.

The system is composed of a PyTorch-based machine learning pipeline, a FastAPI backend for high-performance inference, and a React/Vite frontend featuring a Cyber Brutalist design system for data visualization and analysis.

## System Architecture

The project is divided into three primary domains:

### 1. Machine Learning Pipeline (Core)
The core classification engine utilizes a four-stage architecture:
*   **Signal Preprocessing:** Utilizes MNE-Python to apply 2nd-order Butterworth bandpass filters (EEG: 0.5 to 35 Hz, EOG: 0.3 to 10 Hz) and segment continuous signals into 30-second non-overlapping epochs with Z-score normalization.
*   **Intra-Epoch Feature Extraction:** A Dual-Branch Convolutional Neural Network (CNN) featuring ResidualBlock1D captures both high-frequency (spindles) and low-frequency (delta waves) events directly from raw signals, without relying on hand-engineered features or spectrograms.
*   **Inter-Epoch Sequence Modeling:** A 2-layer Bidirectional LSTM (512 hidden units per direction) models temporal transitions over a sliding window of 20 consecutive epochs (10 minutes of context).
*   **Imbalance Countermeasures:** Employs a custom Focal Loss with smoothed inverse frequency weights and additive Gaussian noise augmentation specifically targeting minority classes like N1 (Light Sleep).

### 2. Backend API
*   Built with FastAPI and served via Uvicorn.
*   Provides RESTful endpoints to handle raw `.edf` file uploads.
*   Manages the execution of the PyTorch inference pipeline and formats the classification sequence into structured analytical data.

### 3. Frontend Web Application
*   Built with React and Vite.
*   Features a custom "Cyber Brutalist" CSS design system (sharp edges, high contrast, neo-brutalist interactions).
*   Allows users to drag-and-drop raw PSG recordings, visualizing the results through summary statistics (Efficiency, Total Sleep, Stage Breakdown) and an interactive Hypnogram chart generated via Recharts.
*   Includes a dedicated, formatted print layout for generating clinical reports.

## Technical Stack
*   **Machine Learning:** PyTorch (CUDA/MPS supported), MNE-Python, Scikit-Learn, NumPy, SciPy
*   **Backend:** Python, FastAPI, Uvicorn, python-multipart
*   **Frontend:** JavaScript (ES6+), React, Vite, Recharts, Axios, Vanilla CSS

## Dataset Specifications
*   **Source:** `Sleep-EDF Cassette` subset from PhysioNet.
*   **Data Volume:** 78 whole-night Holter recordings from 39 healthy Caucasian subjects (2 nights per subject).
*   **Channels Used (3 total):** `EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal`.
*   **Sampling Rate:** 100 Hz.
*   **Splitting Strategy:**
    *   Train: Subjects 1-32 (64 recordings)
    *   Validation: Subjects 33-38 (12 recordings)
    *   Test: Subject 40 (Subject 39 omitted by dataset providers)

## Project Structure
```text
NeuroSleep/
├── src/                # Machine Learning Pipeline (Training, Data Loading, Models)
├── backend/            # FastAPI Server & Inference Logic
├── frontend/           # React / Vite Web Application
├── tests/              # Unit tests for preprocessing and architecture
├── temp_uploads/       # Ephemeral storage for API file processing
├── best_model.pth      # Serialized PyTorch model weights
└── README.md           # Project documentation
```

## Setup & Installation

Ensure you have Python 3.10+ and Node.js 18+ installed on your system.

### 1. Machine Learning & Backend Environment
Create and activate a Python virtual environment, then install the required dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### 2. Frontend Environment
Navigate to the frontend directory and install the Node modules:
```bash
cd frontend
npm install
```

## Usage Guidelines

### Running the Full Application
To use the web interface for sleep analysis, you must start both the backend and frontend servers.

**Start the Backend:**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Start the Frontend:**
```bash
cd frontend
npm run dev
```
Navigate to `http://localhost:5173` in your web browser. Drag and drop a valid `-PSG.edf` file from the Sleep-EDF Cassette dataset to view the analysis.

### Training the Model
If you wish to retrain the neural network from scratch using the `src/` pipeline:
```bash
python src/train.py --epochs 100 --batch_size 32
```

### Evaluating the Model
To evaluate the trained `best_model.pth` on the held-out test subject (Subject 40):
```bash
python src/evaluate.py
```

## Evaluation Target Metrics
The system is evaluated against human inter-scorer reliability benchmarks:
*   **Macro-average F1 Score:** Target 0.77 - 0.80
*   **Cohen's Kappa:** Target 0.73 - 0.78
*   **Per-class F1 Score:** Strict auditing of minority N1 and majority N3 performance.
