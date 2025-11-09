# EEG Motor Imagery Classification  
**Hybrid pipeline combining Matlab preprocessing and Python-based machine learning**

---

## 🎯 Project Goal

The goal of this project is to **classify EEG signals recorded during motor imagery tasks** -that is, when a subject imagines moving a limb (e.g., left or right hand).  
By decoding these imagined movements from brain activity, we aim to support the development of **brain-computer interfaces (BCIs)** for motor rehabilitation, prosthetic control, and neurofeedback systems.

Specifically, this project:
- Uses **EEG recordings** from a motor imagery experiment inspired by  
  *Leeb et al.*, “Brain–Computer Communication: Motivation, Aim, and Impact of Exploring a Virtual Apartment,”  
  *IEEE Trans. Neural Systems and Rehabilitation Engineering, 2007*.
- Implements a **complete processing pipeline**, from raw signal cleaning to feature extraction and classification.
- Compares **traditional linear models** (LDA, Logistic Regression) with a **deep learning approach** (ConvLSTM) for temporal–spatial EEG sequence learning.

---

## Overview

This repository integrates:
- **Signal preprocessing in Matlab**, to clean, segment, and prepare EEG trials.
- **Feature-based classification in Python**, using LDA and Logistic Regression for interpretable baselines.
- **Sequence-based deep learning in Python**, using ConvLSTM to capture spatiotemporal dependencies in raw EEG signals.

Together, these components form a reproducible framework for EEG motor imagery analysis, from signal-level processing to model evaluation.

---

## 1. Data Acquisition & Preprocessing (Matlab)

All low-level signal processing steps are executed in Matlab scripts (`preprocessing/` folder).

**Main steps:**
1. **Artifact removal** - regress out EOG channels using linear regression.
2. **Band-pass filtering (2–60 Hz)** - to isolate EEG frequency bands.
3. **Trial alignment** - correct the starting points of trials based on event markers.
4. **Artifact rejection** - remove epochs exceeding amplitude thresholds.
5. **Segmentation** - extract 3 s motor imagery periods from each trial.
6. **Export to `.mat` format** - each dataset saved as:
   - `training_set.mat`
   - `test_set.mat`
   containing fields:
   ```matlab
   training_set.eeg_sequences = { [750 × 3], ... };
   training_set.label = [N × 1];
   ```


---

## 2. Classification - Linear Models (Python)

Notebook: `EEG_Linear_model.ipynb`

**Input:** band-power features extracted from Matlab (`trainset_feat_new.csv`, `testset_feat_new.csv`)

**Pipeline:**
- Standardization using `StandardScaler`
- Correlation-based feature selection
- Sequential feature selection (`SFS`, `SBS`, `SFFS`, `SBFS`)
- Classification using:
  - Linear Discriminant Analysis (LDA)
  - Logistic Regression

**Outputs:**
- Selected feature list  
- Cross-validated accuracy  
- Confusion matrix and ROC curve  


## 3. Classification - Deep Learning Models (Python)

Notebook: `EEG_Deep_learning.ipynb` (under development)

**Input:** preprocessed `.mat` files (`training_set.mat`, `test_set.mat`)  
Each trial corresponds to 3 channels (C3, Cz, C4) and 3 s (≈ 750 samples).

#### 3.1 Data Augmentation  
Each 3 s segment is split into **three 1 s windows** (250 samples each).  
Each 1 s window is further divided into:
- **5 frames × 50 samples** → temporal representation for ConvLSTM.

Resulting shape: (N, time_steps=5, rows=50, cols=3, channels=1)


#### 3.2 Model Architecture – ConvLSTM
A compact spatiotemporal model using 2 ConvLSTM blocks:

```python
ConvLSTM2D(filters=16, kernel_size=(3,3), return_sequences=True)
ConvLSTM2D(filters=32, kernel_size=(3,3), return_sequences=False)
Dense(64, activation='relu')
Dense(2, activation='softmax')
```

#### 3.3 Training

- **Optimizer:** `Adam (learning rate = 1e-3)`  
- **Loss:** `SparseCategoricalCrossentropy`  

---

## 4. Evaluation

- Confusion Matrix  
- ROC-AUC Score  
- Accuracy on held-out test set


---

## 📊 5. Results 

| Model | Input | Accuracy | AUC | Notes |
|-------|--------|----------|-----|-------|
| LDA | Band-power features | 0.87 | 0.94 | Fast baseline |
| Logistic Regression | Band-power features | 0.85 | 0.93 | - |
| ConvLSTM | Raw 3 × EEG sequences | 0.90| - | Temporal-spatial modeling |




