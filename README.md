# EEG Motor Imagery Classification  
**Hybrid pipeline combining Matlab preprocessing and Python-based machine learning**

---

## 🎯 Project Goal

The goal of this project is to **classify EEG signals recorded during motor imagery tasks**, i.e. when a subject imagines moving a limb (left or right hand).
By decoding these imagined movements from brain activity, we aim to support the development of **brain-computer interfaces (BCIs)** for motor rehabilitation, prosthetic control, and neurofeedback systems.

The pipeline combines:
- Signal preprocessing in Matlab
- Feature-based traditional **linear models** (LDA, Logistic Regression)
- Sequence-based **deep learning** (ConvLSTM) for temporal-spatial EEG sequence learning
to compare interpretability and performance across methods.

This project uses **EEG recordings** from a motor imagery experiment inspired by  
  *Leeb et al.*, “Brain-Computer Communication: Motivation, Aim, and Impact of Exploring a Virtual Apartment,”  
  *IEEE Trans. Neural Systems and Rehabilitation Engineering, 2007*.

---

## Repository Overview

MOTOR_IMAGERY_EEG/
├── data/                          # Processed and raw datasets (.mat, .csv)
│   ├── training_set.mat
│   ├── test_set.mat
│   ├── feature_training_set.csv
│   └── feature_test_set.csv
│
├── notebooks/                     # Main Jupyter notebooks
│   ├── linear_model_notebook.ipynb    # LDA and Logistic Regression
│   └── deep_learning_notebook.ipynb   # ConvLSTM-based sequence model
│
├── src/                           # Python package with reusable modules
│   ├── data_processing.py
│   ├── data_visualisation.py
│   ├── deep_learning.py
│   ├── model_selection.py
│   └── utils.py
│
├── models/                        # Saved trained models (.h5, .npy)
│
├── results/                       # Evaluation outputs (plots, reports)
│   ├── LDA_confusion_matrix.png
│   ├── LDA_roc_curve.png
│   ├── LDA_classification_report.txt
│   ├── ConvLSTM_confusion_matrix.png
│   ├── ConvLSTM_roc_curve.png
│   └── ConvLSTM_classification_report.txt
│
├── matlab/                        # Preprocessing scripts (artifact removal, filtering)
│
├── documentation/                 # Project notes and references
│
├── LICENSE
├── pyproject.toml
├── poetry.lock
└── README.md



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

Notebook: `notebooks/linear_model_notebook.ipynb`

**Input:** band-power features extracted from Matlab (`feature_training_set.csv`, `feature_test_set.csv`)

**Pipeline:**
- Standardization using `StandardScaler`
- Correlation-based feature selection
- Sequential feature selection (`SFS` outperformed `SBS`, `SFFS`, `SBFS` methods)
- Classification using:
  - Linear Discriminant Analysis (LDA)
  - Logistic Regression

**Results:**
| Model                                  | Accuracy |  ROC-AUC | Macro F1 |
| :------------------------------------- | :------: | :------: | :------: |
| **Linear Discriminant Analysis (LDA)** | **0.87** | **0.94** | **0.87** |
| **Logistic Regression**                |   0.85   |   0.93   |   0.86   |



## 3. Classification - Deep Learning Models (Python)

Notebook: `notebooks/deep_learning_notebook.ipynb`

**Input:** preprocessed `.mat` files (`training_set.mat`, `test_set.mat`)  
Each trial corresponds to 3 channels (C3, Cz, C4) and 3s (≈ 750 samples).

Steps:

#### 3.1 Data Augmentation and video-like represntation
Each 3 s segment is split into **three 1 s windows** (250 samples each).  
Each 1 s window is further divided into:
- **5 frames × 50 samples** → temporal representation for ConvLSTM.

Resulting shape: (N, time_steps=5, rows=50, cols=3, channels=1)

#### 3.2 Model Architecture - ConvLSTM
A compact spatiotemporal model using 2 ConvLSTM blocks:

```python
ConvLSTM2D(filters=16, kernel_size=(3,3), return_sequences=True)
ConvLSTM2D(filters=32, kernel_size=(3,3), return_sequences=False)
Dense(64, activation='relu')
Dense(2, activation='softmax')
```

#### 3.3 Training

- **Optimizer:** `Adam (learning rate = 1e-3)`  
- **Loss:** `binary_crossentropy`  

#### 3.4 Evaluation and Results
The model was evaluated on the 1s sequences and on the 3s moro imagery sequence (3s).

| Metric   | Value    |
| -------- | -------- |
| Accuracy | **0.91** |
| ROC-AUC  | **0.98** |
| Macro F1 | **0.91** |


---

## 4. Results  Summary

| Model               | Input               | Accuracy | AUC      | F1 (macro) | Notes                                  |
| ------------------- | ------------------- | -------- | -------- | ---------- | -------------------------------------- |
| LDA                 | Band-power features | **0.87** | **0.94** | 0.87       | Baseline linear classifier             |
| Logistic Regression | Band-power features | 0.85     | 0.93     | 0.86       | Simple linear model                    |
| ConvLSTM            | Raw EEG sequences   | **0.91** | **0.98** | 0.91       | Captures temporal–spatial dependencies |


---

## 5. Key Insights

- Linear models (LDA, LR) achieve strong baseline accuracy on EEG-band-based features.
- The ConvLSTM captures richer spatiotemporal information, improving both accuracy/ f1-score and AUC.

---

## 6. Next Steps

- Integrate cross-subject validation
- Explore frequency–temporal attention models

---

## Author 

Developed by Elisa Vasta



