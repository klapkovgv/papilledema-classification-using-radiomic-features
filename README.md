# papilledema classification using radiomic features
Papilledema classification using radiomic features and a machine learning pipeline. It includes MRMR feature selection, Optuna hyperparameter optimization, patient-level cross-validation, and a soft voting ensemble.

>  This is an educational project developed during my university studies,
> reflecting my hands-on experience.

Note: This repository is currently in the editing phase.

## Introduction

**Papilledema** is a critical clinical finding defined as edema of the optic disc, and serves as a key indicator of elevated intracranial pressure.

**Radiomics** is a rapidly evolving field in medical imaging analysis that bridges radiology and oncology. It involves high-throughput extraction of quantitative features from standard medical images (CT, MRI, PET scans), converting them into mineable, high-dimensional data. A major development in this space is radiogenomics (the integration of radiomics with genomic data), which advances personalized medicine by combining the spatial resolution of imaging data with the molecular specificity of genomic data.

## What this Project Does

This project implements a comprehensive machine learning pipeline to classify papilledema vs. normal cases using radiomic features extracted from optic disc images.

The pipeline includes:
* Data preprocessing
* MRMR (Maximum Relevance Minimum Redundancy) feature selection
* Optuna hyperparameter optimization
* Patient-level cross-validation with 20 repeated splits
* Sigmoid probability calibration
* Soft-voting ensemble model

### Models Evaluated

Logistic Regression (LR), Support Vector Machine (SVM), Random Forest (RF), Extra Trees (ET), Gradient Boosting (GB), K-Nearest Neighbors (KNN).

All preprocessing and feature selection steps are fitted exclusively on training data, and patient-level splitting ensures that samples from the same patient never appear in different subsets simultaneously (strictly preventing data leakage).

## Dataset

The preview of the datasets:

Samples from healthy subjects

<img width="1724" height="414" alt="image" src="https://github.com/user-attachments/assets/ffb56724-5b5c-4a85-90f4-502c9e9ae634" />

Samples from patients diagnosed with papilledema

<img width="1712" height="411" alt="image" src="https://github.com/user-attachments/assets/668a914b-1412-4ff1-8247-b85efab225a7" />

Each sample represents a single ROI (Region of Interest) frame component extracted from optic disc imaging. The dataset is specifically designed to train and evaluate machine learning models to distinguish between healthy eyes and conditions that cause swelling of the optic nerve head.

To load and merge the datasets, I used the following code:

```python
# Load data
normal = pd.read_csv('normal.csv')
papilledema = pd.read_csv('papilledema.csv')

normal['label'] = 0
papilledema['label'] = 1

# Merge
df = pd.concat([normal, papilledema], ignore_index=True)

# Fix infinity values
feature_cols = [col for col in df.columns if col.startswith('Feature_')]
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

# Patient list
patients = df['PatientIndex'].unique()

print("Dataset shape:", df.shape)
print("Total patients:", len(patients))
print("Feature count:", len(feature_cols))
print("Class distribution:\n", df['label'].value_counts())
```

The output is shown below:
```text
Dataset shape: (966, 749)
Total patients: 48
Feature count: 746
Class distribution:
label
0    672
1    294
Name: count, dtype: int64
```

### Important Dataset Characteristics

Multi-instance structure — each patient contributes multiple samples from both the right and left eyes. This makes standard random splitting unsuitable and requires a patient-level splitting strategy.

Mixed-class patients — a single patient can have samples from both classes (e.g., one eye may be normal while the other shows signs of papilledema). This further reinforces the need for patient-level data partitioning rather than sample-level splitting.

The moderate class imbalance (69.6% Normal vs 30.4% Papilledema) was addressed by using Macro-F1 as the primary evaluation metric, which treats both classes equally regardless of their frequency.


















