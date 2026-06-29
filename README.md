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

## Methodology

The principle of this pipeline is strict prevention of data leakage, ensuring that information from test data never inadvertently influences the training process. Applying data transformation to the entire dataset before splitting is a common mistake and can lead to high bias and overly optimistic model performance.

In medical datasets, a single patient often has multiple data points (X-ray scans, CT slices, clinical records). If we split by individual sample rather than by patient, different samples from the same patient end up in both training and test sets which means the model essentially memorizes diagnoses instead of learning real patterns. This leads to strong evaluation results but poor real-world performance.

To address this, the train-test split process is repeated 20 times with different random seeds, providing a more robust and reliable evaluation framework.

The outer split code is shown in the following list:
```python
outer_splits = []

for seed in range(20):
    p_trainval, p_test = train_test_split(patients, test_size=0.2, random_state=seed)
    p_train, p_val = train_test_split(p_trainval, test_size=0.125, random_state=seed)
    outer_splits.append({
        'train_patients': p_train,
        'val_patients':   p_val,
        'test_patients':  p_test,
        'seed':           seed
    })
```

A nested cross-validation approach was used:
* **Outer loop** — splits data into train/validation/test at the patient level (these are 20 splits)
* **Inner loop** — performs model selection and hyperparameter optimization via Optuna on the training fold only

This ensures hyperparameter selection is based solely on training data, with no information leaking from the test set. Benefits of this approach:
* reduces overfitting by evaluating the model on different data subsets
* provides stable and reliable performance estimates across unseen data

Building a robust, clinical-grade ML framework requires careful architectural design to prevent data leakage, handle high-dimensional feature spaces, and ensure reliable probability estimates. To address these challenges, I implemented the following pipeline:

```text
1.  Data loading and binary label assignment (Normal=0, Papilledema=1)
2.  20 repeated patient-level splits (70% train / 10% val / 20% test)
3.  For each outer split, for each model:
    ├── Inner CV with Optuna hyperparameter optimization
    ├── Preprocessing fitted on inner-training fold only
    ├── MRMR feature selection on inner-training fold only
    └── Model training + inner-validation evaluation (Macro-F1)
4.  Best configuration refitted on full training set
5.  Sigmoid calibration on training set
6.  Validation set used for:
    ├── Threshold optimization (range 0.05–0.95, step 0.01)
    └── Aggregation strategy selection (Macro-F1 + balanced accuracy)
7.  Train + validation merged → final model refitted
8.  Independent test set evaluation
9.  Aggregation of results across 20 splits
10. Soft-voting ensemble (RF + ET + GB)
11. Feature stability analysis
12. Statistical comparison (Friedman, Wilcoxon, Bonferroni)
```

The validation set serves two specific purposes:

1. Threshold Optimization — class-specific probability thresholds are optimized by searching the range 0.05-0.95 in steps of 0.01, maximizing Macro-F1.
2. Aggregation Strategy Selection — the best patient-level aggregation strategy is selected based on Macro-F1 and balanced accuracy. The following strategies were implemented and evaluated:
```python
THRESHOLDS = np.arange(0.05, 0.95, 0.01)

def aggregate_probs(probs, strategy):
    """Aggregate row-level probs to patient level."""
    if strategy == 'mean':
        return np.mean(probs)
    elif strategy == 'max':
        return np.max(probs)
    elif strategy == 'majority_vote':
        preds = (probs >= 0.5).astype(int)
        return 1 if np.sum(preds) > len(preds) / 2 else 0
    elif strategy == 'top3_mean':
        top3 = np.sort(probs)[-3:]
        return np.mean(top3)
    elif strategy == 'p90':
        return np.percentile(probs, 90)
    elif strategy == 'confidence_weighted':
        weights = np.abs(probs - 0.5)
        return np.average(probs, weights=weights)
    elif strategy == 'entropy_weighted':
        entropy = -(probs * np.log(probs + 1e-10) + (1 - probs) * np.log(1 - probs + 1e-10))
        weights = 1 / (entropy + 1e-10)
        return np.average(probs, weights=weights)

AGGREGATION_STRATEGIES = [
    'mean', 'max', 'majority_vote', 'top3_mean',
    'p90', 'confidence_weighted', 'entropy_weighted'
]
```

After threshold and aggregation strategy selection, the training and validation sets are merged and the final model is retrained using the best hyperparameters found during optimization. Sigmoid calibration is then applied before test set evaluation.

## Preprocessing

To prevent data leakage and ensure a realistic evaluation of my ML models, it is critical that all preprocessing steps are fitted exclusively on the training portion of the data and then applied to the validation and test sets.

The preprocessing function:

```python
def preprocess(X_train, X_val, X_test, cols):
    # Median imputation
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=cols)
    X_val   = pd.DataFrame(imputer.transform(X_val), columns=cols)
    X_test  = pd.DataFrame(imputer.transform(X_test), columns=cols)

    # Low-variance filtering
    var = VarianceThreshold(threshold=0.01)
    var.fit(X_train)
    cols = [c for c, k in zip(cols, var.get_support()) if k]
    X_train = X_train[cols]
    X_val   = X_val[cols]
    X_test  = X_test[cols]

    # High-correlation filtering
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    cols = [c for c in cols if c not in to_drop]
    X_train = X_train[cols]
    X_val   = X_val[cols]
    X_test  = X_test[cols]

    # RobustScaler
    scaler = RobustScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
    X_val   = pd.DataFrame(scaler.transform(X_val), columns=cols)
    X_test  = pd.DataFrame(scaler.transform(X_test), columns=cols)

    return X_train, X_val, X_test, cols
```

Prior to imputation, all infinite values in the feature matrix were replaced with `NaN`. A total of 1 infinite value was detected across the dataset.

```python
feature_cols = [col for col in df.columns if col.startswith('Feature_')]
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
```

A complete preprocessing summary is shown below:

| Step | Method | Result |
|------|--------|--------|
| Infinite value handling | Replace with NaN | 1 value fixed |
| Missing value imputation | Median imputation (train only) | 0 remaining NaN |
| Low-variance filtering | VarianceThreshold (threshold=0.01) | 746 → ~419 features |
| High-correlation filtering | Pearson correlation > 0.95 | ~419 → ~147 features |
| Feature scaling | RobustScaler (train only) | Median ≈ 0.0 |

## Feature Selection

Minimum Redundancy Maximum Relevance (MRMR) is a supervised, filter-based feature selection algorithm designed to find the most useful features for a target variable while reducing feature overlap.

The core objective of MRMR is to balance two competing criteria when selecting features:
* Maximum Relevance that selects features with high mutual information with the target class
* Minimum Redundancy that ensures selected features share low mutual information with each other





















