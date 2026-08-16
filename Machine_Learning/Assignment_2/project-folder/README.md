# ML Assignment 2: Classification Models & Streamlit Deployment

**Student Information**
- **Student ID:** 2025AC05684
- **Name:** S Shobanaa
- **Course:** Machine Learning (ML)
- **Institution:** BITS Pilani

## Submission Links and Evidence

- **GitHub Repository Link:** [Submission](https://github.com/SShobanaa/bits/tree/submission/Machine_Learning/Assignment_2/project-folder)
- **Live Streamlit Deployment Link:** [Breast Cancer Prediction - Classification Models Evaluation](https://fo7zbv7gpnyykmdm8oqsvq.streamlit.app/)
- **BITS Virtual Lab Screenshot:** [BITS Virtual Lab execution screenshot](https://github.com/SShobanaa/bits/edit/submission/Machine_Learning/Assignment_2/project-folder/screenshots)

## Problem Statement

This project builds and compares five classification models for the Breast Cancer Wisconsin Diagnostic dataset. The models predict whether a tumor is malignant or benign from 30 numeric measurements derived from digitized fine needle aspirate images of breast masses.

The project objectives are to:
- Implement five classification algorithms.
- Evaluate them with Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Compare their predictions on a held-out test set.
- Deploy the evaluation workflow as an interactive Streamlit application.

## Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset

**Source:** UCI Machine Learning Repository, imported through `sklearn.datasets.load_breast_cancer()`.

| Characteristic | Value |
|---|---|
| Instances | 569 |
| Features | 30 numeric features |
| Classes | 2 |
| Class distribution | Benign: 357 (62.7%), Malignant: 212 (37.3%) |
| Missing values | None |
| Train/test split | 455 training samples / 114 test samples |
| Scaling | `StandardScaler` |

### Class Labels

The scikit-learn dataset uses:
- `0 = Malignant (cancerous)`
- `1 = Benign (non-cancerous)`

For clinical screening, malignant-class recall and false negatives should receive particular attention.

### Features

The 30 measurements are grouped into mean, standard-error, and worst-value versions of ten characteristics: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

## Models Used

### Logistic Regression

A scaled linear probabilistic classifier. It is fast, interpretable, and performs well when the classes are close to linearly separable. The saved model uses `max_iter=10000` and `random_state=42`.

### Decision Tree

A rule-based model that recursively partitions the feature space. It is easy to explain and captures non-linear relationships, but a single tree can be sensitive to training-data variation. The model uses `max_depth=10` and `random_state=42`.

### K-Nearest Neighbor

A distance-based classifier that assigns a class using the nearest training samples. It uses `k=5` and requires feature scaling so that large-valued features do not dominate the distance calculation.

### Gaussian Naive Bayes

A probabilistic classifier that assumes conditional independence between continuous features and models each feature with a Gaussian distribution. It is fast and produces probability estimates.

### Random Forest

An ensemble of decision trees trained with bagging and random feature selection. It can model feature interactions and reduce the variance of a single decision tree. The saved model uses 100 estimators, `random_state=42`, and parallel processing.

## Model Performance Comparison

### Metric Definitions

| Metric | Meaning in this task |
|---|---|
| Accuracy | Proportion of all malignant and benign test samples classified correctly. |
| AUC | How well the model separates the two classes across decision thresholds. |
| Precision | For the default positive class (`1`, benign), the proportion of predicted benign samples that are actually benign. |
| Recall | For the default positive class (`1`, benign), the proportion of actual benign samples identified by the model. |
| F1 Score | Harmonic mean of precision and recall for the default positive class. |
| MCC | Correlation between predicted and actual labels using all confusion-matrix outcomes; useful when class sizes differ. |

> **Important:** `scikit-learn` reports precision, recall, and F1 Score for class `1` unless `pos_label` is changed. Since `0` is malignant in this dataset, inspect the malignant row of the classification report and confusion matrix when evaluating cancer-screening risk.

### Verified Test-Set Results

The following values are from `model_results.csv` for the 114-row test set.

| Rank | ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **Logistic Regression** | **0.9825** | **0.9954** | **0.9861** | **0.9861** | **0.9861** | **0.9623** |
| 2 | K-Nearest Neighbor | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| 3 | Random Forest | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| 4 | Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| 5 | Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |

## Observations & Analysis

The table below is the assignment observation table. It explains what each model result signifies for this dataset rather than only listing metric definitions.

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Best performer on the Breast Cancer Wisconsin test dataset, with **98.25% accuracy**, **99.54% AUC**, and **0.9623 MCC**. Precision and recall for class `1` (benign) are both **98.61%**, showing highly consistent predictions. Its strong result suggests that the scaled measurements support an effective linear decision boundary. |
| **K-Nearest Neighbor** | Achieves **95.61% accuracy** and **97.88% AUC**. Recall for class `1` (benign) is **97.22%**, higher than precision at **95.89%**, so it identifies most benign samples but produces some additional false-positive benign predictions. Its performance depends on the StandardScaler preprocessing because KNN uses distances. |
| **Random Forest (Ensemble)** | Tied for second by accuracy at **95.61%** and achieves a strong **99.39% AUC**, indicating good separation across thresholds. Its F1 Score (**0.9655**) and MCC (**0.9054**) are below Logistic Regression on this test split, although the ensemble remains useful for non-linear feature interactions and feature-importance analysis. |
| **Naive Bayes** | Achieves **92.98% accuracy** and **98.68% AUC**. Its high AUC indicates good ranking ability, while equal precision and recall of **94.44%** show a balanced error profile for class `1`. The lower accuracy may reflect the model's feature-independence assumption. |
| **Decision Tree** | Lowest accuracy at **91.23%**, lowest AUC at **91.57%**, and lowest MCC at **0.8174**. Its class-`1` recall is **90.28%**, so it misses more samples from that class than the other models. It remains easy to interpret, but a single tree is less effective for this test split. |
| **Overall Winner for this Dataset** | **Logistic Regression** is the overall winner because it has the highest accuracy (**98.25%**), F1 Score (**0.9861**), and MCC (**0.9623**), with excellent AUC (**99.54%**) and balanced precision/recall. Before clinical use, malignant-class recall and the cost of false negatives must also be evaluated explicitly. |

### Recommendation

For this test split, Logistic Regression is the strongest overall choice. Random Forest is a useful alternative when non-linear feature interactions or feature-importance analysis are more important. Neither model should be used for clinical diagnosis without validation on an independent dataset and an evaluation focused explicitly on malignant-case false negatives.

## Streamlit Application Features

- Upload a CSV test dataset or automatically evaluate the bundled `test_data.csv`.
- Select one or more trained models.
- Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Show a confusion matrix with labeled TN, FP, FN, and TP cells.
- Show a per-class classification report.
- Compare selected models in a ranked table and performance graph.
- Display predictions and class probabilities for the first ten test samples.

## Repository Structure

```text
Machine_Learning/Assignment_2/project-folder
├── app.py
├── requirements.txt
├── README.md
├── ML_Classification_Models.ipynb
├── test_data.csv
├── model_results.csv
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── k_nearest_neighbor.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
```

## Installation and Setup

Prerequisites: Python 3.8 or later and `pip`.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The trained model files must be present in the `models/` directory. To retrain them, run the relevant cells in `ML_Classification_Models.ipynb`.

## How to Run

From the `Machine_Learning/Assignment_2/project-folder` directory:

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`.

The bundled `test_data.csv` is used automatically when no file is uploaded. To evaluate another dataset, upload a CSV containing the same 30 feature columns and optionally a `True_Label` column with values `0` or `1`.

## Deployment

The application can be deployed to Streamlit Community Cloud with `app.py`, `requirements.txt`, the `models/` directory, and `test_data.csv` committed to the repository. Configure the app entry point as `app.py`.

## References

1. [UCI Breast Cancer Wisconsin Diagnostic Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
2. [Scikit-learn documentation](https://scikit-learn.org/)
3. [Streamlit documentation](https://docs.streamlit.io/)
4. [Scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
