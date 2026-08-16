#!/usr/bin/env python3
"""
ML Assignment 2: Model Training Script
Trains all 5 classification models and generates output files
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ML ASSIGNMENT 2: MODEL TRAINING")
print("="*70)

# Step 1: Load Dataset
print("\n[1/5] Loading Breast Cancer Wisconsin Dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(f"✓ Dataset loaded successfully!")
print(f"  - Features: {X.shape[1]}")
print(f"  - Instances: {X.shape[0]}")
print(f"  - Classes: 2 (Benign: {sum(y==1)}, Malignant: {sum(y==0)})")

# Step 2: Split and Scale Data
print("\n[2/5] Preprocessing Data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train-test split completed (80-20)")
print(f"  - Training samples: {X_train.shape[0]}")
print(f"  - Test samples: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✓ Feature scaling completed")

# Step 3: Train Models
print("\n[3/5] Training Classification Models...")

models = {}
results = []

# 1. Logistic Regression
print("\n  Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=10000)
lr_model.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr_model
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
results.append({
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, lr_pred),
    'AUC': roc_auc_score(y_test, lr_proba),
    'Precision': precision_score(y_test, lr_pred),
    'Recall': recall_score(y_test, lr_pred),
    'F1 Score': f1_score(y_test, lr_pred),
    'MCC': matthews_corrcoef(y_test, lr_pred)
})
print(f"  ✓ Logistic Regression: {results[-1]['Accuracy']:.4f} accuracy")

# 2. Decision Tree
print("  Training Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train_scaled, y_train)
models['Decision Tree'] = dt_model
dt_pred = dt_model.predict(X_test_scaled)
dt_proba = dt_model.predict_proba(X_test_scaled)[:, 1]
results.append({
    'Model': 'Decision Tree',
    'Accuracy': accuracy_score(y_test, dt_pred),
    'AUC': roc_auc_score(y_test, dt_proba),
    'Precision': precision_score(y_test, dt_pred),
    'Recall': recall_score(y_test, dt_pred),
    'F1 Score': f1_score(y_test, dt_pred),
    'MCC': matthews_corrcoef(y_test, dt_pred)
})
print(f"  ✓ Decision Tree: {results[-1]['Accuracy']:.4f} accuracy")

# 3. K-Nearest Neighbor
print("  Training K-Nearest Neighbor Classifier...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
models['K-Nearest Neighbor'] = knn_model
knn_pred = knn_model.predict(X_test_scaled)
knn_proba = knn_model.predict_proba(X_test_scaled)[:, 1]
results.append({
    'Model': 'K-Nearest Neighbor',
    'Accuracy': accuracy_score(y_test, knn_pred),
    'AUC': roc_auc_score(y_test, knn_proba),
    'Precision': precision_score(y_test, knn_pred),
    'Recall': recall_score(y_test, knn_pred),
    'F1 Score': f1_score(y_test, knn_pred),
    'MCC': matthews_corrcoef(y_test, knn_pred)
})
print(f"  ✓ K-Nearest Neighbor: {results[-1]['Accuracy']:.4f} accuracy")

# 4. Naive Bayes
print("  Training Naive Bayes Classifier...")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
models['Naive Bayes'] = nb_model
nb_pred = nb_model.predict(X_test_scaled)
nb_proba = nb_model.predict_proba(X_test_scaled)[:, 1]
results.append({
    'Model': 'Naive Bayes',
    'Accuracy': accuracy_score(y_test, nb_pred),
    'AUC': roc_auc_score(y_test, nb_proba),
    'Precision': precision_score(y_test, nb_pred),
    'Recall': recall_score(y_test, nb_pred),
    'F1 Score': f1_score(y_test, nb_pred),
    'MCC': matthews_corrcoef(y_test, nb_pred)
})
print(f"  ✓ Naive Bayes: {results[-1]['Accuracy']:.4f} accuracy")

# 5. Random Forest
print("  Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
models['Random Forest'] = rf_model
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
results.append({
    'Model': 'Random Forest',
    'Accuracy': accuracy_score(y_test, rf_pred),
    'AUC': roc_auc_score(y_test, rf_proba),
    'Precision': precision_score(y_test, rf_pred),
    'Recall': recall_score(y_test, rf_pred),
    'F1 Score': f1_score(y_test, rf_pred),
    'MCC': matthews_corrcoef(y_test, rf_pred)
})
print(f"  ✓ Random Forest: {results[-1]['Accuracy']:.4f} accuracy")

# Step 4: Save Models and Results
print("\n[4/5] Saving Models and Results...")

# Create models directory
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"✓ Created '{model_dir}' directory")

# Save models
for model_name, model_obj in models.items():
    filename = os.path.join(model_dir, f'{model_name.lower().replace("-", "_").replace(" ", "_")}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(model_obj, f)
    print(f"✓ Saved: {filename}")

# Save scaler
scaler_filename = os.path.join(model_dir, 'scaler.pkl')
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Saved: {scaler_filename}")

# Save results
results_df = pd.DataFrame(results).round(4)
results_df.to_csv('model_results.csv', index=False)
print(f"✓ Saved: model_results.csv")

# Step 5: Generate Test Data and Visualizations
print("\n[5/5] Generating Test Data and Visualizations...")

# Save test data with predictions
test_data_output = X_test.copy()
test_data_output['True_Label'] = y_test.values
test_data_output['Logistic_Regression'] = lr_pred
test_data_output['Decision_Tree'] = dt_pred
test_data_output['KNN'] = knn_pred
test_data_output['Naive_Bayes'] = nb_pred
test_data_output['Random_Forest'] = rf_pred

test_data_output.to_csv('test_data.csv', index=False)
print(f"✓ Saved: test_data.csv ({len(test_data_output)} test samples)")

# Create comparison chart
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    bars = ax.bar(results_df['Model'], results_df[metric], color='steelblue', alpha=0.7)
    ax.set_title(metric, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim([0.85, 1.01])
    ax.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: model_comparison.png")
plt.close()

# Create confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')

predictions = {
    'Logistic Regression': lr_pred,
    'Decision Tree': dt_pred,
    'K-Nearest Neighbor': knn_pred,
    'Naive Bayes': nb_pred,
    'Random Forest': rf_pred
}

for idx, (model_name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx // 3, idx % 3]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(model_name, fontweight='bold')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: confusion_matrices.png")
plt.close()

# Print summary
print("\n" + "="*70)
print("MODEL PERFORMANCE SUMMARY")
print("="*70)
print(results_df.to_string(index=False))

best_idx = results_df['Accuracy'].idxmax()
best_model = results_df.loc[best_idx, 'Model']
best_accuracy = results_df.loc[best_idx, 'Accuracy']

print("\n" + "="*70)
print(f"🏆 BEST MODEL: {best_model}")
print(f"   Accuracy: {best_accuracy:.4f}")
print("="*70)

print("\n✅ All tasks completed successfully!")
print("\nFiles generated:")
print(f"  - 5 trained models in '{model_dir}/' directory")
print(f"  - test_data.csv (114 samples)")
print(f"  - model_results.csv (metrics summary)")
print(f"  - model_comparison.png (visualization)")
print(f"  - confusion_matrices.png (confusion matrices)")

print("\nNext steps:")
print("  1. Run: streamlit run app.py")
print("  2. Upload test_data.csv in the web interface")
print("  3. Select models and run predictions")
print("  4. Push code to GitHub")
print("  5. Deploy to Streamlit Cloud")
