import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class CompatibleUnpickler(pickle.Unpickler):
    """Unpickler that maps NumPy 2.x module paths to NumPy 1.x paths."""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def normalize_loaded_model(model):
    """Patch known missing attrs for cross-version sklearn pickles."""
    if model.__class__.__name__ == 'LogisticRegression' and not hasattr(model, 'multi_class'):
        model.multi_class = 'auto'
    return model


def generate_model_observation(model_name, accuracy, precision, recall, f1, auc, mcc):
    """Generate detailed observation text based on model metrics with quality levels."""
    observations = []
    
    # Accuracy observation
    if accuracy >= 0.95:
        observations.append(("Excellent Accuracy ({:.2%}): Model demonstrates outstanding overall correctness in predictions.".format(accuracy), 'good'))
    elif accuracy >= 0.85:
        observations.append(("Strong Accuracy ({:.2%}): Model shows reliable performance with high correctness rate.".format(accuracy), 'good'))
    elif accuracy >= 0.75:
        observations.append(("Good Accuracy ({:.2%}): Model performs reasonably well but has room for improvement.".format(accuracy), 'neutral'))
    else:
        observations.append(("Moderate Accuracy ({:.2%}): Model needs refinement for better performance.".format(accuracy), 'poor'))
    
    # Precision observation
    if precision >= 0.9:
        observations.append(("High Precision ({:.2%}): Minimal false positives; when model predicts positive, it's highly reliable.".format(precision), 'good'))
    elif precision >= 0.75:
        observations.append(("Good Precision ({:.2%}): Model maintains reasonable confidence in positive predictions.".format(precision), 'good'))
    else:
        observations.append(("Precision Needs Attention ({:.2%}): Model produces several false positives; consider adjusting decision threshold.".format(precision), 'poor'))
    
    # Recall observation
    if recall >= 0.9:
        observations.append(("Excellent Recall ({:.2%}): Model captures most actual positive cases; minimal false negatives.".format(recall), 'good'))
    elif recall >= 0.75:
        observations.append(("Good Recall ({:.2%}): Model identifies majority of positive cases effectively.".format(recall), 'good'))
    else:
        observations.append(("Recall Needs Improvement ({:.2%}): Model misses several actual positive cases; may require retraining.".format(recall), 'poor'))
    
    # Precision-Recall trade-off
    diff = abs(precision - recall)
    if diff < 0.05:
        observations.append(("Balanced Precision-Recall: Model maintains good balance between precision and recall (diff: {:.2%}).".format(diff), 'good'))
    elif precision > recall:
        observations.append(("Precision > Recall: Model prioritizes avoiding false positives over catching all positives.", 'neutral'))
    else:
        observations.append(("Recall > Precision: Model prioritizes catching positive cases over avoiding false positives.", 'neutral'))
    
    # F1 Score observation
    if f1 >= 0.9:
        observations.append(("Outstanding F1 Score ({:.4f}): Excellent harmonic balance of precision and recall.".format(f1), 'good'))
    elif f1 >= 0.8:
        observations.append(("Strong F1 Score ({:.4f}): Model demonstrates well-rounded performance.".format(f1), 'good'))
    else:
        observations.append(("F1 Score ({:.4f}): Consider optimizing for better precision-recall balance.".format(f1), 'neutral'))
    
    # AUC observation
    if auc >= 0.95:
        observations.append(("Outstanding AUC ({:.4f}): Excellent discrimination ability across all classification thresholds.".format(auc), 'good'))
    elif auc >= 0.85:
        observations.append(("Strong AUC ({:.4f}): Model shows good discrimination capability.".format(auc), 'good'))
    elif auc >= 0.7:
        observations.append(("Fair AUC ({:.4f}): Model has acceptable discrimination; further tuning recommended.".format(auc), 'neutral'))
    else:
        observations.append(("Poor AUC ({:.4f}): Model discrimination capability is below acceptable threshold.".format(auc), 'poor'))
    
    # MCC observation
    if mcc >= 0.8:
        observations.append(("Excellent MCC ({:.4f}): Strongest correlation coefficient; model shows excellent performance on imbalanced data.".format(mcc), 'good'))
    elif mcc >= 0.6:
        observations.append(("Good MCC ({:.4f}): Reliable performance on imbalanced datasets.".format(mcc), 'good'))
    else:
        observations.append(("MCC ({:.4f}): Performance on imbalanced data needs improvement.".format(mcc), 'neutral'))
    
    return observations


def generate_model_comparison_observation(model_name, rank, total_models, accuracy, precision, recall, f1):
    """Generate observation for model comparison table."""
    if rank == 1:
        return ("Best performer with {:.2%} accuracy. Excellent precision ({:.2%}) and recall ({:.2%}). Recommended for deployment.".format(accuracy, precision, recall), 'good')
    elif rank == 2:
        return ("Strong performer. Close to best model with {:.2%} accuracy. Consider for backup or ensemble approaches.".format(accuracy), 'good')
    elif rank <= total_models - 1:
        return ("Moderate performance. {:.2%} accuracy. May require hyperparameter tuning or feature engineering.".format(accuracy), 'neutral')
    else:
        return ("Least effective on this dataset. {:.2%} accuracy. Consider revisiting model selection or training approach.".format(accuracy), 'poor')


def render_responsive_table(dataframe, formatters=None):
    """Render a readable table with wrapping and consistent left-middle alignment."""
    formatted_df = dataframe.copy()
    for column, formatter in (formatters or {}).items():
        if column in formatted_df.columns:
            formatted_df[column] = formatted_df[column].map(formatter)

    table_html = formatted_df.to_html(index=False, classes='app-table', border=0, escape=True)
    st.markdown(
        f"<div class='app-table-wrap'>{table_html}</div>",
        unsafe_allow_html=True
    )

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Model Evaluation",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .model-header {
        color: #1f77b4;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    /* Use blue chips for selected models in the multiselect control. */
    [data-baseweb="select"] [data-baseweb="tag"] {
        background-color: #0066cc !important;
        color: #ffffff !important;
    }
    [data-baseweb="select"] [data-baseweb="tag"] span {
        color: #ffffff !important;
    }
    [data-baseweb="select"] [data-baseweb="tag"] svg {
        fill: #ffffff !important;
    }
    /* Keep the informational panel visually separate from prediction tabs. */
    [data-testid="column"] {
        min-width: 0;
    }
    .info-panel-note {
        color: #8b949e;
        font-size: 0.85rem;
    }
    .info-panel {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(31, 119, 180, 0.18), rgba(31, 119, 180, 0.04));
        border: 1px solid rgba(80, 160, 220, 0.32);
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 104px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
    }
    [data-testid="stMetricLabel"] {
        color: #a9c9e8;
        font-size: 0.82rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #f4f8fb;
        font-size: 1.65rem;
        font-weight: 700;
    }
    .result-section-title {
        color: #dbeafe;
        font-size: 1.05rem;
        font-weight: 700;
        border-left: 4px solid #1f77b4;
        padding-left: 10px;
        margin: 18px 0 12px;
    }
    .comparison-title {
        color: #dbeafe;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 16px 0 12px;
    }
    .observation-table {
        background: linear-gradient(135deg, rgba(30, 144, 255, 0.08), rgba(100, 149, 237, 0.08));
        border: 1px solid rgba(30, 144, 255, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 16px 0;
    }
    .observation-header {
        color: #0066cc;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .interpretation-note {
        background: linear-gradient(135deg, rgba(65, 105, 225, 0.06), rgba(30, 144, 255, 0.06));
        border-left: 4px solid #4169e1;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 12px 0;
        color: #1e3a8a;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .obs-table-header {
        background: linear-gradient(90deg, #0066cc, #4169e1);
        color: white;
        font-weight: 700;
        padding: 10px;
        border-radius: 6px 6px 0 0;
    }
    .obs-table-row-model {
        background: linear-gradient(90deg, rgba(30, 144, 255, 0.04), rgba(100, 149, 237, 0.04));
        border-bottom: 1px solid rgba(30, 144, 255, 0.2);
        padding: 10px;
    }
    .obs-table-row-obs {
        background: linear-gradient(90deg, rgba(65, 105, 225, 0.03), rgba(30, 144, 255, 0.03));
        border-bottom: 1px solid rgba(65, 105, 225, 0.15);
        padding: 10px;
        color: #1e3a8a;
        line-height: 1.4;
    }
    .metric-interpretation {
        background: linear-gradient(135deg, rgba(30, 144, 255, 0.05), rgba(100, 149, 237, 0.05));
        border-radius: 8px;
        padding: 10px 12px;
        margin: 6px 0;
        font-size: 0.9rem;
        color: #1e3a8a;
    }
    .good-observation {
        color: #0d6e2f;
        font-weight: bold;
        background-color: #d1f2eb;
        padding: 12px 16px;
        border-radius: 6px;
        border-left: 4px solid #0d6e2f;
        margin: 8px 0;
        line-height: 1.5;
    }
    .neutral-observation {
        color: #1e3a8a;
        background-color: #e8f4f8;
        padding: 12px 16px;
        border-radius: 6px;
        border-left: 4px solid #4169e1;
        margin: 8px 0;
        line-height: 1.5;
    }
    .poor-observation {
        color: #7c2d12;
        background-color: #fed7aa;
        padding: 12px 16px;
        border-radius: 6px;
        border-left: 4px solid #ea580c;
        margin: 8px 0;
        line-height: 1.5;
    }
    [data-testid="stDataframe"] {
        width: 100% !important;
        overflow-x: auto !important;
    }
    [data-testid="stDataframe"] div {
        word-wrap: break-word !important;
        white-space: normal !important;
    }
    [data-testid="stDataframe"] table {
        table-layout: auto !important;
        width: 100% !important;
    }
    [data-testid="stDataframe"] td {
        word-break: break-word !important;
        white-space: pre-wrap !important;
        max-width: 800px;
        overflow-wrap: anywhere !important;
        padding: 12px !important;
    }
    [data-testid="stDataframe"] th {
        word-break: break-word !important;
        white-space: pre-wrap !important;
        overflow-wrap: anywhere !important;
        padding: 12px !important;
    }
    .summary-table-wrap {
        width: 100%;
        overflow-x: auto;
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 12px;
    }
    .summary-table {
        width: 100%;
        min-width: 720px;
        border-collapse: collapse;
        table-layout: auto;
    }
    .summary-table th,
    .summary-table td {
        padding: 12px 14px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.28);
        text-align: left;
        vertical-align: top;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: normal;
        line-height: 1.45;
    }
    .summary-table th {
        background-color: rgba(255, 255, 255, 0.05);
        color: #b8bec9;
        font-weight: 600;
    }
    .summary-table th:nth-child(1),
    .summary-table td:nth-child(1) {
        width: 18%;
    }
    .summary-table th:nth-child(2),
    .summary-table td:nth-child(2) {
        width: 12%;
    }
    .summary-table th:nth-child(3),
    .summary-table td:nth-child(3) {
        width: 70%;
    }
    .summary-table .summary-value {
        color: #ffffff;
        font-weight: 600;
        white-space: nowrap;
    }
    .app-table-wrap {
        width: 100%;
        overflow-x: auto;
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 12px;
    }
    .app-table {
        width: 100%;
        min-width: 720px;
        border-collapse: collapse;
        table-layout: auto;
    }
    .app-table th,
    .app-table td {
        padding: 12px 14px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.28);
        text-align: left !important;
        vertical-align: middle !important;
        white-space: normal !important;
        overflow-wrap: anywhere;
        word-break: normal;
        line-height: 1.45;
    }
    .app-table th {
        background-color: rgba(255, 255, 255, 0.05);
        color: #b8bec9;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    "<h1 class='main-title'>🩺 Breast Cancer Prediction - Classification Models Evaluation</h1>",
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Classification Models Evaluation")
st.sidebar.markdown("---")

# Project Information
with st.sidebar.expander("ℹ️ About", expanded=False):
    st.write("""
    This application demonstrates classification models trained on the Breast Cancer Wisconsin dataset.

    **Problem Type:** Binary Classification

    **Dataset Details:**
    - Features: 30
    - Instances: 569
    - Classes: 2 (Malignant, Benign)
    - Train/Test Split: 80/20

    **Metrics Calculated:**
    - Accuracy
    - AUC Score
    - Precision
    - Recall
    - F1 Score
    - MCC (Matthews Correlation Coefficient)

    **Models Available:**
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbor
    - Naive Bayes
    - Random Forest
    """)

with st.sidebar.expander("📋 Information", expanded=False):
    st.write("""
    **Assignment:** Machine Learning Assignment 2
    
    **Student ID:** 2025AC05684
    
    **Name:** S Shobanaa
    """)

st.sidebar.markdown("---")

# Load models and scaler
@st.cache_resource
def load_models_and_scaler():
    """Load all trained models and scaler."""
    try:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        models = {}
        
        # Load models
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Decision Tree': 'decision_tree.pkl',
            'K-Nearest Neighbor': 'k_nearest_neighbor.pkl',
            'Naive Bayes': 'naive_bayes.pkl',
            'Random Forest': 'random_forest.pkl'
        }
        
        for model_name, file_name in model_files.items():
            file_path = os.path.join(model_dir, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    models[model_name] = normalize_loaded_model(CompatibleUnpickler(f).load())
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = CompatibleUnpickler(f).load()
        else:
            scaler = None
        
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}, None

# Load models
models, scaler = load_models_and_scaler()

if not models:
    st.error("❌ Models not found. Please ensure 'models' directory with trained models exists.")
    st.info("Run the Jupyter notebook first to train and save the models.")
    st.stop()

# Feature names for Breast Cancer dataset
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
    'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Main content uses the full available page width; project information is in the sidebar.
col1 = st.container()

with col1:
    st.subheader("📤 Dataset Upload & Model Selection")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload test data (CSV format)",
        type="csv",
        help="Upload a CSV file with the same features as the training data"
    )

    default_test_data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_data.csv'
    )
    using_default_test_data = uploaded_file is None and os.path.exists(default_test_data_path)
    data_source = default_test_data_path if using_default_test_data else uploaded_file

    with st.expander("Expected Data Format", expanded=False):
        st.write("""
        The uploaded CSV should contain:
        - **30 features** from the Breast Cancer Wisconsin dataset
        - Optional: `True_Label` column with actual class labels (0 or 1)

        **Features:**
        """ + ", ".join(feature_names[:10]) + ", ... and 20 more")

        st.write("""
        **Sample structure:**
        """)
        sample_df = pd.DataFrame(
            np.random.randn(5, 30),
            columns=feature_names
        )
        sample_df['True_Label'] = np.random.randint(0, 2, 5)
        render_responsive_table(sample_df.head())
    
    if data_source is not None:
        try:
            # Load uploaded data
            data = pd.read_csv(data_source)
            if using_default_test_data:
                st.info(f"Using bundled test data. Shape: {data.shape}")
            else:
                st.success(f"✅ File uploaded successfully! Shape: {data.shape}")
            
            # Check if data has true labels
            has_true_labels = 'True_Label' in data.columns
            
            # Prepare data for prediction
            if has_true_labels:
                X_data = data.drop(['True_Label', *[col for col in data.columns if col.endswith('_Label') or col in ['Logistic_Regression', 'Decision_Tree', 'KNN', 'Naive_Bayes', 'Random_Forest']]], axis=1)
                y_true = data['True_Label']
            else:
                X_data = data
                y_true = None
            
            # Check if all required features are present
            available_features = [f for f in feature_names if f in X_data.columns]
            
            if len(available_features) < 30:
                st.warning(f"⚠️ Warning: Only {len(available_features)}/30 features found in the data.")
            
            # Scale the data
            if scaler is not None:
                X_scaled = scaler.transform(X_data[available_features])
            else:
                st.warning("⚠️ Scaler not found. Using unscaled data.")
                X_scaled = X_data[available_features].values
            
            # Model selection
            st.markdown("---")
            st.subheader("🔧 Model Selection")
            
            selected_models = st.multiselect(
                "Select one or more models to evaluate:",
                list(models.keys()),
                default=list(models.keys()),
                help="Choose models to make predictions"
            )
            
            run_predictions = st.button("🚀 Run Predictions", key="predict_button")
            if selected_models and (run_predictions or using_default_test_data):
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Create a comparison tab only when multiple models are selected.
                # Model Comparison tab comes first if multiple models are selected
                if len(selected_models) > 1:
                    tab_labels = ["🏆 Model Comparison"] + selected_models
                else:
                    tab_labels = selected_models
                tabs = st.tabs(tab_labels)
                
                all_results = []
                
                # Determine the starting index for model tabs (skip first tab if it's comparison)
                model_tab_start_index = 1 if len(selected_models) > 1 else 0
                
                for tab_idx, model_name in enumerate(selected_models):
                    tab = tabs[model_tab_start_index + tab_idx]
                    with tab:
                        try:
                            st.subheader(f"➡️ {model_name}")
                            model = normalize_loaded_model(models[model_name])

                            # Make predictions
                            y_pred = model.predict(X_scaled)

                            # Get prediction probabilities if available
                            if hasattr(model, 'predict_proba'):
                                y_proba = model.predict_proba(X_scaled)[:, 1]
                            else:
                                y_proba = model.decision_function(X_scaled)

                            # Prepare predictions for the summary table shown at the bottom of the model tab.
                            predictions_df = pd.DataFrame({
                                'Test Sample': range(1, len(y_pred) + 1),
                                'Predicted Class': y_pred,
                                'Positive-Class Probability': y_proba
                            })

                            # If true labels available, calculate metrics
                            if y_true is not None:
                                st.markdown(
                                    "<div class='result-section-title'>📈 Model Evaluation Metrics</div>",
                                    unsafe_allow_html=True
                                )

                                # Calculate metrics
                                accuracy = accuracy_score(y_true, y_pred)
                                precision = precision_score(y_true, y_pred, zero_division=0)
                                recall = recall_score(y_true, y_pred, zero_division=0)
                                f1 = f1_score(y_true, y_pred, zero_division=0)
                                mcc = matthews_corrcoef(y_true, y_pred)

                                # Calculate AUC
                                try:
                                    if hasattr(model, 'predict_proba'):
                                        auc = roc_auc_score(y_true, y_proba)
                                    else:
                                        auc = roc_auc_score(y_true, model.decision_function(X_scaled))
                                except:
                                    auc = 0.0

                                # Display metrics as visual score cards
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Accuracy", f"{accuracy:.4f}", help="Overall correctness")
                                with col2:
                                    st.metric("AUC Score", f"{auc:.4f}", help="Area under ROC curve")
                                with col3:
                                    st.metric("F1 Score", f"{f1:.4f}", help="Harmonic mean of precision & recall")

                                col4, col5, col6 = st.columns(3)
                                with col4:
                                    st.metric("Precision", f"{precision:.4f}", help="True positives / predicted positives")
                                with col5:
                                    st.metric("Recall", f"{recall:.4f}", help="True positives / actual positives")
                                with col6:
                                    st.metric("MCC", f"{mcc:.4f}", help="Matthews Correlation Coefficient")

                                # Confusion Matrix
                                st.markdown("---")
                                st.markdown(
                                    "<div class='result-section-title'>🔲 Confusion Matrix</div>",
                                    unsafe_allow_html=True
                                )
                                with st.expander("About this confusion matrix", expanded=False):
                                    st.info(
                                        "This heatmap compares the model's predicted labels with the actual "
                                        "labels in the test data. Rows show the true class and columns show "
                                        "the predicted class. Values on the diagonal are correct predictions. "
                                        "For this dataset, class 0 is malignant and class 1 is benign; "
                                        "therefore, an actual malignant sample predicted as benign is a "
                                        "clinically important missed malignant case."
                                    )
                                cm = confusion_matrix(y_true, y_pred)

                                fig, ax = plt.subplots(figsize=(6, 4))
                                class_labels = ['Malignant (0)', 'Benign (1)']
                                cell_labels = np.array([
                                    [f'TN\n{cm[0, 0]}', f'FP\n{cm[0, 1]}'],
                                    [f'FN\n{cm[1, 0]}', f'TP\n{cm[1, 1]}']
                                ])
                                sns.heatmap(
                                    cm,
                                    annot=cell_labels,
                                    fmt='',
                                    cmap='Blues',
                                    ax=ax,
                                    cbar=False,
                                    xticklabels=class_labels,
                                    yticklabels=class_labels
                                )
                                ax.set_xlabel('Predicted Class')
                                ax.set_ylabel('Actual Class')
                                ax.set_title(f'Confusion Matrix - {model_name}')
                                st.pyplot(fig, use_container_width=True)
                                plt.close()

                                # Classification Report
                                st.markdown("---")
                                st.markdown(
                                    "<div class='result-section-title'>📋 Classification Report</div>",
                                    unsafe_allow_html=True
                                )
                                report_dict = classification_report(
                                    y_true,
                                    y_pred,
                                    output_dict=True,
                                    zero_division=0
                                )
                                report_df = pd.DataFrame(report_dict).T.reset_index()
                                report_df = report_df.rename(columns={
                                    'index': 'Class',
                                    'precision': 'Precision',
                                    'recall': 'Recall',
                                    'f1-score': 'F1 Score',
                                    'support': 'Support'
                                })
                                report_df['Class'] = report_df['Class'].replace({
                                    '0': 'Malignant (0)',
                                    '1': 'Benign (1)',
                                    'macro avg': 'Macro Average',
                                    'weighted avg': 'Weighted Average'
                                })
                                with st.expander("About this classification report", expanded=False):
                                    st.info(
                                        "This report shows precision, recall, F1 Score, and sample support "
                                        "for malignant (class 0) and benign (class 1) samples in the test "
                                        "dataset. Review the malignant row directly when assessing missed cancer cases."
                                    )
                                render_responsive_table(
                                    report_df,
                                    formatters={
                                        'Precision': lambda value: f'{value:.3f}',
                                        'Recall': lambda value: f'{value:.3f}',
                                        'F1 Score': lambda value: f'{value:.3f}',
                                        'Support': lambda value: f'{value:.0f}'
                                    }
                                )

                                # Model Interpretation & Observations
                                st.markdown("---")
                                with st.expander("📊 Model Interpretation & Observations", expanded=True):
                                    observations = generate_model_observation(
                                        model_name, accuracy, precision, recall, f1, auc, mcc
                                    )
                                    
                                    for obs_text, quality in observations:
                                        if quality == 'good':
                                            st.markdown(
                                                f"<div class='good-observation'>{obs_text}</div>",
                                                unsafe_allow_html=True
                                            )
                                        elif quality == 'poor':
                                            st.markdown(
                                                f"<div class='poor-observation'>{obs_text}</div>",
                                                unsafe_allow_html=True
                                            )
                                        else:
                                            st.markdown(
                                                f"<div class='neutral-observation'>{obs_text}</div>",
                                                unsafe_allow_html=True
                                            )
                                
                                # Observation Summary Table
                                st.markdown(
                                    "<div class='result-section-title'>📋 Observation Summary Table</div>",
                                    unsafe_allow_html=True
                                )
                                
                                obs_summary_data = {
                                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC Score', 'MCC'],
                                    'Value': [f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{auc:.4f}", f"{mcc:.4f}"],
                                    'Interpretation': [
                                        f"Correctly classifies {accuracy:.2%} of malignant and benign test samples; the remaining samples are misclassified.",
                                        f"Of the samples predicted as benign (class 1), {precision:.2%} are actually benign; a lower value would mean more malignant samples are incorrectly labeled benign.",
                                        f"Identifies {recall:.2%} of the benign samples in this test set. Review the malignant-class row separately because class 0 is the clinically important cancer class.",
                                        f"Combines precision and recall for the benign class into a single score of {f1:.2%}; this indicates consistency in the class-1 predictions.",
                                        f"Separates malignant and benign samples with {auc:.2%} discrimination quality across decision thresholds; 1.0 is perfect separation.",
                                        f"Shows {mcc:.2%} agreement between predictions and the actual malignant/benign labels after accounting for all confusion-matrix outcomes; 1.0 is perfect prediction."
                                    ]
                                }
                                obs_summary_df = pd.DataFrame(obs_summary_data)
                                with st.expander("About this metric summary table", expanded=False):
                                    st.info(
                                        "This table summarizes the model's scores on the Breast Cancer Wisconsin "
                                        "test dataset. The interpretation column explains what each score means "
                                        "for malignant and benign classification."
                                    )
                                table_rows = []
                                for row_index, row in obs_summary_df.iterrows():
                                    value = float(row['Value'])
                                    blue_level = max(0, min(255, int(255 - (value * 150))))
                                    table_rows.append(
                                        f"<tr><td>{row['Metric']}</td>"
                                        f"<td class='summary-value' style='background-color: rgb(0, {blue_level}, 180);'>"
                                        f"{row['Value']}</td>"
                                        f"<td>{row['Interpretation']}</td></tr>"
                                    )

                                st.markdown(
                                    "<div class='summary-table-wrap'><table class='summary-table'>"
                                    "<thead><tr><th>Metric</th><th>Value</th>"
                                    "<th>Dataset-Specific Interpretation</th></tr></thead><tbody>"
                                    + "".join(table_rows)
                                    + "</tbody></table></div>",
                                    unsafe_allow_html=True
                                )

                                # Store results for comparison
                                all_results.append({
                                    'Model': model_name,
                                    'Accuracy': accuracy,
                                    'AUC': auc,
                                    'Precision': precision,
                                    'Recall': recall,
                                    'F1 Score': f1,
                                    'MCC': mcc
                                })

                            st.markdown("---")
                            st.markdown(
                                "<div class='result-section-title'>🧾 Test-Sample Prediction Summary</div>",
                                unsafe_allow_html=True
                            )
                            with st.expander("About this prediction summary table", expanded=False):
                                st.info(
                                    "This table shows the first 10 test samples evaluated by "
                                    f"the {model_name} model. Predicted Class is 0 (Malignant) or "
                                    "1 (Benign). Positive-Class Probability is the model's "
                                    "confidence that the sample belongs to class 1."
                                )
                            prediction_display_df = predictions_df.head(10).rename(columns={
                                'Test Sample': 'Test Sample #',
                                'Predicted Class': 'Predicted Class (0/1)',
                                'Positive-Class Probability': 'Benign Probability'
                            })
                            render_responsive_table(
                                prediction_display_df,
                                formatters={
                                    'Test Sample #': lambda value: f'{int(value)}',
                                    'Predicted Class (0/1)': lambda value: f'{int(value)}',
                                    'Benign Probability': lambda value: f'{value:.4f}'
                                }
                            )
                        except Exception as model_error:
                            st.error(f"❌ {model_name} failed: {str(model_error)}")
                
                # Model Comparison tab
                if len(selected_models) > 1:
                    with tabs[0]:
                        st.markdown(
                            "<div class='comparison-title'>🏆 Model Comparison</div>",
                            unsafe_allow_html=True
                        )

                        if not all_results:
                            st.info("Comparison metrics require a True_Label column in the uploaded CSV.")
                        else:
                            comparison_df = pd.DataFrame(all_results).sort_values(
                                'Accuracy', ascending=False
                            ).reset_index(drop=True)
                            
                            # Add Rank column
                            comparison_df.insert(0, 'Rank', range(1, len(comparison_df) + 1))
                            
                            best_model = comparison_df.iloc[0]

                            st.success(
                                f"🏆 **Best Performing Model: {best_model['Model']}** "
                                f"with Accuracy: **{best_model['Accuracy']:.4f}**"
                            )

                            def highlight_best(row):
                                # Blue gradient for best model
                                if row.name == 0:
                                    return ['background: linear-gradient(90deg, #0066cc, #4169e1); color: white; font-weight: bold'] * len(row)
                                else:
                                    return ['background: linear-gradient(90deg, rgba(30, 144, 255, 0.08), rgba(100, 149, 237, 0.08))'] * len(row)

                            render_responsive_table(
                                comparison_df,
                                formatters={
                                    column: lambda value: f'{value:.4f}'
                                    for column in comparison_df.columns
                                    if column not in ['Model', 'Rank']
                                }
                            )

                            with st.expander("About this model comparison table", expanded=False):
                                st.info(
                                    "This table ranks the selected models by accuracy on the same test dataset. "
                                    "The remaining columns show how each model performs across discrimination, "
                                    "malignant-case detection, precision, balance, and correlation metrics."
                                )

                            # Visualization
                            st.markdown(
                                "<div class='result-section-title'>📈 Model Performance Comparison</div>",
                                unsafe_allow_html=True
                            )
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = ['#0066cc', '#1e90ff', '#4169e1', '#6495ed', '#87ceeb', '#add8e6']
                            comparison_df.set_index('Model')[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']].plot(
                                kind='bar', ax=ax, width=0.8, color=colors[:6]
                            )
                            with st.expander("About this model performance graph", expanded=False):
                                st.info(
                                    "This chart compares the selected models across six evaluation metrics "
                                    "on the same test dataset. Each bar represents a model-metric score; "
                                    "higher values indicate better classification performance, with 1.0 "
                                    "representing the maximum score."
                                )
                            ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', color='#0066cc')
                            ax.set_ylabel('Score', fontsize=11, color='#1e3a8a')
                            ax.set_xlabel('Model', fontsize=11, color='#1e3a8a')
                            ax.legend(
                                loc='upper left',
                                bbox_to_anchor=(1.02, 1),
                                borderaxespad=0,
                                fancybox=True,
                                shadow=True
                            )
                            fig.subplots_adjust(right=0.78, bottom=0.25)
                            ax.grid(axis='y', alpha=0.3, color='#4169e1')
                            plt.xticks(rotation=45)
                            st.pyplot(fig, use_container_width=True)
                            plt.close()

                            # Observation Table for Model Comparison
                            st.markdown("---")
                            st.markdown(
                                "<div class='result-section-title'>📊 Model Performance Observations</div>",
                                unsafe_allow_html=True
                            )
                            
                            obs_data = []
                            for idx, row in comparison_df.iterrows():
                                obs_text, quality = generate_model_comparison_observation(
                                    row['Model'], 
                                    idx + 1, 
                                    len(comparison_df),
                                    row['Accuracy'],
                                    row['Precision'],
                                    row['Recall'],
                                    row['F1 Score']
                                )
                                obs_data.append({
                                    'ML Model Name': row['Model'],
                                    'Observation about Model Performance': obs_text,
                                    'Quality': quality
                                })
                            
                            obs_df = pd.DataFrame(obs_data)
                            
                            # Display observations in a more readable format
                            for idx, row in obs_df.iterrows():
                                model_name = row['ML Model Name']
                                observation = row['Observation about Model Performance']
                                quality = row['Quality']
                                
                                # Create a styled observation display
                                if quality == 'good':
                                    st.markdown(
                                        f"<div class='good-observation'><strong>{model_name}:</strong> {observation}</div>",
                                        unsafe_allow_html=True
                                    )
                                elif quality == 'poor':
                                    st.markdown(
                                        f"<div class='poor-observation'><strong>{model_name}:</strong> {observation}</div>",
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f"<div class='neutral-observation'><strong>{model_name}:</strong> {observation}</div>",
                                        unsafe_allow_html=True
                                    )

                            # Detailed observation table for assignment reporting.
                            detailed_observations = []
                            for idx, row in comparison_df.iterrows():
                                rank = idx + 1
                                detailed_observations.append({
                                    'ML Model Name': row['Model'],
                                    'Observation about Model Performance': (
                                        f"On the Breast Cancer Wisconsin test dataset, {row['Model']} "
                                        f"ranked {rank} of {len(comparison_df)} with {row['Accuracy']:.2%} "
                                        f"accuracy. It correctly classified most benign and malignant "
                                        f"samples, identified {row['Recall']:.2%} of benign cases (class 1), and "
                                        f"achieved {row['Precision']:.2%} precision for benign predictions. "
                                        f"Its F1 Score of {row['F1 Score']:.4f} indicates the balance between "
                                        "correctly identifying benign cases and limiting incorrect benign labels."
                                    )
                                })

                            detailed_observations.append({
                                'ML Model Name': 'Overall Winner for this Dataset',
                                'Observation about Model Performance': (
                                    f"{best_model['Model']} is the overall winner on this Breast Cancer "
                                    f"Wisconsin test dataset because it achieved the highest accuracy "
                                    f"({best_model['Accuracy']:.2%}) among the selected models. Its precision "
                                    f"({best_model['Precision']:.2%}) and recall ({best_model['Recall']:.2%}) "
                                    "show that it provides a strong balance for the default benign class. "
                                    "The malignant-class row must also be reviewed before clinical use."
                                )
                            })

                            st.markdown(
                                "<div class='result-section-title'>📝 Detailed Dataset-Based Observation Table</div>",
                                unsafe_allow_html=True
                            )
                            with st.expander("About this detailed observation table", expanded=False):
                                st.info(
                                    "These observations interpret each model's results on the uploaded or bundled "
                                    "Breast Cancer Wisconsin test dataset. Class 0 represents malignant cases, "
                                    "so the malignant-class row must be reviewed because malignant samples "
                                    "predicted as benign are clinically important misses."
                                )
                            detailed_observations_df = pd.DataFrame(detailed_observations)
                            render_responsive_table(detailed_observations_df)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        st.info("Please upload a CSV file to get started!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    <p>Breast Cancer Wisconsin | Classification Model Evaluation</p>
    <p>Student ID: 2025AC05684 | BITS Pilani</p>
</div>
""", unsafe_allow_html=True)
