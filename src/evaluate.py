import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import os

def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model", output_dir="reports/figures"):
    """
    Evaluates a model using multiple metrics and plots ROC curve and Confusion Matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['ROC AUC'] = roc_auc_score(y_true, y_prob)
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["ROC AUC"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--') # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, f"roc_curve_{model_name.replace(' ', '_')}.png"))
        plt.close()
        
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png"))
    plt.close()

    return metrics
