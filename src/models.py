import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
from .evaluate import evaluate_model

def train_and_evaluate(X_train_path, X_test_path, y_train_path, y_test_path):
    print("Loading processed data...")
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze() # Squeeze to make it a Series
    y_test = pd.read_csv(y_test_path).squeeze()
    
    # Define models dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced')
    }
    
    results = {}
    
    print("\n--- Model Training & Evaluation ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        # Check if model has predict_proba for ROC AUC calculation
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Evaluate using our custom module
        metrics = evaluate_model(y_test, y_pred, y_prob, model_name=name)
        results[name] = metrics
        
    # Create comparison DataFrame
    results_df = pd.DataFrame(results).T
    print("\n--- Initial Model Comparison ---")
    print(results_df)
    
    # Save results to csv
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv("reports/model_comparison.csv")
    print("\nInitial results saved to reports/model_comparison.csv")
    
    return results_df

if __name__ == "__main__":
    current_dir = os.getcwd()
    if not current_dir.endswith("Credit Score"):
        print("Please run this script from the project root directory ('Credit Score').")
    
    train_and_evaluate(
        "data/processed/X_train.csv",
        "data/processed/X_test.csv",
        "data/processed/y_train.csv",
        "data/processed/y_test.csv"
    )
