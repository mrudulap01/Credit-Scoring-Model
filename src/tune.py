import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os
from evaluate import evaluate_model

def tune_random_forest(X_train_path, X_test_path, y_train_path, y_test_path):
    print("Loading processed data for tuning...")
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()
    
    # We select Random Forest as it was the best performing model in Phase 3
    rf = RandomForestClassifier(random_state=42)
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    print("\nStarting Grid Search for Random Forest...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=5, n_jobs=-1, scoring='roc_auc', verbose=1)
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV ROC AUC: {grid_search.best_score_:.4f}")
    
    # Evaluate best model on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    print("\n--- Tuned Random Forest Evaluation ---")
    metrics = evaluate_model(y_test, y_pred, y_prob, model_name="Tuned Random Forest")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Save the best model
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nSaved best model to {model_path}")
    
    return best_model

if __name__ == "__main__":
    current_dir = os.getcwd()
    if not current_dir.endswith("Credit Score"):
        print("Please run this script from the project root directory ('Credit Score').")
        
    tune_random_forest(
        "data/processed/X_train.csv",
        "data/processed/X_test.csv",
        "data/processed/y_train.csv",
        "data/processed/y_test.csv"
    )
