import os
import sys

# Add the project root to the path so python can find the src module easily
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data import fetch_credit_data, perform_eda
from src.features import preprocess_data
from src.models import train_and_evaluate
from src.tune import tune_random_forest

def main():
    print("="*50)
    print("CREDIT SCORING MODEL END-TO-END PIPELINE")
    print("="*50)
    
    # Phase 1
    print("\n[Phase 1] Data Fetching and EDA")
    df = fetch_credit_data()
    perform_eda(df)
    
    # Phase 2
    print("\n[Phase 2] Feature Engineering")
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Phase 3
    print("\n[Phase 3] Initial Model Training")
    train_and_evaluate(
        "data/processed/X_train.csv",
        "data/processed/X_test.csv",
        "data/processed/y_train.csv",
        "data/processed/y_test.csv"
    )
    
    # Phase 4 & 5
    print("\n[Phase 4 & 5] Hyperparameter Tuning and Model Saving")
    tune_random_forest(
        "data/processed/X_train.csv",
        "data/processed/X_test.csv",
        "data/processed/y_train.csv",
        "data/processed/y_test.csv"
    )
    
    print("\n" + "="*50)
    print("Pipeline execution completed successfully!")
    print("The final tuned model is saved at 'models/best_model.joblib'")
    print("The preprocessing pipeline is saved at 'models/preprocessor.joblib'")
    print("="*50)

if __name__ == "__main__":
    main()
