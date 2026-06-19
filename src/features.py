import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def preprocess_data(input_path="data/raw/credit_data.csv", output_dir="data/processed", test_size=0.2, random_state=42):
    """
    Loads raw data, performs feature engineering (imputation, encoding, scaling),
    splits into train and test sets, and saves the processed data.
    """
    print("Loading raw data...")
    df = pd.read_csv(input_path)
    
    # Target variable is 'class' (good vs bad credit)
    # Convert 'class' to binary: 'good' -> 1, 'bad' -> 0
    if df['class'].dtype == 'object' or df['class'].dtype.name == 'category':
        df['class'] = df['class'].map({'good': 1, 'bad': 0})
        
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    print(f"Numerical features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Create preprocessing pipelines
    # 1. Numerical pipeline: Impute missing values with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Categorical pipeline: Impute missing values with mode, then One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    print("Fitting preprocessor and transforming data...")
    # Fit on training data and transform both train and test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_feature_names)
    
    print("Applying SMOTE to balance training data...")
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    # Convert back to DataFrame for easier inspection and saving
    X_train_df = pd.DataFrame(X_train_resampled, columns=feature_names)
    y_train = pd.Series(y_train_resampled, name='class')
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    
    print("Saving processed datasets...")
    X_train_df.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    # Save the preprocessor to use later on new data
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    print("Saved preprocessor to models/preprocessor.joblib")
    
    print("Feature engineering complete!")
    return X_train_df, X_test_df, y_train, y_test

if __name__ == "__main__":
    current_dir = os.getcwd()
    if not current_dir.endswith("Credit Score"):
        print("Please run this script from the project root directory ('Credit Score').")
    
    preprocess_data()
