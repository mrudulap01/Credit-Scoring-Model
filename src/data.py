import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

def fetch_credit_data(save_path="data/raw/credit_data.csv"):
    """
    Fetches the German Credit dataset from OpenML and saves it locally.
    Returns the loaded pandas DataFrame.
    """
    print("Fetching German Credit Data from OpenML...")
    # Fetching the German Credit dataset (ID: 31)
    data = fetch_openml('credit-g', version=1, as_frame=True, parser='auto')
    df = data.frame
    
    # Save raw data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Data successfully saved to {save_path}")
    return df

def perform_eda(df, output_dir="reports/figures"):
    """
    Performs basic Exploratory Data Analysis and saves plots to the output directory.
    """
    print(f"\nPerforming Exploratory Data Analysis...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Target Variable Distribution (Good vs Bad Credit)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='class', palette='Set2')
    plt.title('Target Class Distribution (Creditability)')
    plt.xlabel('Credit Risk')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()
    print("- Created class_distribution.png")
    
    # 2. Correlation Heatmap for Numerical Features
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title('Correlation Matrix of Numerical Features')
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        plt.close()
        print("- Created correlation_matrix.png")
    
    # 3. Distribution of Credit Amount by Class
    if 'credit_amount' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x='credit_amount', hue='class', kde=True, bins=30, palette='Set2')
        plt.title('Distribution of Credit Amount by Class')
        plt.xlabel('Credit Amount')
        plt.savefig(os.path.join(output_dir, "credit_amount_distribution.png"))
        plt.close()
        print("- Created credit_amount_distribution.png")
        
    print(f"All EDA plots saved to {output_dir}")

if __name__ == "__main__":
    # Ensure working directory is the project root when running as script
    # This allows it to find the data/ and reports/ folders properly
    current_dir = os.getcwd()
    if not current_dir.endswith("Credit Score"):
        print("Please run this script from the project root directory ('Credit Score').")
    
    # 1. Fetch data
    df = fetch_credit_data()
    
    # 2. Print basic stats
    print("\n--- Dataset Info ---")
    print(f"Shape: {df.shape}")
    print("\n--- Missing Values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0]) # Print columns with missing values
    
    # 3. EDA
    perform_eda(df)
