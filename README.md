# 🏦 Premium Credit Scoring AI

A full-stack, production-ready Machine Learning application designed to predict creditworthiness using historical financial behavior. Built with a robust Scikit-Learn pipeline and a beautiful, animated, glassmorphism-styled Streamlit dashboard.

---

## 🌟 Key Features

### 🧠 Advanced Machine Learning Pipeline
- **End-to-End Orchestration:** Modular architecture separating data ingestion, feature engineering, model training, and evaluation.
- **Zero Data Leakage:** Uses strict Scikit-Learn `Pipeline` and `ColumnTransformer` objects to ensure scaling and encoding are calculated strictly on the training folds.
- **Algorithmic Benchmarking:** Automatically compares Logistic Regression, Decision Tree, and Random Forest models.
- **Hyperparameter Tuning:** Built-in 5-fold `GridSearchCV` optimization explicitly targeting the **ROC-AUC** metric to properly handle class imbalances.
- **Automated Overfitting Detection:** Pipeline automatically calculates Train vs. Test score gaps to identify overfitting models.

### 🎨 Hyper-Modern Web Application
- **Dynamic Risk Categorization:** Calculates approval probability and classifies applicants into Low, Medium, or High Risk tiers.
- **Actionable Insights:** Dynamically provides the end-user with "Pro-Tips" and specific suggestions to improve their score based on their exact risk bracket.
- **Premium Aesthetics:** Features an animated deep-blue gradient background, frosted glassmorphism form cards, neon hover-glows, and user-friendly human-readable dropdowns.

---

## 📁 Project Structure

```text
Credit Score/
├── app/                      # Streamlit Web Application
│   ├── main.py               # Dashboard entry point and CSS engine
│   ├── prediction.py         # Prediction logic & @st.cache_resource model loader
│   └── ui_components.py      # Human-readable UI mapping & layout
├── models/                   # Serialized ML Artifacts
│   ├── best_model.joblib     # Tuned Random Forest Classifier
│   └── preprocessor.joblib   # Fitted Scalers and OneHotEncoders
├── src/                      # ML Pipeline Source Code
│   ├── data.py               # Fetches German Credit Dataset from OpenML
│   ├── evaluate.py           # Metrics calculation (Accuracy, Precision, Recall, F1, ROC-AUC)
│   ├── features.py           # Train-test splitting, data imputation, and encoding
│   ├── models.py             # Baseline model definitions
│   └── tune.py               # GridSearchCV implementation
├── .streamlit/               # Streamlit Configurations
│   └── config.toml           # Base UI theme configurations
├── main.py                   # Orchestration script to run the entire ML pipeline
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/credit-scoring-ai.git
   cd credit-scoring-ai
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Train the Machine Learning Model
To fetch the dataset, train all models, perform hyperparameter tuning, and save the absolute best model, run the orchestration script:
```bash
python main.py
```
*Note: This will overwrite `models/best_model.joblib` with the newly optimized weights.*

### 2. Launch the Streamlit Dashboard
To start the interactive web application, run:
```bash
python -m streamlit run app/main.py
```
Navigate to `http://localhost:8501` in your web browser to interact with the premium UI.

---

## 📊 Model Details
- **Dataset:** German Credit Dataset (fetched dynamically via OpenML).
- **Target Variable:** `class` (Good vs. Bad credit risk).
- **Best Performing Algorithm:** Random Forest Classifier.
- **Optimization Strategy:** Because the dataset inherently contains a 70/30 class imbalance, traditional Accuracy is a flawed metric. The pipeline actively targets **Area Under the Receiver Operating Characteristic Curve (ROC-AUC)**, ensuring the model effectively distinguishes between safe and risky applicants.

---

## 🛠️ Technologies Used
- **Python 3**
- **Scikit-Learn** (Pipelines, GridSearchCV, Models, Metrics)
- **Pandas & NumPy** (Data manipulation)
- **Streamlit** (Frontend framework)
- **Joblib** (Model serialization)
- **CSS3** (Custom glassmorphism & animations)
