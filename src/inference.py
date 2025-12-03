from pathlib import Path
import xgboost as xgb
from src.data_prep import preprocess_raw_data
from src.feature_engineering import apply_advanced_features
import pandas as pd
import shap
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final_xgb_model.json"


def load_model(model_path: Path = MODEL_PATH) -> xgb.XGBClassifier:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model

def dict_to_dataframe(input_dict: dict) -> pd.DataFrame:

    return pd.DataFrame([input_dict])

def preprocess_input(df_raw: pd.DataFrame) -> pd.DataFrame:

    df_clean = preprocess_raw_data(df_raw)
    df_final = apply_advanced_features(df_clean)
    return df_final

def predict_single(input_dict: dict, model=None, threshold: float = 0.5):
    # Load model if not passed
    if model is None:
        model = load_model()

    # Convert input → DataFrame
    df_raw = pd.DataFrame([input_dict])

    # Preprocess + feature engineering
    df_final = preprocess_input(df_raw)

    # Predict probability of default (class 1)
    proba = model.predict_proba(df_final)[:, 1][0]

    # Convert probability to 0/1 label
    prediction = int(proba >= threshold)

    return {
        "probability": float(proba),
        "prediction": prediction
    }

def explain_single(df_final, model):
    """Return shap values and expected value for a single preprocessed row."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_final)
    return explainer, shap_values

if __name__ == "__main__":
    model = load_model()

    sample_input = {
        "LIMIT_BAL": 50000,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 30,
        "PAY_0": 1,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 20000,
        "BILL_AMT2": 21000,
        "BILL_AMT3": 22000,
        "BILL_AMT4": 23000,
        "BILL_AMT5": 24000,
        "BILL_AMT6": 25000,
        "PAY_AMT1": 5000,
        "PAY_AMT2": 4000,
        "PAY_AMT3": 3000,
        "PAY_AMT4": 3000,
        "PAY_AMT5": 2000,
        "PAY_AMT6": 1000
    }

    result = predict_single(sample_input, model)
    print(result)

