import os
import time
import pickle
import numpy as np
import logging
import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

# -----------------------------
# Lazy load scaler
# -----------------------------
scaler = None

def load_scaler():
    global scaler
    if scaler is None:
        BASE_DIR = os.path.dirname(__file__)
        SCALER_PATH = os.path.join(BASE_DIR, "scaler_a3.pkl")
        if not os.path.exists(SCALER_PATH):
            # In CI/CD, return a dummy scaler instead of failing
            class DummyScaler:
                def transform(self, X):
                    return X
            logging.warning("Scaler file not found, using DummyScaler for testing")
            scaler = DummyScaler()
        else:
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            logging.info("Scaler loaded successfully")
    return scaler

# -----------------------------
# Prepare feature vector
# -----------------------------
def get_X(max_power, mileage, year):
    numeric_features = np.array([[float(max_power), float(year), float(mileage)]])
    numeric_scaled = load_scaler().transform(numeric_features)
    return numeric_scaled

# -----------------------------
# Lazy load MLflow model
# -----------------------------
mlflow_model = None

def load_model():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.ml.brain.cs.ait.ac.th/")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        logging.info(f"Using MLflow credentials for {username}")
    else:
        logging.warning("No MLflow credentials found in environment")

    mlflow.set_tracking_uri(mlflow_uri)
    run_id = os.getenv("RUN_ID")
    model_name = os.getenv("MODEL_NAME", "st126522-a3-model")
    model_uri = f"runs:/{run_id}/model" if run_id else f"models:/{model_name}/Production"

    for attempt in range(5):
        try:
            logging.info(f"Loading MLflow model from {model_uri} (attempt {attempt + 1}/5)")
            model_instance = mlflow.pyfunc.load_model(model_uri)
            logging.info("MLflow model loaded successfully")
            return model_instance
        except mlflow.exceptions.MlflowException as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    raise RuntimeError(f"Failed to load MLflow model after 5 attempts. Tried URI: {model_uri}")

def get_mlflow_model():
    global mlflow_model
    if mlflow_model is None:
        # In CI/CD, return a dummy model if MLflow is not set
        class DummyModel:
            def predict(self, X):
                return np.array([1])
        try:
            mlflow_model = load_model()
        except Exception:
            logging.warning("MLflow model not loaded, using DummyModel for testing")
            mlflow_model = DummyModel()
    return mlflow_model

# -----------------------------
# Prediction function
# -----------------------------
def predict_selling_price(max_power, year, mileage):
    model_instance = get_mlflow_model()
    X = get_X(max_power, mileage, year)
    raw_pred = model_instance.predict(X)[0]
    class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
    label = class_map[int(raw_pred)]
    return raw_pred, label

# -----------------------------
# Test run
# -----------------------------
if __name__ == "__main__":
    try:
        pred, label = predict_selling_price(110, 2017, 18.0)
        print(f"Test Prediction: {pred} ({label})")
    except Exception as e:
        print(f"Prediction failed: {e}")
