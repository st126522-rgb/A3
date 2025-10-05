import os
import time
import pickle
import numpy as np
import logging
import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv
import os

BASE_DIR = os.path.dirname(__file__)  # directory where model.py resides
SCALER_PATH = os.path.join(BASE_DIR, "scaler_a3.pkl")

logging.basicConfig(level=logging.INFO)
load_dotenv()


# Load scaler

with open("scaler_a3.pkl", "rb") as f:
    scaler = pickle.load(f)


# Prepare feature vector

def get_X(max_power, mileage, year):
    numeric_features = np.array([[float(max_power), float(year), float(mileage)]])
    numeric_scaled = scaler.transform(numeric_features)
    return numeric_scaled


# Load MLflow model

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
            model = mlflow.pyfunc.load_model(model_uri)
            logging.info("✅ MLflow model loaded successfully")
            return model
        except mlflow.exceptions.MlflowException as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3)

    raise RuntimeError(f"Failed to load MLflow model after 5 attempts. Tried URI: {model_uri}")


# Load model at import

try:
    mlflow_model = load_model()
except Exception as e:
    logging.error(f"MLflow model could not be loaded: {e}")
    mlflow_model = None


# Predict function

def predict_selling_price(max_power, year, mileage):
    if mlflow_model is None:
        raise RuntimeError("MLflow model is not loaded. Cannot make predictions.")

    X = get_X(max_power, mileage, year)
    raw_pred = mlflow_model.predict(X)[0]
    class_map = ["Cheap", "Average", "Expensive", "Very Expensive"]
    label = class_map[int(raw_pred)]
    return raw_pred, label
 
if __name__ == "__main__":
    # Test prediction
    test_max_power = 110
    test_year = 2017
    test_mileage = 18.0

    try:
        pred, label = predict_selling_price(test_max_power, test_year, test_mileage)
        print(f"Test Prediction: {pred} ({label})")
    except Exception as e:
        print(f"Prediction failed: {e}")