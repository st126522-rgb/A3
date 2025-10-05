# dash_quantile_app.py
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import mlflow
import mlflow.pyfunc
import traceback
import os
from dotenv import load_dotenv
import pickle

load_dotenv()  # Reads .env file (must contain MLFLOW_TRACKING_URI and either TOKEN or USER/PASS)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Authentication — token-based preferred
if os.getenv("MLFLOW_TRACKING_TOKEN"):
    os.environ["MLFLOW_TRACKING_TOKEN"] = os.getenv("MLFLOW_TRACKING_TOKEN")
else:
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


MODEL_NAME = "st126522-a3-model"
model_version = 2

model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{model_version}")
# MODEL_URI = f"models/{MODEL_NAME}/Production"

# try:
#     model = mlflow.pyfunc.load_model(MODEL_URI)
# except Exception as e:
#     raise RuntimeError(f" Failed to load model from MLflow:\n{e}")

# -------------------------------
# Dash App
# -------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Car Price Quantile Prediction", className="text-center my-3"),

    dbc.Alert([
        html.H5("Instructions:", className="mb-2"),
        html.Ul([
            html.Li("Enter the car's manufacturing year, mileage, and max power."),
            html.Li("Model predicts which price range your car belongs to."),
            html.Li("Output shows whether it’s in bottom/mid/top price range.")
        ])
    ], color="info"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Year"),
            dcc.Input(id="year", type="number", placeholder="e.g. 2017", className="form-control")
        ], width=4),
        dbc.Col([
            dbc.Label("Mileage (kmpl)"),
            dcc.Input(id="mileage", type="number", placeholder="e.g. 18.0", className="form-control")
        ], width=4),
        dbc.Col([
            dbc.Label("Max Power (bhp)"),
            dcc.Input(id="max_power", type="number", placeholder="e.g. 110", className="form-control")
        ], width=4),
    ], className="mb-4"),

    dbc.Button("Predict Range", id="predict-btn", color="primary", className="mb-3"),
    html.Div(id="output", className="h4 text-success mt-3 text-center")
], fluid=True)

# -------------------------------
# Quantile Interpretation
# -------------------------------
def interpret_quantile(pred):
    mapping = {
        0: "Bottom 25% (Lowest price range)",
        1: "25–50% range (Lower-mid range)",
        2: "50–75% range (Upper-mid range)",
        3: "Top 25% (Highest price range)"
    }
    return mapping.get(int(pred), "Unknown range")

# -------------------------------
# Callback
# -------------------------------
@app.callback(
    Output("output", "children"),
    Input("predict-btn", "n_clicks"),
    State("year", "value"),
    State("mileage", "value"),
    State("max_power", "value"),
)
def predict_quantile(n_clicks, year, mileage, max_power):
    if not n_clicks:
        return ""
    try:
        if None in (year, mileage, max_power):
            return "Please fill in all fields."

        features = np.array([[year, mileage, max_power,1]])
        pred_class = model.predict(features)[0]

        range_text = interpret_quantile(pred_class)
        return f"Your car's predicted price range: {range_text}"

    except Exception as e:
        return f"Error: {str(e)}\n\n{traceback.format_exc()}"

# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)

