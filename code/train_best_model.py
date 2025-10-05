import os
import time 
import numpy as np
import pandas as pd
from mlflow import pyfunc
import mlflow.sklearn
from code.MultiLogisticRegression import MultinomialLogisticRegression, classification_report_custom
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature


def train_and_register_best_model(X_train, Y_train, X_test, Y_test):
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    if os.getenv("MLFLOW_TRACKING_TOKEN"):
        os.environ["MLFLOW_TRACKING_TOKEN"] = os.getenv("MLFLOW_TRACKING_TOKEN")
    else:
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    mlflow.set_experiment("st126522-a3")

    learning_rates = [0.01]
    ridge_values = [0.0, 1.0]
    methods = ["minibatch", "sto"]
    max_iter = 4000

    best_macro_f1 = -np.inf
    best_model = None
    best_run_name = None
    all_results = []

    for lr in learning_rates:
        for ridge in ridge_values:
            for method in methods:
                run_name = f"{int(time.time())}_method_{method}_ridge_{ridge}_lr_{lr}"
                print(f"\n=== Run: {run_name} ===")

                with mlflow.start_run(run_name=run_name):
                    model = MultinomialLogisticRegression(
                        n_classes=Y_train.shape[1],
                        n_features=X_train.shape[1],
                        lr=lr,
                        max_iter=max_iter,
                        method=method,
                        batch_frac=0.25,
                        l2_lambda=ridge,
                        verbose=False,
                    )
                    model.fit(X_train, Y_train)

                    y_pred = model.predict(X_test)
                    acc = np.mean(Y_test == y_pred)
                    per_class, macro, weighted = classification_report_custom(Y_test, y_pred)

                    mlflow.log_params({
                        "method": method,
                        "learning_rate": lr,
                        "l2_lambda": ridge,
                        "max_iter": max_iter
                    })
                    mlflow.log_metrics({
                        "accuracy": acc,
                        "macro_f1": macro['f1'],
                        "weighted_f1": weighted['f1']
                    })

                    all_results.append({
                        "run_name": run_name,
                        "method": method,
                        "lr": lr,
                        "ridge": ridge,
                        "accuracy": acc,
                        "macro_f1": macro['f1'],
                        "weighted_f1": weighted['f1'],
                    })

                if macro['f1'] > best_macro_f1:
                    best_macro_f1 = macro['f1']
                    best_model = model
                    best_run_name = run_name

    print("\n Best model:", best_run_name, "Macro F1:", best_macro_f1)

    best_result = next(r for r in all_results if r['run_name'] == best_run_name)
    registered_model_name = "st126522-a3-model"

    with mlflow.start_run(run_name=f"best_model_{best_run_name}"):
        mlflow.log_params({
            "description": "Best model from grid search based on macro F1",
            "method": best_model.method,
            "lr": best_model.lr,
            "l2_lambda": best_model.l2
        })
        mlflow.log_metrics({
            "accuracy": best_result['accuracy'],
            "macro_f1": best_result['macro_f1'],
            "weighted_f1": best_result['weighted_f1']
        })

        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_train[:5],
            signature=signature,
            registered_model_name=registered_model_name,
        )

        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(registered_model_name, stages=["None"])[-1].version
        client.transition_model_version_stage(
            name=registered_model_name,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True,
        )

    print(f"\n Best model promoted to Production: {best_run_name}")
    return registered_model_name