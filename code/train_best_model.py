from dotenv import load_dotenv, set_key
import os
import time
import numpy as np
import mlflow
import mlflow.sklearn
from MultiLogisticRegression import MultinomialLogisticRegression, classification_report_custom

def train_and_register_best_model(X_train, Y_train, X_test, Y_test):
    # Load environment variables
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    if os.getenv("MLFLOW_TRACKING_TOKEN"):
        os.environ["MLFLOW_TRACKING_TOKEN"] = os.getenv("MLFLOW_TRACKING_TOKEN")
    else:
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    mlflow.set_experiment("st126522-a3")

    # learning_rates = [0.01, 0.05, 0.1]
    learning_rates = [0.01, 0.05]
    # ridge_values = [0.0, 0.5, 1.0]
    ridge_values = [0.0, 0.5]
    # methods = ["minibatch", "sto", "batch"]
    methods = ["minibatch", "batch"]
    max_iter = 4000

    best_macro_f1 = -np.inf
    best_model = None
    best_run_id = None

    # Train all combinations
    for lr in learning_rates:
        for ridge in ridge_values:
            for method in methods:
                run_name = f"{int(time.time())}_method_{method}_ridge_{ridge}_lr_{lr}"
                print(f"\n=== Run: {run_name} ===")

                with mlflow.start_run(run_name=run_name) as run:
                    run_id = run.info.run_id

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
                    _, macro, weighted = classification_report_custom(Y_test, y_pred)

                    # Log parameters and metrics
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

                    # Log model as artifact
                    mlflow.sklearn.log_model(model, artifact_path="model")

                    # Track best model
                    if macro['f1'] > best_macro_f1:
                        best_macro_f1 = macro['f1']
                        best_model = model
                        best_run_id = run_id

    print("\nBest Macro F1:", best_macro_f1, "from run_id:", best_run_id)

    # Ensure last run is closed
    mlflow.end_run()

    # Register the best model (requires registry-enabled MLflow)
    registered_model_name = "st126522-a3-model"
    model_uri = f"runs:/{best_run_id}/model"
    try:
        registered_model = mlflow.register_model(model_uri, registered_model_name)
        print(f"Model registered as '{registered_model_name}'")
    except Exception as e:
        print("Model registration failed. Are you using a registry-enabled MLflow? Error:", e)

    # Update .env with best run_id
    env_path = ".env"
    if not os.path.exists(env_path):
        open(env_path, "a").close()
    set_key(env_path, "RUN_ID", best_run_id)

    return best_model, best_run_id
