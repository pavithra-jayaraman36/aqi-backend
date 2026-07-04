"""
train_model.py
================
Model training script for the Air Quality Prediction System.

This script:
    1. Runs the preprocessing pipeline (preprocess.py) to obtain a clean
       DataFrame with the synthetic 'AQI' target.
    2. Splits the data into train/test sets (80/20).
    3. Scales features using StandardScaler.
    4. Trains a RandomForestRegressor.
    5. Evaluates the model using MAE, MSE, RMSE, and R^2.
    6. Persists the trained model, scaler, and feature name list to disk
       so the FastAPI inference layer can load them at request time.

Usage:
    python train_model.py

Author: AQI Backend Team
"""

import logging
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocess import FEATURE_COLUMNS, get_feature_matrix_and_target, preprocess_pipeline

# --------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_model")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

TEST_SIZE = 0.2
RANDOM_STATE = 42

RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 15


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
def prepare_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Run the preprocessing pipeline and return the model-ready feature
    matrix (X) and target vector (y).

    Returns:
        Tuple of (X, y).

    Raises:
        FileNotFoundError: If the raw dataset cannot be located.
        KeyError: If expected feature columns are missing after cleaning.
    """
    logger.info("Running preprocessing pipeline to prepare training data...")
    processed_df = preprocess_pipeline()
    X, y = get_feature_matrix_and_target(processed_df)
    logger.info(
        "Prepared data: X shape=%s, y shape=%s, features=%s",
        X.shape,
        y.shape,
        FEATURE_COLUMNS,
    )
    return X, y


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_random_forest(
    X_train_scaled: np.ndarray, y_train: pd.Series
) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor on the scaled training data.

    Args:
        X_train_scaled: Scaled training feature matrix.
        y_train: Training target vector.

    Returns:
        Fitted RandomForestRegressor instance.
    """
    logger.info(
        "Training RandomForestRegressor (n_estimators=%d, max_depth=%d, "
        "random_state=%d)...",
        RF_N_ESTIMATORS,
        RF_MAX_DEPTH,
        RANDOM_STATE,
    )
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)
    logger.info("Model training complete.")
    return model


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_model(
    model: RandomForestRegressor, X_test_scaled: np.ndarray, y_test: pd.Series
) -> dict:
    """
    Evaluate the trained model on the held-out test set.

    Computes MAE, MSE, RMSE, and R^2 score, logs and prints them.

    Args:
        model: Fitted RandomForestRegressor.
        X_test_scaled: Scaled test feature matrix.
        y_test: True test target values.

    Returns:
        Dictionary containing the computed metrics.
    """
    logger.info("Evaluating model on test set...")
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    logger.info("Evaluation metrics: %s", metrics)

    # Print a clean, human-readable summary as requested.
    print("\n===== Model Evaluation Metrics =====")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print("=====================================\n")

    return metrics


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_artifacts(
    model: RandomForestRegressor, scaler: StandardScaler, feature_names: list
) -> None:
    """
    Persist the trained model, fitted scaler, and feature name list to
    the model/ directory using pickle.

    Args:
        model: Fitted RandomForestRegressor.
        scaler: Fitted StandardScaler.
        feature_names: Ordered list of feature column names used in
            training (needed at inference time to build the input row
            in the correct order).

    Raises:
        OSError: If the model directory cannot be created or files
            cannot be written.
    """
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info("Ensured model directory exists at '%s'", MODEL_DIR)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved trained model to %s", MODEL_PATH)

        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Saved fitted scaler to %s", SCALER_PATH)

        with open(FEATURE_NAMES_PATH, "wb") as f:
            pickle.dump(feature_names, f)
        logger.info("Saved feature names to %s", FEATURE_NAMES_PATH)

    except OSError as exc:
        logger.exception("Failed to save model artifacts: %s", exc)
        raise


# --------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------
def main() -> None:
    """
    Run the full training pipeline: data prep, split, scale, train,
    evaluate, and persist artifacts.
    """
    try:
        # 1. Prepare data.
        X, y = prepare_data()

        # 2. Train/test split (80/20).
        logger.info(
            "Splitting data into train/test sets (test_size=%.2f, "
            "random_state=%d)...",
            TEST_SIZE,
            RANDOM_STATE,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        logger.info(
            "Split complete: X_train=%s, X_test=%s", X_train.shape, X_test.shape
        )

        # 3. Feature scaling. Fit only on training data to avoid leakage.
        logger.info("Fitting StandardScaler on training features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 4. Train model.
        model = train_random_forest(X_train_scaled, y_train)

        # 5. Evaluate model.
        evaluate_model(model, X_test_scaled, y_test)

        # 6. Persist artifacts for the FastAPI inference layer.
        save_artifacts(model, scaler, FEATURE_COLUMNS)

        logger.info("Training pipeline finished successfully.")

    except FileNotFoundError as exc:
        logger.error("Required data file was not found: %s", exc)
        raise
    except KeyError as exc:
        logger.error("Data schema issue encountered: %s", exc)
        raise
    except Exception as exc:  # noqa: BLE001 - top-level safety net for a CLI script
        logger.exception("Unexpected error during training pipeline: %s", exc)
        raise


if __name__ == "__main__":
    main()