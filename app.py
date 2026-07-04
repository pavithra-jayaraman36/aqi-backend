"""
app.py
=======
FastAPI application for the Air Quality Prediction System.

This service loads the pre-trained RandomForestRegressor model, the
fitted StandardScaler, and the feature name list (all produced by
train_model.py) exactly once at startup, then exposes REST endpoints
that the Lovable frontend can call to obtain AQI predictions.

Endpoints:
    GET  /         - Basic liveness/info message.
    GET  /health    - Health check endpoint (used by Render / uptime checks).
    POST /predict   - Predict AQI from pollutant + weather readings.

Run locally with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Author: AQI Backend Team
"""

import logging
import os
import pickle
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

# Maps the incoming request field names to the exact column names the
# model was trained on (see preprocess.py FEATURE_COLUMNS).
REQUEST_FIELD_TO_MODEL_FEATURE = {
    "co": "CO(GT)",
    "nmhc": "NMHC(GT)",
    "c6h6": "C6H6(GT)",
    "nox": "NOx(GT)",
    "no2": "NO2(GT)",
    "temperature": "T",
    "humidity": "RH",
    "absolute_humidity": "AH",
}

# Container for model artifacts loaded once at startup.
# Populated inside the lifespan handler below.
ml_artifacts: Dict[str, Any] = {}


# --------------------------------------------------------------------------
# Startup / shutdown lifecycle
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model artifacts exactly once when the application starts, and
    make them available via the module-level `ml_artifacts` dict for the
    lifetime of the process. This avoids reloading the (large) model
    file on every request.
    """
    logger.info("Application startup: loading model artifacts...")
    try:
        with open(MODEL_PATH, "rb") as f:
            ml_artifacts["model"] = pickle.load(f)
        logger.info("Loaded model from %s", MODEL_PATH)

        with open(SCALER_PATH, "rb") as f:
            ml_artifacts["scaler"] = pickle.load(f)
        logger.info("Loaded scaler from %s", SCALER_PATH)

        with open(FEATURE_NAMES_PATH, "rb") as f:
            ml_artifacts["feature_names"] = pickle.load(f)
        logger.info(
            "Loaded feature names from %s: %s",
            FEATURE_NAMES_PATH,
            ml_artifacts["feature_names"],
        )

        logger.info("All model artifacts loaded successfully. Service ready.")

    except FileNotFoundError as exc:
        # Fail loudly at startup rather than letting the app run in a
        # broken state where every /predict call would 500.
        logger.error("Could not find model artifact file: %s", exc)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while loading model artifacts: %s", exc)
        raise

    yield  # Application runs here.

    logger.info("Application shutdown: clearing model artifacts from memory.")
    ml_artifacts.clear()


# --------------------------------------------------------------------------
# FastAPI app instance
# --------------------------------------------------------------------------
app = FastAPI(
    title="AQI Prediction API",
    description="Backend service for predicting Air Quality Index (AQI) "
    "from pollutant and weather readings.",
    version="1.0.0",
    lifespan=lifespan,
)

# --------------------------------------------------------------------------
# CORS configuration
# --------------------------------------------------------------------------
# Allow all origins so the Lovable-hosted frontend (and any preview
# deployments) can call this API without CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Pydantic request / response schemas
# --------------------------------------------------------------------------
class AQIPredictionRequest(BaseModel):
    """
    Request schema for POST /predict.

    Field-level constraints:
        - All pollutant concentrations must be non-negative.
        - Relative humidity must be within [0, 100].
    """

    co: float = Field(..., ge=0, description="CO concentration (mg/m^3). Must be >= 0.")
    nmhc: float = Field(..., ge=0, description="NMHC concentration. Must be >= 0.")
    c6h6: float = Field(..., ge=0, description="Benzene (C6H6) concentration. Must be >= 0.")
    nox: float = Field(..., ge=0, description="NOx concentration (ppb). Must be >= 0.")
    no2: float = Field(..., ge=0, description="NO2 concentration. Must be >= 0.")
    temperature: float = Field(..., description="Ambient temperature in degrees Celsius.")
    humidity: float = Field(
        ..., ge=0, le=100, description="Relative humidity percentage. Must be between 0 and 100."
    )
    absolute_humidity: float = Field(..., ge=0, description="Absolute humidity. Must be >= 0.")

    class Config:
        json_schema_extra = {
            "example": {
                "co": 2.3,
                "nmhc": 150,
                "c6h6": 8.4,
                "nox": 120,
                "no2": 60,
                "temperature": 27,
                "humidity": 65,
                "absolute_humidity": 1.25,
            }
        }


class AQIPredictionResponse(BaseModel):
    """Response schema for POST /predict."""

    predicted_aqi: float
    category: str


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def get_aqi_category(aqi_value: float) -> str:
    """
    Map a numeric AQI value (0-500 scale) to a standard AQI category
    label (US EPA style breakpoints).

    Args:
        aqi_value: Predicted AQI value.

    Returns:
        Category label string.
    """
    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Moderate"
    if aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi_value <= 200:
        return "Unhealthy"
    if aqi_value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def build_feature_row(request: AQIPredictionRequest, feature_names: List[str]) -> pd.DataFrame:
    """
    Convert a validated request into a single-row DataFrame whose
    columns exactly match the order the model/scaler were trained on.

    Args:
        request: Validated AQIPredictionRequest instance.
        feature_names: Ordered list of feature column names loaded from
            model/feature_names.pkl.

    Returns:
        Single-row pandas DataFrame ready to be passed through the
        scaler and model.
    """
    request_dict = request.model_dump()

    # Build a lookup from model feature name -> incoming value using the
    # request-field-to-model-feature mapping.
    value_by_model_feature = {
        model_feature: request_dict[request_field]
        for request_field, model_feature in REQUEST_FIELD_TO_MODEL_FEATURE.items()
    }

    # Assemble the row strictly in the order defined by feature_names.pkl
    # so it always matches what the scaler/model expect, even if the
    # training feature order ever changes.
    ordered_values = [[value_by_model_feature[name] for name in feature_names]]
    return pd.DataFrame(ordered_values, columns=feature_names)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, str]:
    """Basic liveness/info endpoint."""
    return {"message": "AQI Prediction API Running"}


@app.get("/health")
def health() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns a simple healthy status if the process is up. Render and
    other platforms can poll this to verify the service is alive.
    """
    return {"status": "healthy"}


@app.post("/predict", response_model=AQIPredictionResponse)
def predict(request: AQIPredictionRequest) -> AQIPredictionResponse:
    """
    Predict the Air Quality Index (AQI) from pollutant and weather
    readings.

    Args:
        request: Validated input readings (see AQIPredictionRequest).

    Returns:
        AQIPredictionResponse containing the predicted AQI value and
        its category label.

    Raises:
        HTTPException(503): If model artifacts are not loaded.
        HTTPException(500): If prediction fails unexpectedly.
    """
    if "model" not in ml_artifacts or "scaler" not in ml_artifacts:
        logger.error("Prediction requested before model artifacts were loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Please try again shortly.",
        )

    try:
        model = ml_artifacts["model"]
        scaler = ml_artifacts["scaler"]
        feature_names = ml_artifacts["feature_names"]

        # Build the feature row in the correct column order.
        feature_row = build_feature_row(request, feature_names)
        logger.info("Received prediction request: %s", request.model_dump())

        # Scale features using the same scaler fitted during training.
        scaled_features = scaler.transform(feature_row)

        # Run inference.
        prediction = model.predict(scaled_features)
        predicted_aqi = float(np.clip(prediction[0], 0, 500))
        predicted_aqi = round(predicted_aqi, 2)

        category = get_aqi_category(predicted_aqi)

        logger.info(
            "Prediction successful: predicted_aqi=%.2f, category=%s",
            predicted_aqi,
            category,
        )

        return AQIPredictionResponse(predicted_aqi=predicted_aqi, category=category)

    except HTTPException:
        # Re-raise HTTPExceptions untouched (e.g. the 503 above).
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating the prediction.",
        )