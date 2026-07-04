"""
preprocess.py
==============
Data preprocessing and feature-engineering module for the Air Quality
Prediction System.

Responsibilities:
    1. Load the raw UCI Air Quality dataset (semicolon-separated, European
       decimal format).
    2. Clean the data:
         - Drop completely empty columns.
         - Replace the dataset's sentinel value (-200) with NaN.
         - Impute missing values using the column median.
         - Drop duplicate rows.
         - Ensure all feature columns are numeric.
    3. Engineer a synthetic AQI target (0-500 scale) from pollutant
       concentrations, since the raw dataset has no AQI column.

This module is meant to be imported by train.py, but can also be run
standalone to produce a cleaned CSV for inspection:

    python preprocess.py

Author: AQI Backend Team
"""

import logging
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("preprocess")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
# Path to the raw dataset (relative to project root).
RAW_DATA_PATH = os.path.join("dataset", "AirQualityUCI.csv")

# Path where the cleaned/engineered dataset will be written for inspection.
PROCESSED_DATA_PATH = os.path.join("dataset", "processed_data.csv")

# The sentinel value the UCI dataset uses to represent missing readings.
MISSING_VALUE_SENTINEL = -200

# Columns that should never be used as model features.
NON_FEATURE_COLUMNS = ["Date", "Time"]

# Pollutant columns used both as model input features and as the basis
# for the synthetic AQI target.
POLLUTANT_COLUMNS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

# Full list of input features used for model training (pollutants + weather).
FEATURE_COLUMNS = POLLUTANT_COLUMNS + ["T", "RH", "AH"]

# Approximate real-world "healthy ceiling" concentrations used to rescale
# each pollutant onto a common 0-500 index before combining them into the
# synthetic AQI. These are rough, illustrative reference points (not an
# official regulatory standard) chosen so that typical dataset values map
# into a realistic-looking 0-500 AQI range.
POLLUTANT_REFERENCE_MAX = {
    "CO(GT)": 10.0,      # mg/m^3
    "NMHC(GT)": 1000.0,  # microg/m^3
    "C6H6(GT)": 50.0,    # microg/m^3
    "NOx(GT)": 600.0,    # ppb
    "NO2(GT)": 200.0,    # microg/m^3
}

# Relative importance of each pollutant when combining them into a single
# synthetic AQI score. Weights sum to 1.0.
POLLUTANT_WEIGHTS = {
    "CO(GT)": 0.25,
    "NMHC(GT)": 0.10,
    "C6H6(GT)": 0.15,
    "NOx(GT)": 0.20,
    "NO2(GT)": 0.30,
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Air Quality UCI dataset.

    The UCI CSV export uses ';' as the field separator and ',' as the
    decimal separator (European format), and typically has two trailing
    empty columns due to a trailing ';;' on every line.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Raw pandas DataFrame exactly as read from disk.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not os.path.exists(filepath):
        logger.error("Dataset file not found at: %s", filepath)
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    logger.info("Loading raw dataset from %s", filepath)
    df = pd.read_csv(filepath, sep=";", decimal=",", engine="python")
    logger.info("Raw dataset loaded with shape %s", df.shape)
    return df


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are completely empty (all NaN)."""
    before_cols = set(df.columns)
    df = df.dropna(axis=1, how="all")
    dropped = before_cols - set(df.columns)
    if dropped:
        logger.info("Dropped completely empty columns: %s", sorted(dropped))
    return df


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that are completely empty except for Date/Time.

    Some exports of this dataset contain trailing blank rows at the end
    of the file. These rows carry no signal and must be removed before
    any numeric processing.
    """
    feature_like_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    before_rows = len(df)
    df = df.dropna(axis=0, how="all", subset=feature_like_cols)
    removed = before_rows - len(df)
    if removed:
        logger.info("Dropped %d completely empty rows", removed)
    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every column except Date/Time is numeric.

    Any value that cannot be parsed as a number becomes NaN, which is
    then handled later by median imputation.
    """
    numeric_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("Converted %d columns to numeric dtype", len(numeric_cols))
    return df


def replace_sentinel_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the dataset's -200 missing-value marker with NaN."""
    numeric_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    sentinel_count = int((df[numeric_cols] == MISSING_VALUE_SENTINEL).sum().sum())
    df[numeric_cols] = df[numeric_cols].replace(MISSING_VALUE_SENTINEL, np.nan)
    logger.info(
        "Replaced %d occurrences of sentinel value (%s) with NaN",
        sentinel_count,
        MISSING_VALUE_SENTINEL,
    )
    return df


def impute_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values using each column's median."""
    numeric_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    missing_before = int(df[numeric_cols].isna().sum().sum())
    for col in numeric_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    logger.info(
        "Imputed %d missing values using column medians", missing_before
    )
    return df


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before_rows = len(df)
    df = df.drop_duplicates()
    removed = before_rows - len(df)
    if removed:
        logger.info("Dropped %d duplicate rows", removed)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline on the raw dataset.

    Order of operations matters:
        1. Drop empty columns.
        2. Drop empty rows.
        3. Convert everything (except Date/Time) to numeric.
        4. Replace -200 sentinel values with NaN.
        5. Median-impute missing values.
        6. Drop duplicate rows.

    Args:
        df: Raw DataFrame as loaded from CSV.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Starting data cleaning pipeline...")
    df = drop_empty_columns(df)
    df = drop_empty_rows(df)
    df = convert_numeric_columns(df)
    df = replace_sentinel_with_nan(df)
    df = impute_missing_with_median(df)
    df = drop_duplicate_rows(df)
    df = df.reset_index(drop=True)
    logger.info("Data cleaning complete. Final shape: %s", df.shape)
    return df


# --------------------------------------------------------------------------
# Feature engineering: synthetic AQI target
# --------------------------------------------------------------------------
def compute_pollutant_sub_index(series: pd.Series, reference_max: float) -> pd.Series:
    """
    Convert a raw pollutant concentration series into a 0-500 sub-index.

    Values are linearly scaled against `reference_max` (the concentration
    considered to correspond to an index of 500) and clipped to [0, 500]
    so extreme outliers don't blow up the scale.

    Args:
        series: Raw pollutant concentration values (must be >= 0).
        reference_max: Concentration value mapped to a sub-index of 500.

    Returns:
        Series of sub-index values in the range [0, 500].
    """
    # Clip negative values (shouldn't occur post-cleaning, but defensive).
    safe_series = series.clip(lower=0)
    sub_index = (safe_series / reference_max) * 500.0
    return sub_index.clip(lower=0, upper=500)


def engineer_aqi_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a synthetic 'AQI' target column on a 0-500 scale.

    Since the raw UCI dataset does not include an official AQI value,
    we construct one as a weighted combination of per-pollutant
    sub-indices (each individually scaled to 0-500 using illustrative
    reference concentrations), which mirrors the general approach used
    by real-world AQI calculations (e.g. CPCB/EPA style sub-index
    aggregation) while remaining fully derivable from the available
    pollutant columns.

    Args:
        df: Cleaned DataFrame containing the pollutant columns.

    Returns:
        DataFrame with an added 'AQI' column.
    """
    logger.info("Engineering synthetic AQI target...")

    sub_indices = pd.DataFrame(index=df.index)
    for pollutant in POLLUTANT_COLUMNS:
        ref_max = POLLUTANT_REFERENCE_MAX[pollutant]
        sub_indices[pollutant] = compute_pollutant_sub_index(df[pollutant], ref_max)

    # Weighted combination of sub-indices produces the final synthetic AQI.
    weights = np.array([POLLUTANT_WEIGHTS[p] for p in POLLUTANT_COLUMNS])
    weighted_sum = (sub_indices[POLLUTANT_COLUMNS].values * weights).sum(axis=1)

    df = df.copy()
    df["AQI"] = np.clip(weighted_sum, 0, 500).round(2)

    logger.info(
        "Synthetic AQI target created. Range: [%.2f, %.2f], Mean: %.2f",
        df["AQI"].min(),
        df["AQI"].max(),
        df["AQI"].mean(),
    )
    return df


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def preprocess_pipeline(
    raw_path: str = RAW_DATA_PATH,
) -> pd.DataFrame:
    """
    Run the complete preprocessing + feature engineering pipeline.

    Args:
        raw_path: Path to the raw dataset CSV.

    Returns:
        Fully cleaned DataFrame with the synthetic 'AQI' target column
        included, ready for model training.
    """
    df = load_raw_data(raw_path)
    df = clean_data(df)
    df = engineer_aqi_target(df)
    return df


def get_feature_matrix_and_target(df: pd.DataFrame):
    """
    Split a processed DataFrame into the model's feature matrix (X) and
    target vector (y), using the fixed FEATURE_COLUMNS list.

    Args:
        df: Processed DataFrame (output of preprocess_pipeline).

    Returns:
        Tuple of (X, y) as pandas DataFrame / Series.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected feature columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df["AQI"].copy()

    # Explicitly enforce float dtype on both the feature matrix and target
    # so downstream consumers (StandardScaler, RandomForestRegressor, and
    # the FastAPI inference layer) always receive consistent numeric types.
    X = X.astype(float)
    y = y.astype(float)

    return X, y


# --------------------------------------------------------------------------
# Standalone execution
# --------------------------------------------------------------------------
if __name__ == "__main__":
    processed_df = preprocess_pipeline()

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info("Processed dataset saved to %s", PROCESSED_DATA_PATH)

    logger.info("Preview of processed data:\n%s", processed_df.head())
    logger.info(
        "Feature columns used for training: %s", FEATURE_COLUMNS
    )