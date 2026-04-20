import json
import logging
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/feature_frame.csv"
MODEL_DIR = "models"

FEATURE_COLS = [
    "user_order_seq", "ordered_before", "abandoned_before", "active_snoozed",
    "set_as_regular", "normalised_price", "discount_pct", "global_popularity",
    "count_adults", "count_children", "count_babies", "count_pets",
    "people_ex_baby", "days_since_purchase_variant_id", "avg_days_to_buy_variant_id",
    "std_days_to_buy_variant_id", "days_since_purchase_product_type",
    "avg_days_to_buy_product_type", "std_days_to_buy_product_type",
]
TARGET_COL = "outcome"
MIN_BASKET_SIZE = 5

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
}


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load feature frame and validate expected columns exist."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL, "order_id", "order_date"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to orders with at least MIN_BASKET_SIZE items."""
    items_per_order = df[df[TARGET_COL] == 1].groupby("order_id").size()
    valid = items_per_order[items_per_order >= MIN_BASKET_SIZE].index
    return df[df["order_id"].isin(valid)].copy()


def handler_fit(event: dict, _) -> dict:
    """Train model, save to disk. Returns model path in API-style response."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    params = event.get("model_parametrisation", DEFAULT_PARAMS)

    # Load and preprocess
    df = load_data()
    df = preprocess(df)

    # Temporal split — train on everything before Feb 2021, validate on Feb-Mar
    train = df[df["order_date"] < "2021-02-01"]
    val = df[df["order_date"] >= "2021-02-01"]

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_val, y_val = val[FEATURE_COLS], val[TARGET_COL]

    # Train
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate
    proba = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, proba)
    logger.info(f"Validation PR-AUC: {pr_auc:.4f}")

    # Save with date-based naming as specified in TDD
    os.makedirs(MODEL_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y_%m_%d")
    model_name = f"push_{date_str}"
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    return {
        "statusCode": "200",
        "body": json.dumps({"model_path": model_path}),
    }


if __name__ == "__main__":
    result = handler_fit({"model_parametrisation": DEFAULT_PARAMS}, None)
    print(result)