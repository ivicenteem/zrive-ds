import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/feature_frame.csv"
MODEL_DIR = "models"

FEATURE_COLS = [
    "user_order_seq",
    "ordered_before",
    "abandoned_before",
    "active_snoozed",
    "set_as_regular",
    "normalised_price",
    "discount_pct",
    "global_popularity",
    "count_adults",
    "count_children",
    "count_babies",
    "count_pets",
    "people_ex_baby",
    "days_since_purchase_variant_id",
    "avg_days_to_buy_variant_id",
    "std_days_to_buy_variant_id",
    "days_since_purchase_product_type",
    "avg_days_to_buy_product_type",
    "std_days_to_buy_product_type",
]

TARGET_COL = "outcome"
MIN_BASKET_SIZE = 5


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load feature frame CSV and apply basic validations."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)

    expected_cols = FEATURE_COLS + [TARGET_COL, "order_id", "order_date"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["order_date"] = pd.to_datetime(df["order_date"])
    logger.info(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def filter_min_basket(df: pd.DataFrame, min_items: int = MIN_BASKET_SIZE) -> pd.DataFrame:
    """Keep only orders where the user purchased at least min_items products."""
    items_per_order = df[df[TARGET_COL] == 1].groupby("order_id").size()
    valid_orders = items_per_order[items_per_order >= min_items].index
    df_filtered = df[df["order_id"].isin(valid_orders)].copy()
    logger.info(
        f"Filtered to {df_filtered.shape[0]:,} rows "
        f"({df_filtered['order_id'].nunique()} orders with >= {min_items} items)"
    )
    return df_filtered


def temporal_split(
    df: pd.DataFrame,
    train_end: str = "2021-01-01",
    val_end: str = "2021-02-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by order_date to prevent information leakage."""
    train = df[df["order_date"] < train_end].copy()
    val = df[(df["order_date"] >= train_end) & (df["order_date"] < val_end)].copy()
    test = df[df["order_date"] >= val_end].copy()

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        logger.info(
            f"{name}: {split.shape[0]:,} rows, "
            f"{split['order_id'].nunique()} orders, "
            f"positive rate: {split[TARGET_COL].mean():.4f}"
        )
    return train, val, test


def build_model(penalty: str = "l1", C: float = 0.01) -> Pipeline:
    """Create a sklearn pipeline with scaling and logistic regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty=penalty,
            C=C,
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )),
    ])


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute PR-AUC and ROC-AUC for a fitted model."""
    proba = model.predict_proba(X)[:, 1]
    return {
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
    }


def save_model(model: Pipeline, model_dir: str = MODEL_DIR) -> str:
    """Save model to disk with timestamp for versioning."""
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lr_l1_C0.01_{timestamp}.joblib"
    path = os.path.join(model_dir, filename)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path


def train_pipeline() -> None:
    """End-to-end training pipeline: load, preprocess, train, evaluate, save."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    # 1. Data loading
    df = load_data()

    # 2. Preprocessing
    df = filter_min_basket(df)
    train, val, test = temporal_split(df)

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_val, y_val = val[FEATURE_COLS], val[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    # 3. Model training and selection
    configs = [
        ("L2_C0.01", "l2", 0.01),
        ("L2_C1.0", "l2", 1.0),
        ("L1_C0.01", "l1", 0.01),
        ("L1_C1.0", "l1", 1.0),
    ]

    best_model = None
    best_pr_auc = -1.0
    best_name = ""

    for name, penalty, C in configs:
        model = build_model(penalty=penalty, C=C)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_val, y_val)
        logger.info(f"{name}: PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_model = model
            best_name = name

    logger.info(f"Best model on validation: {best_name} (PR-AUC={best_pr_auc:.4f})")

    # 4. Final evaluation on test (once)
    test_metrics = evaluate(best_model, X_test, y_test)
    logger.info(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}, ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # 5. Save best model
    save_model(best_model)