import json
import logging

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "user_order_seq", "ordered_before", "abandoned_before", "active_snoozed",
    "set_as_regular", "normalised_price", "discount_pct", "global_popularity",
    "count_adults", "count_children", "count_babies", "count_pets",
    "people_ex_baby", "days_since_purchase_variant_id", "avg_days_to_buy_variant_id",
    "std_days_to_buy_variant_id", "days_since_purchase_product_type",
    "avg_days_to_buy_product_type", "std_days_to_buy_product_type",
]

DEFAULT_MODEL_PATH = "models/push_latest.joblib"


def handler_predict(event: dict, _) -> dict:
    """Load model, predict purchase probability for each user."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    model_path = event.get("model_path", DEFAULT_MODEL_PATH)
    model = joblib.load(model_path)

    users_data = json.loads(event["users"])
    df = pd.DataFrame.from_dict(users_data, orient="index")

    # Validate expected features
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return {
            "statusCode": "400",
            "body": json.dumps({"error": f"Missing features: {missing}"}),
        }

    predictions = model.predict_proba(df[FEATURE_COLS])[:, 1]
    result = dict(zip(df.index, predictions.round(6)))

    return {
        "statusCode": "200",
        "body": json.dumps({"prediction": result}),
    }