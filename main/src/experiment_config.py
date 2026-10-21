from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "power_data_pricing.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

TARGET_COL = "dynamic_price"
TIME_COL = "timestamp"

RAW_FEATURES = [
    "load_mw",
    "load_forecast_mw",
    "electricity_price",
    "renewable_generation_mw",
]

ASSET_FEATURES = [
    "quality_score",
    "timeliness_score",
]

MARKET_FEEDBACK_FEATURES = [
    "market_demand",
    "risk_score",
]

VALUE_FEATURES = [
    "base_value",
]

FEATURE_SETS = {
    "raw": RAW_FEATURES,
    "asset": RAW_FEATURES + ASSET_FEATURES,
    "feedback": RAW_FEATURES + ASSET_FEATURES + MARKET_FEEDBACK_FEATURES,
    "all": RAW_FEATURES + ASSET_FEATURES + MARKET_FEEDBACK_FEATURES + VALUE_FEATURES,
}


@dataclass(frozen=True)
class ExperimentConfig:
    data_path: Path = DATA_PATH
    results_dir: Path = RESULTS_DIR
    target_col: str = TARGET_COL
    time_col: str = TIME_COL
    test_size: float = 0.2
    random_state: int = 42
