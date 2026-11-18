"""
Configuration for Foundation Model Break-Even Experiments
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Ensure directories exist
for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Training size fractions (log-spaced)
TRAIN_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]

# Random seeds for variance estimation
SEEDS = [0, 1, 2]

# Dataset configurations
@dataclass
class DatasetConfig:
    name: str
    freq: str
    horizon: int
    seasonal_period: int
    source: str  # 'monash', 'ett', 'custom'
    url: str = ""
    
DATASETS = {
    "ETTh1": DatasetConfig(
        name="ETTh1",
        freq="h",
        horizon=24,
        seasonal_period=24,
        source="ett",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    ),
    "ETTh2": DatasetConfig(
        name="ETTh2", 
        freq="h",
        horizon=24,
        seasonal_period=24,
        source="ett",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
    ),
    "ETTm1": DatasetConfig(
        name="ETTm1",
        freq="15min",
        horizon=96,
        seasonal_period=96,
        source="ett",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
    ),
    "ETTm2": DatasetConfig(
        name="ETTm2",
        freq="15min",
        horizon=96,
        seasonal_period=96,
        source="ett",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv"
    ),
    "Exchange": DatasetConfig(
        name="Exchange",
        freq="D",
        horizon=30,
        seasonal_period=5,  # weekly pattern (5 trading days)
        source="custom",
        url="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt"
    ),
    "ILI": DatasetConfig(
        name="ILI",
        freq="W",
        horizon=24,
        seasonal_period=52,
        source="custom",
        url=""
    ),
    "Traffic": DatasetConfig(
        name="Traffic",
        freq="h",
        horizon=24,
        seasonal_period=24,
        source="custom",
        url=""
    ),
    "M4-Hourly": DatasetConfig(
        name="M4-Hourly",
        freq="h",
        horizon=24,
        seasonal_period=24,
        source="custom",
        url=""
    ),
    "Electricity": DatasetConfig(
        name="Electricity",
        freq="h",
        horizon=24,
        seasonal_period=24,
        source="custom",
        url=""
    ),
    "Weather": DatasetConfig(
        name="Weather",
        freq="10min",
        horizon=720,          # 5 days — matches data_loader.py standard benchmark
        seasonal_period=144,  # daily pattern at 10min frequency
        source="custom",
        url=""
    ),
}

# Model configurations
@dataclass 
class ModelConfig:
    name: str
    model_type: str  # 'classical' or 'foundation'
    requires_gpu: bool
    params: Dict[str, Any] = field(default_factory=dict)

CLASSICAL_MODELS = {
    "Naive": ModelConfig(
        name="Naive",
        model_type="classical",
        requires_gpu=False,
        params={"seasonal": True}
    ),
    "ETS": ModelConfig(
        name="ETS",
        model_type="classical",
        requires_gpu=False,
        params={"season_length": "auto"}
    ),
    "ARIMA": ModelConfig(
        name="ARIMA",
        model_type="classical",
        requires_gpu=False,
        params={"season_length": "auto"}
    ),
    "XGBoost": ModelConfig(
        name="XGBoost",
        model_type="classical",
        requires_gpu=False,
        params={
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_lags": 24,
        }
    ),
}

FOUNDATION_MODELS = {
    "Chronos-Bolt-S": ModelConfig(
        name="Chronos-Bolt-S",
        model_type="foundation",
        requires_gpu=True,
        params={"model_path": "amazon/chronos-bolt-small"}
    ),
    "Chronos-Bolt-B": ModelConfig(
        name="Chronos-Bolt-B",
        model_type="foundation", 
        requires_gpu=True,
        params={"model_path": "amazon/chronos-bolt-base"}
    ),
    "Chronos-Bolt-L": ModelConfig(
        name="Chronos-Bolt-L",
        model_type="foundation",
        requires_gpu=True,
        params={"model_path": "amazon/chronos-bolt-large"}
    ),
    "Moirai-S": ModelConfig(
        name="Moirai-S",
        model_type="foundation",
        requires_gpu=True,
        params={"model_path": "Salesforce/moirai-1.0-R-small"}
    ),
    "Moirai-B": ModelConfig(
        name="Moirai-B",
        model_type="foundation",
        requires_gpu=True,
        params={"model_path": "Salesforce/moirai-1.0-R-base"}
    ),
    "Moirai-L": ModelConfig(
        name="Moirai-L",
        model_type="foundation",
        requires_gpu=True,
        params={"model_path": "Salesforce/moirai-1.0-R-large"}
    ),
    "Lag-Llama": ModelConfig(
        name="Lag-Llama",
        model_type="foundation",
        requires_gpu=True,
        params={"model_path": "time-series-foundation-models/Lag-Llama"}
    ),
}

ALL_MODELS = {**CLASSICAL_MODELS, **FOUNDATION_MODELS}
