"""
Data loading for fm_breakeven experiments.

Original 10 datasets + 20 new datasets for extended benchmark.
All datasets download automatically on first use.

New datasets are sourced from:
  - Monash Time Series Forecasting Repository (forecastingdata.org / Zenodo)
  - M4 Competition (GitHub)
  - thuml Time-Series-Library (HuggingFace)

TSF format (Monash) is parsed by the built-in _parse_tsf() function —
no external tsf_loader dependency required.
"""

import os
import io
import zipfile
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

DATA_DIR = os.environ.get('FM_DATA_DIR',
           os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(DATA_DIR, exist_ok=True)


# ── Dataset configuration ──────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    name: str
    freq: str            # pandas-compatible frequency string
    horizon: int         # forecast horizon (steps)
    seasonal_period: int # m for MASE denominator
    target_col: str      # column name in loaded DataFrame
    url: str = ""        # primary download URL
    url_alt: str = ""    # fallback URL
    n_series: int = 1    # for multi-series datasets, number averaged
    drop_constant: bool = False  # remove long constant stretches (see get_target_series)


# ── Original 10 datasets ───────────────────────────────────────────────────────

_ORIGINAL = {
    "ETTh1": DatasetConfig(
        name="ETTh1", freq="h", horizon=24, seasonal_period=24,
        target_col="OT",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    ),
    "ETTh2": DatasetConfig(
        name="ETTh2", freq="h", horizon=24, seasonal_period=24,
        target_col="OT",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
    ),
    "ETTm1": DatasetConfig(
        name="ETTm1", freq="15min", horizon=96, seasonal_period=96,
        target_col="OT",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
    ),
    "ETTm2": DatasetConfig(
        name="ETTm2", freq="15min", horizon=96, seasonal_period=96,
        target_col="OT",
        url="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv"
    ),
    "Electricity": DatasetConfig(
        name="Electricity", freq="h", horizon=24, seasonal_period=24,
        target_col="MT_001",
        url="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv"
    ),
    "Traffic": DatasetConfig(
        name="Traffic", freq="h", horizon=24, seasonal_period=24,
        target_col="0",
        url="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv"
    ),
    "Weather": DatasetConfig(
        name="Weather", freq="10min", horizon=720, seasonal_period=144,
        target_col="T (degC)",
        url="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv"
    ),
    "Exchange": DatasetConfig(
        name="Exchange", freq="D", horizon=30, seasonal_period=5,
        target_col="OT",
        url="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/exchange_rate/exchange_rate.csv"
    ),
    "M4-Hourly": DatasetConfig(
        name="M4-Hourly", freq="h", horizon=24, seasonal_period=24,
        target_col="OT",
        url="https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Hourly-train.csv",
        n_series=50
    ),
    "ILI": DatasetConfig(
        name="ILI", freq="W", horizon=24, seasonal_period=52,
        target_col="OT",
        url="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/illness/national_illness.csv"
    ),
}

# ── 20 new datasets ────────────────────────────────────────────────────────────
# Sourced from Monash Forecasting Repository (Zenodo) and M4 Competition
# Each TSF zip contains one .tsf file; we average across all series.

_MONASH_BASE = "https://zenodo.org/record"

_NEW = {
    # ── M4 additional frequencies ──────────────────────────────────────────────
    "M4-Daily": DatasetConfig(
        name="M4-Daily", freq="D", horizon=14, seasonal_period=7,
        target_col="OT",
        url="https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Daily-train.csv",
        n_series=50
    ),
    "M4-Weekly": DatasetConfig(
        name="M4-Weekly", freq="W", horizon=13, seasonal_period=52,
        target_col="OT",
        url="https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Weekly-train.csv",
        n_series=200  # more series → longer p25 → n_train > 52 (weekly seasonal period)
    ),
    "M4-Monthly-Monash": DatasetConfig(
        name="M4-Monthly-Monash", freq="ME", horizon=18, seasonal_period=12,
        target_col="OT",
        # Loaded by the M4 CSV loader (_M4_NAMES), so it needs the M4 CSV, not
        # the Monash zip (which used to be saved as Monthly-train.csv and fail).
        url="https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv",
        n_series=100
    ),
    "KDD-Cup-Hourly": DatasetConfig(
        name="KDD-Cup-Hourly", freq="h", horizon=48, seasonal_period=24,
        target_col="OT",
        url=f"{_MONASH_BASE}/4656756/files/kdd_cup_2018_dataset_without_missing_values.zip",
        n_series=50
    ),
    # ── Monash datasets (TSF format from Zenodo) ───────────────────────────────
    "NN5-Daily": DatasetConfig(
        name="NN5-Daily", freq="D", horizon=56, seasonal_period=7,
        target_col="OT",
        url=f"{_MONASH_BASE}/4656117/files/nn5_daily_dataset_without_missing_values.zip"
    ),
    "Tourism-Monthly": DatasetConfig(
        name="Tourism-Monthly", freq="ME", horizon=24, seasonal_period=12,
        target_col="OT",
        url=f"{_MONASH_BASE}/4656096/files/tourism_monthly_dataset.zip",
        url_alt=f"{_MONASH_BASE}/4656096/files/tourism_monthly_with_missing_values_dataset.zip"
    ),
    "M3-Quarterly": DatasetConfig(
        name="M3-Quarterly", freq="QE", horizon=8, seasonal_period=4,
        target_col="OT",
        url=f"{_MONASH_BASE}/4656262/files/m3_quarterly_dataset.zip",
        n_series=100
    ),
    "Hospital": DatasetConfig(
        name="Hospital", freq="ME", horizon=12, seasonal_period=12,
        target_col="OT",
        url=f"{_MONASH_BASE}/4656014/files/hospital_dataset.zip"
    ),
    "Solar-10min": DatasetConfig(
        name="Solar-10min", freq="10min", horizon=60, seasonal_period=144,
        target_col="OT",
        url=f"{_MONASH_BASE}/4656144/files/solar_10_minutes_dataset.zip"
    ),
    "Pedestrian": DatasetConfig(
        name="Pedestrian", freq="h", horizon=24, seasonal_period=24,
        target_col="OT",
        url=f"{_MONASH_BASE}/4656626/files/pedestrian_counts_dataset.zip"
    ),
    "Saugeen": DatasetConfig(
        name="Saugeen", freq="D", horizon=30, seasonal_period=365,
        target_col="OT",
        url=f"{_MONASH_BASE}/4656058/files/saugeenday_dataset.zip"
    ),
    "USBirths": DatasetConfig(
        name="USBirths", freq="D", horizon=30, seasonal_period=7,
        target_col="OT",
        url=f"{_MONASH_BASE}/4656049/files/us_births_dataset.zip"
    ),
    "M3-Monthly": DatasetConfig(
        name="M3-Monthly", freq="ME", horizon=18, seasonal_period=12,
        target_col="OT",
        url=f"{_MONASH_BASE}/4656298/files/m3_monthly_dataset.zip"
    ),
    "FRED-MD": DatasetConfig(
        name="FRED-MD", freq="ME", horizon=12, seasonal_period=12,
        target_col="OT",
        url=f"{_MONASH_BASE}/4654833/files/fred_md_dataset.zip"
    ),
    "Dominick": DatasetConfig(
        name="Dominick", freq="W", horizon=8, seasonal_period=52,
        target_col="OT",
        url=f"{_MONASH_BASE}/4654802/files/dominick_dataset.zip"
    ),
    "AusElectricity": DatasetConfig(
        name="AusElectricity", freq="30min", horizon=48, seasonal_period=48,
        target_col="OT",
        url=f"{_MONASH_BASE}/4659727/files/australian_electricity_demand_dataset.zip"
    ),
    "Sunspot": DatasetConfig(
        name="Sunspot", freq="D", horizon=30, seasonal_period=365,
        target_col="OT",
        url=f"{_MONASH_BASE}/4654722/files/sunspot_dataset_without_missing_values.zip"
    ),
    "Rideshare": DatasetConfig(
        name="Rideshare", freq="h", horizon=24, seasonal_period=24,
        target_col="OT",
        url=f"{_MONASH_BASE}/5122232/files/rideshare_dataset_without_missing_values.zip",
        url_alt=f"{_MONASH_BASE}/5122114/files/rideshare_dataset_with_missing_values.zip",
        drop_constant=True,  # ~200 h of COVID-era zero padding in the averaged series
    ),
    "COVID-Deaths": DatasetConfig(
        name="COVID-Deaths", freq="D", horizon=30, seasonal_period=7,
        target_col="OT",
        url=f"{_MONASH_BASE}/4656009/files/covid_deaths_dataset.zip",
        url_alt=f"{_MONASH_BASE}/4656009/files/covid_deaths_dataset.zip"
    ),
    "Wind-Farms": DatasetConfig(
        name="Wind-Farms", freq="min", horizon=60, seasonal_period=1440,
        target_col="OT",
        url=f"{_MONASH_BASE}/4654858/files/wind_farms_minutely_dataset_without_missing_values.zip"
    ),
}

DATASETS: Dict[str, DatasetConfig] = {**_ORIGINAL, **_NEW}
ORIGINAL_DATASETS = list(_ORIGINAL.keys())
NEW_DATASETS      = list(_NEW.keys())
ALL_DATASETS      = list(DATASETS.keys())


# ── Utilities ──────────────────────────────────────────────────────────────────

def download_file(url: str, dest: str, verbose: bool = True) -> bool:
    """Download url → dest, skip if already exists."""
    if os.path.exists(dest):
        if verbose:
            print(f"  [cached] {os.path.basename(dest)}")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        if verbose:
            print(f"  Downloading {os.path.basename(dest)} ...")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        if verbose:
            print(f"  Download failed: {e}")
        return False


def download_zip_extract(url: str, dest_dir: str,
                         verbose: bool = True) -> Optional[str]:
    """
    Download a zip from url, extract to dest_dir, return path of first .tsf file.
    Returns None on failure.
    """
    zip_path = os.path.join(dest_dir, url.split("/")[-1])
    if not download_file(url, zip_path, verbose):
        return None
    try:
        with zipfile.ZipFile(zip_path) as zf:
            tsf_names = [n for n in zf.namelist() if n.endswith('.tsf')]
            if not tsf_names:
                print(f"  No .tsf file found in {zip_path}")
                return None
            tsf_name = tsf_names[0]
            tsf_path = os.path.join(dest_dir, tsf_name)
            if not os.path.exists(tsf_path):
                zf.extract(tsf_name, dest_dir)
            return tsf_path
    except Exception as e:
        print(f"  Zip extract failed: {e}")
        return None


# ── Monash TSF parser ──────────────────────────────────────────────────────────

# pandas < 2.2 (required by gluonts 0.14 / Lag-Llama) does not know the
# period-end aliases ME/QE/YE introduced in pandas 2.2.
_LEGACY_ALIASES = {"ME": "M", "QE": "Q", "YE": "A"}

def pandas_freq(freq: str) -> str:
    """Return a frequency alias the installed pandas accepts."""
    major, minor = (int(x) for x in pd.__version__.split(".")[:2])
    if (major, minor) < (2, 2):
        import re
        m = re.match(r"^(\d*)(.*)$", str(freq))
        return m.group(1) + _LEGACY_ALIASES.get(m.group(2), m.group(2))
    return freq


def _parse_tsf(tsf_path: str) -> Tuple[List[np.ndarray], dict]:
    """
    Parse a Monash .tsf file.
    Returns (list_of_series, metadata_dict).
    Each series is a 1-D numpy float array (NaN-stripped).
    """
    metadata = {}
    series_list: List[np.ndarray] = []

    with open(tsf_path, 'r', encoding='utf-8', errors='replace') as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower() == '@data':
                in_data = True
                continue
            if line.startswith('@'):
                parts = line[1:].split(None, 1)
                key = parts[0].lower()
                val = parts[1] if len(parts) > 1 else ''
                metadata[key] = val.strip()
                continue
            if in_data:
                # Each data line: optional_meta:series_values (colon-separated)
                # Values are comma-separated; some files have no meta prefix
                # Format: series_name:start_timestamp:val1,val2,...
                # or just: val1,val2,...
                parts = line.split(':')
                # Last part is always the values
                value_str = parts[-1]
                try:
                    values = np.array([float(v) for v in value_str.split(',')
                                      if v.strip() not in ('', '?', 'NaN')])
                    if len(values) > 0:
                        series_list.append(values)
                except ValueError:
                    continue

    return series_list, metadata


def _average_series(series_list: List[np.ndarray],
                    n_series: int = 50) -> np.ndarray:
    """
    Average up to n_series series after zero-mean normalisation per-series.
    Uses the minimum common length to allow stacking.
    """
    if not series_list:
        raise ValueError("No series loaded from TSF file")

    # Sample up to n_series uniformly
    if len(series_list) > n_series:
        idx = np.linspace(0, len(series_list) - 1, n_series, dtype=int)
        selected = [series_list[i] for i in idx]
    else:
        selected = series_list

    # Use 25th percentile of lengths to avoid short outliers pulling everything down
    lengths = [len(s) for s in selected]
    p25_len = int(np.percentile(lengths, 25))
    min_len = max(p25_len, 10)
    # Keep only series long enough to contribute
    selected = [s for s in selected if len(s) >= min_len]
    if not selected:
        selected = sorted([s for s in (series_list if len(series_list) <= n_series else
                          [series_list[i] for i in np.linspace(0,len(series_list)-1,n_series,dtype=int)])],
                         key=len, reverse=True)[:1]
        min_len = len(selected[0])

    # Truncate and z-score each, then average
    normed = []
    for s in selected:
        s_trim = s[-min_len:]  # take most recent
        mu, sig = s_trim.mean(), s_trim.std()
        if sig < 1e-8:
            sig = 1.0
        normed.append((s_trim - mu) / sig)

    avg = np.stack(normed).mean(axis=0)
    return avg


# ── M4 CSV loader (multi-series CSV, one row per series) ──────────────────────

def _load_m4_csv(filepath: str, n_series: int = 50) -> np.ndarray:
    """
    Load M4 training CSV (rows = series, cols = values).
    Averages n_series representative series.
    """
    df = pd.read_csv(filepath, index_col=0)
    # Each row is a series; values are in columns V1, V2, ...
    # Drop rows with too many NaNs
    df = df.dropna(how='all')

    # Sample uniformly
    if len(df) > n_series:
        idx = np.linspace(0, len(df) - 1, n_series, dtype=int)
        df = df.iloc[idx]

    series_list = []
    for _, row in df.iterrows():
        vals = row.dropna().values.astype(float)
        if len(vals) >= 30:
            series_list.append(vals)

    if not series_list:
        raise ValueError("No usable M4 series found")

    return _average_series(series_list, n_series=len(series_list))


# ── Dataset-specific loaders ───────────────────────────────────────────────────

def _load_csv_simple(filepath: str, config: DatasetConfig) -> pd.DataFrame:
    """Generic CSV loader: handles 'date' index, returns DataFrame."""
    df = pd.read_csv(filepath)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    elif df.index.dtype == object:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df


def _load_ett(config: DatasetConfig, data_dir: str) -> pd.DataFrame:
    fp = os.path.join(data_dir, f"{config.name}.csv")
    if not os.path.exists(fp):
        if not download_file(config.url, fp):
            raise FileNotFoundError(f"Cannot load {config.name}")
    df = pd.read_csv(fp)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


def _load_thuml(config: DatasetConfig, filename: str,
                data_dir: str) -> pd.DataFrame:
    """Load a CSV from thuml/Time-Series-Library mirror."""
    fp = os.path.join(data_dir, filename)
    if not os.path.exists(fp):
        if not download_file(config.url, fp):
            raise FileNotFoundError(f"Cannot download {filename}")
    df = pd.read_csv(fp)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    return df


def _load_m4_variant(config: DatasetConfig, data_dir: str) -> pd.DataFrame:
    """Load an M4 frequency CSV, average n_series series, return DataFrame."""
    fname = config.name.replace("M4-", "").lower() + "-train.csv"
    # Proper filename as on GitHub
    freq_map = {
        "M4-Hourly":    "Hourly-train.csv",
        "M4-Daily":     "Daily-train.csv",
        "M4-Weekly":    "Weekly-train.csv",
        "M4-Monthly-Monash": "Monthly-train.csv",
            }
    fname = freq_map.get(config.name, fname)
    fp = os.path.join(data_dir, fname)
    if not os.path.exists(fp):
        if not download_file(config.url, fp):
            raise FileNotFoundError(f"Cannot download {fname}")
    y = _load_m4_csv(fp, n_series=config.n_series)
    # Wrap in DataFrame with synthetic date index
    freq_str = {
        "M4-Hourly": "h", "M4-Daily": "D",
        "M4-Weekly": "W",
    }.get(config.name, "D")
    dates = pd.date_range("2000-01-01", periods=len(y), freq=pandas_freq(freq_str))
    return pd.DataFrame({"OT": y}, index=dates)


def _load_monash_tsf(config: DatasetConfig, data_dir: str) -> pd.DataFrame:
    """
    Download Monash zip, parse .tsf, average series, return DataFrame.
    Tries url_alt as fallback if primary url returns 404.
    """
    tsf_path = download_zip_extract(config.url, data_dir)
    if tsf_path is None and config.url_alt:
        print(f"  Trying fallback URL for {config.name}...")
        tsf_path = download_zip_extract(config.url_alt, data_dir)
    if tsf_path is None:
        raise FileNotFoundError(f"Cannot load Monash dataset {config.name}")
    series_list, meta = _parse_tsf(tsf_path)
    print(f"  TSF: {len(series_list)} series, metadata: "
          f"freq={meta.get('frequency','?')}, "
          f"horizon={meta.get('horizon','?')}")
    y = _average_series(series_list, n_series=50)
    dates = pd.date_range("2000-01-01", periods=len(y), freq=pandas_freq(config.freq))
    return pd.DataFrame({"OT": y}, index=dates)


# ── Main loader dispatcher ─────────────────────────────────────────────────────

_MONASH_NAMES = {
    "NN5-Daily", "Tourism-Monthly", "M3-Quarterly", "Hospital",
    "Solar-10min", "Pedestrian", "Saugeen", "USBirths", "M3-Monthly",
    "FRED-MD", "Dominick", "AusElectricity", "Sunspot",
    "Rideshare", "COVID-Deaths", "Wind-Farms",
    "M4-Monthly-Monash", "KDD-Cup-Hourly",
}

_M4_NAMES = {"M4-Hourly", "M4-Daily", "M4-Weekly", "M4-Monthly-Monash"}


def load_dataset(name: str, data_dir: str = DATA_DIR) \
        -> Tuple[pd.DataFrame, DatasetConfig]:
    """Load dataset by name. Downloads on first call."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Available: {sorted(DATASETS.keys())}")
    config = DATASETS[name]
    print(f"Loading {name}...")

    if name.startswith("ETT"):
        df = _load_ett(config, data_dir)

    elif name == "Electricity":
        df = _load_thuml(config, "electricity.csv", data_dir)
        df.columns = [f"MT_{i:03d}" for i in range(len(df.columns))]

    elif name == "Traffic":
        df = _load_thuml(config, "traffic.csv", data_dir)
        df.columns = [str(i) for i in range(len(df.columns))]

    elif name == "Weather":
        df = _load_thuml(config, "weather.csv", data_dir)

    elif name == "Exchange":
        df = _load_thuml(config, "exchange_rate.csv", data_dir)
        # Ensure OT column exists (first value column)
        if "OT" not in df.columns:
            df = df.rename(columns={df.columns[0]: "OT"})

    elif name == "ILI":
        df = _load_thuml(config, "national_illness.csv", data_dir)
        if "OT" not in df.columns:
            df = df.rename(columns={df.columns[0]: "OT"})

    elif name in _M4_NAMES:
        df = _load_m4_variant(config, data_dir)

    elif name in _MONASH_NAMES:
        df = _load_monash_tsf(config, data_dir)

    else:
        raise ValueError(f"No loader defined for '{name}'")

    print(f"  Shape: {df.shape}  |  "
          f"Range: {df.index[0]} → {df.index[-1]}")
    return df, config


def drop_constant_stretches(y: np.ndarray, min_len: int, merge_gap: int) -> np.ndarray:
    """Remove runs of >= min_len identical consecutive values. Runs separated by
    fewer than merge_gap points are removed together with the gap between them."""
    same = np.concatenate([[False], np.abs(np.diff(y)) < 1e-12])
    runs, i, n = [], 0, len(y)
    while i < n:
        if same[i]:
            j = i
            while j < n and same[j]:
                j += 1
            start, end = i - 1, j          # [start, end) are identical values
            if end - start >= min_len:
                if runs and start - runs[-1][1] < merge_gap:
                    runs[-1][1] = end
                else:
                    runs.append([start, end])
            i = j
        else:
            i += 1
    keep = np.ones(n, dtype=bool)
    for a, b in runs:
        keep[a:b] = False
    return y[keep]


def get_target_series(df: pd.DataFrame, config: DatasetConfig) -> np.ndarray:
    """Extract univariate target series.

    For datasets with drop_constant=True (Rideshare), constant stretches of at
    least two seasonal periods are removed before the train/test split: they are
    padding in the source data, and a constant training window makes the MASE
    denominator zero and the classical fits degenerate.
    """
    if config.target_col in df.columns:
        y = df[config.target_col].values.astype(float)
    else:  # fall back to first column
        y = df.iloc[:, 0].values.astype(float)
    if config.drop_constant:
        y = drop_constant_stretches(y, min_len=2 * config.seasonal_period,
                                    merge_gap=config.seasonal_period)
    return y


def prepare_splits(y: np.ndarray,
                   train_ratio: float = 0.6,
                   val_ratio: float = 0.2
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological 60/20/20 split."""
    n = len(y)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return y[:t1], y[t1:t2], y[t2:]


def subsample_train(y_train: np.ndarray, fraction: float) -> np.ndarray:
    """Take the most recent fraction of training data."""
    n_keep = max(1, int(len(y_train) * fraction))
    return y_train[-n_keep:]


def make_test_windows(n_test: int, horizon: int, n_windows: int = 5):
    """Start offsets of up to `n_windows` evenly spaced, non-overlapping
    forecast windows inside a test segment of length `n_test`."""
    h = min(horizon, n_test)
    last = n_test - h
    if last <= 0:
        return [0], h
    k = max(1, min(n_windows, last // h + 1))
    starts = sorted(set(int(round(v)) for v in np.linspace(0, last, k)))
    return starts, h


def load_and_prepare(name: str,
                     train_fraction: float = 1.0,
                     data_dir: str = DATA_DIR,
                     n_windows: int = 5) -> Dict:
    """
    Load dataset and prepare splits for a single experiment.

    Chronological 60/20/20 split. Models are trained on the most recent
    `train_fraction` of the first 60% (the training pool). The middle 20%
    is NOT used for training: it is context, i.e. recent history the models
    condition on before forecasting. Every forecast therefore starts at (or
    after) the 80% mark, directly before the window it is scored on.

    Returns dict with keys:
        y_train       training data (fraction of the pool)
        y_train_full  full training pool (first 60%)
        windows       list of (context, target) pairs; context = all history
                      before the window (pool + buffer + earlier test points)
        y_test        target of the first window (kept for compatibility)
        train_start   absolute index of y_train[0] in the series (contexts
                      start at index 0)
        config, n_train, n_train_full, scaler
    """
    df, config = load_dataset(name, data_dir)
    y = get_target_series(df, config)

    # Remove NaN / Inf
    y = y[np.isfinite(y)]
    min_required = max(config.seasonal_period, 15)  # lowered to allow short competition series
    if len(y) < min_required:
        raise ValueError(
            f"{name}: only {len(y)} valid points, need >= {min_required} "
            f"(seasonal_period={config.seasonal_period} + horizon={config.horizon} + 10)"
        )

    y_train_full, y_buffer, y_test_full = prepare_splits(y)

    # Z-score normalisation (fit on the training pool only)
    mu, sigma = y_train_full.mean(), y_train_full.std()
    if sigma < 1e-8:
        sigma = 1.0
    y_train_full = (y_train_full - mu) / sigma
    y_buffer     = (y_buffer     - mu) / sigma
    y_test_full  = (y_test_full  - mu) / sigma

    y_train = subsample_train(y_train_full, train_fraction)

    # History available before the test segment: training pool + buffer
    y_hist = np.concatenate([y_train_full, y_buffer])
    starts, h = make_test_windows(len(y_test_full), config.horizon, n_windows)
    windows = [(np.concatenate([y_hist, y_test_full[:s]]), y_test_full[s:s + h])
               for s in starts]

    return {
        'y_train':      y_train,
        'y_train_full': y_train_full,
        'windows':      windows,
        'y_test':       windows[0][1],
        'train_start':  len(y_train_full) - len(y_train),
        'config':       config,
        'n_train':      len(y_train),
        'n_train_full': len(y_train_full),
        'scaler':       {'mean': mu, 'std': sigma},
    }


AVAILABLE_DATASETS = ALL_DATASETS


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] if len(sys.argv) > 1 else ORIGINAL_DATASETS

    print("=" * 60)
    print(f"Testing {len(names)} datasets")
    print("=" * 60)
    ok, fail = [], []
    for name in names:
        try:
            d = load_and_prepare(name, train_fraction=0.1)
            print(f"  ✓  n_train(10%)={d['n_train']:5d}  "
                  f"n_test={len(d['y_test']):4d}  "
                  f"horizon={d['config'].horizon}")
            ok.append(name)
        except Exception as e:
            print(f"  ✗  {name}: {e}")
            fail.append(name)
    print(f"\nPassed: {len(ok)}/{len(names)}")
    if fail:
        print(f"Failed: {fail}")
