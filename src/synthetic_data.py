"""
Synthetic data generator that mimics real dataset characteristics.

Creates datasets with similar statistical properties to:
- ETTh1/ETTh2: Energy data with daily/weekly seasonality
- Exchange: Financial data with high persistence, low seasonality
- Traffic: High autocorrelation, complex patterns
- Weather: Strong seasonality, moderate noise
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, DATASETS


def generate_ar_process(n: int, phi: float, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    """Generate AR(1) process."""
    np.random.seed(seed)
    y = np.zeros(n)
    y[0] = np.random.normal(0, sigma)
    for t in range(1, n):
        y[t] = phi * y[t-1] + np.random.normal(0, sigma)
    return y


def generate_seasonal_pattern(n: int, period: int, amplitude: float = 1.0) -> np.ndarray:
    """Generate deterministic seasonal pattern."""
    t = np.arange(n)
    return amplitude * np.sin(2 * np.pi * t / period)


def generate_trend(n: int, slope: float = 0.001, curvature: float = 0.0) -> np.ndarray:
    """Generate trend component."""
    t = np.arange(n)
    return slope * t + curvature * t**2


def generate_ett_like(n: int = 17420, seed: int = 42) -> pd.DataFrame:
    """Generate ETT-like energy data.
    
    Characteristics:
    - Hourly frequency
    - Strong daily seasonality (period=24)
    - Moderate weekly seasonality (period=168)
    - Low noise
    - Moderate autocorrelation
    """
    np.random.seed(seed)
    
    t = np.arange(n)
    
    # Base level
    base = 50
    
    # Daily seasonality (temperature peaks in afternoon)
    daily = 10 * np.sin(2 * np.pi * (t - 6) / 24)  # Peak at hour 14
    
    # Weekly seasonality (lower on weekends)
    weekly = 3 * np.sin(2 * np.pi * t / 168)
    
    # Yearly seasonality (if long enough)
    yearly = 15 * np.sin(2 * np.pi * t / 8760) if n > 8760 else np.zeros(n)
    
    # AR(1) residual for autocorrelation
    ar_component = generate_ar_process(n, phi=0.7, sigma=2, seed=seed)
    
    # Small noise
    noise = np.random.normal(0, 1, n)
    
    # Combine
    oil_temp = base + daily + weekly + yearly + ar_component + noise
    
    # Add some correlated features
    load = 0.8 * oil_temp + 20 + np.random.normal(0, 3, n)
    
    # Create DataFrame
    dates = pd.date_range(start='2016-01-01', periods=n, freq='h')
    df = pd.DataFrame({
        'date': dates,
        'OT': oil_temp,
        'HUFL': load + np.random.normal(0, 2, n),
        'HULL': load * 0.9 + np.random.normal(0, 2, n),
        'MUFL': load * 0.8 + np.random.normal(0, 2, n),
        'MULL': load * 0.7 + np.random.normal(0, 2, n),
        'LUFL': load * 0.6 + np.random.normal(0, 2, n),
        'LULL': load * 0.5 + np.random.normal(0, 2, n),
    })
    df = df.set_index('date')
    
    return df


def generate_exchange_like(n: int = 7588, seed: int = 42) -> pd.DataFrame:
    """Generate Exchange-like financial data.
    
    Characteristics:
    - Daily frequency
    - Near random walk (high persistence)
    - Very weak/no seasonality
    - High noise ratio
    - Long memory (high Hurst exponent)
    """
    np.random.seed(seed)
    
    n_currencies = 8
    data = {}
    
    for i in range(n_currencies):
        # Start with price level
        base_price = 1.0 + 0.2 * i
        
        # Random walk with small returns (realistic FX volatility ~0.5% daily)
        returns = np.random.normal(0, 0.005, n)
        
        # Add slight mean reversion to prevent explosion
        price = np.zeros(n)
        price[0] = base_price
        
        for t in range(1, n):
            # Mean-reverting random walk
            mean_revert = -0.001 * (price[t-1] - base_price)
            price[t] = price[t-1] * (1 + returns[t] + mean_revert)
        
        # Ensure positive
        price = np.maximum(price, 0.1)
        
        data[f'currency_{i}'] = price
    
    dates = pd.date_range(start='1990-01-01', periods=n, freq='D')
    df = pd.DataFrame(data, index=dates)
    
    return df


def generate_traffic_like(n: int = 17544, seed: int = 42) -> pd.DataFrame:
    """Generate Traffic-like transportation data.
    
    Characteristics:
    - Hourly frequency
    - Complex patterns (rush hours)
    - High autocorrelation
    - Moderate seasonality with sharp peaks
    """
    np.random.seed(seed)
    
    t = np.arange(n)
    hours = t % 24
    days = (t // 24) % 7
    
    # Base occupancy
    base = 0.3
    
    # Rush hour patterns (morning and evening peaks)
    morning_rush = 0.4 * np.exp(-0.5 * ((hours - 8) / 2)**2)
    evening_rush = 0.5 * np.exp(-0.5 * ((hours - 17) / 2)**2)
    
    # Weekend reduction
    weekend = np.where((days == 5) | (days == 6), -0.2, 0)
    
    # Long-range dependence via fractional noise
    ar_long = generate_ar_process(n, phi=0.95, sigma=0.05, seed=seed)
    
    # Noise
    noise = np.random.normal(0, 0.03, n)
    
    # Combine and clip to [0, 1]
    occupancy = base + morning_rush + evening_rush + weekend + ar_long + noise
    occupancy = np.clip(occupancy, 0, 1)
    
    dates = pd.date_range(start='2015-01-01', periods=n, freq='h')
    
    # Multiple sensors
    n_sensors = 10
    data = {'sensor_0': occupancy}
    for i in range(1, n_sensors):
        # Correlated sensors with slight variations
        sensor_data = occupancy + np.random.normal(0, 0.02, n)
        sensor_data = np.clip(sensor_data, 0, 1)
        data[f'sensor_{i}'] = sensor_data
    
    df = pd.DataFrame(data, index=dates)
    
    return df


def generate_weather_like(n: int = 52560, seed: int = 42) -> pd.DataFrame:
    """Generate Weather-like meteorological data.
    
    Characteristics:
    - 10-minute frequency
    - Strong daily and yearly seasonality
    - Moderate noise
    - Correlated variables
    """
    np.random.seed(seed)
    
    t = np.arange(n)
    
    # Daily pattern (144 points per day at 10min)
    daily_period = 144
    daily = 8 * np.sin(2 * np.pi * (t - daily_period/4) / daily_period)
    
    # Yearly pattern (if long enough)
    yearly_period = 144 * 365
    if n > yearly_period:
        yearly = 12 * np.sin(2 * np.pi * t / yearly_period)
    else:
        yearly = 12 * np.sin(2 * np.pi * t / (n * 0.8))
    
    # AR component
    ar = generate_ar_process(n, phi=0.8, sigma=1.5, seed=seed)
    
    # Noise
    noise = np.random.normal(0, 1, n)
    
    # Temperature
    temp = 15 + daily + yearly + ar + noise
    
    # Humidity (inversely related to temperature)
    humidity = 70 - 0.5 * (temp - 15) + np.random.normal(0, 5, n)
    humidity = np.clip(humidity, 20, 100)
    
    # Pressure
    pressure = 1013 + 0.3 * generate_ar_process(n, phi=0.95, sigma=3, seed=seed+1)
    
    dates = pd.date_range(start='2020-01-01', periods=n, freq='10min')
    df = pd.DataFrame({
        'temperature': temp,
        'humidity': humidity,
        'pressure': pressure,
    }, index=dates)
    
    return df


def generate_m4_hourly_like(n: int = 960, n_series: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate M4-Hourly-like mixed data.
    
    Characteristics:
    - Hourly frequency
    - Mixed patterns (some seasonal, some trending)
    - Short series (748-1008 points)
    - Lower autocorrelation on average
    """
    np.random.seed(seed)
    
    all_series = []
    
    for i in range(n_series):
        series_seed = seed + i
        np.random.seed(series_seed)
        
        # Random characteristics
        has_trend = np.random.random() > 0.5
        has_seasonality = np.random.random() > 0.3
        ar_coef = np.random.uniform(0.3, 0.7)
        noise_level = np.random.uniform(0.5, 2.0)
        
        t = np.arange(n)
        y = np.zeros(n)
        
        # Base level
        base = np.random.uniform(50, 200)
        
        if has_trend:
            slope = np.random.uniform(-0.01, 0.01)
            y += slope * t
        
        if has_seasonality:
            amplitude = np.random.uniform(5, 20)
            y += amplitude * np.sin(2 * np.pi * t / 24)
        
        # AR component
        y += generate_ar_process(n, phi=ar_coef, sigma=noise_level, seed=series_seed)
        
        y += base
        
        all_series.append(y)
    
    # Use mean of series as representative
    dates = pd.date_range(start='2017-01-01', periods=n, freq='h')
    df = pd.DataFrame({f'series_{i}': s for i, s in enumerate(all_series)}, index=dates)
    
    return df



def generate_electricity_like(n: int = 15782, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Electricity-like dataset.
    
    Properties:
    - Hourly frequency
    - Strong daily seasonality (period=24)
    - Moderate weekly seasonality (period=168)
    - Multiple overlapping consumption patterns
    """
    rng = np.random.RandomState(seed)
    
    # Daily seasonality (peak morning and evening)
    daily = generate_seasonal_pattern(n, 24, amplitude=1.5)
    daily = np.where(np.arange(n) % 24 < 6, daily * 0.6, daily)  # low at night
    
    # Weekly seasonality (lower on weekends)
    weekly = generate_seasonal_pattern(n, 168, amplitude=0.4)
    
    # Slow upward trend with noise
    trend = np.linspace(0, 0.3, n)
    noise = rng.randn(n) * 0.2
    
    value = 2.0 + daily + weekly + trend + noise
    value = np.maximum(value, 0.1)  # electricity consumption is non-negative
    
    dates = pd.date_range(start='2012-01-01', periods=n, freq='h')
    df = pd.DataFrame({'value': value}, index=dates)
    return df

# Dataset generators mapping
GENERATORS = {
    'ETTh1': lambda seed: generate_ett_like(n=17420, seed=seed),
    'ETTh2': lambda seed: generate_ett_like(n=17420, seed=seed+100),
    'ETTm1': lambda seed: generate_ett_like(n=69680, seed=seed+200),
    'ETTm2': lambda seed: generate_ett_like(n=69680, seed=seed+300),
    'Exchange': lambda seed: generate_exchange_like(n=7588, seed=seed),
    'Traffic': lambda seed: generate_traffic_like(n=17544, seed=seed),
    'Weather': lambda seed: generate_weather_like(n=52560, seed=seed),
    'Electricity': lambda seed: generate_electricity_like(n=15782, seed=seed),
    'M4-Hourly': lambda seed: generate_m4_hourly_like(n=960, n_series=50, seed=seed),
}


def generate_all_datasets(seed: int = 42, save: bool = True) -> Dict[str, pd.DataFrame]:
    """Generate all synthetic datasets.
    
    Returns:
        Dictionary of dataset name -> DataFrame
    """
    datasets = {}
    
    for name, generator in GENERATORS.items():
        print(f"Generating {name}...")
        df = generator(seed)
        datasets[name] = df
        
        if save:
            filepath = os.path.join(DATA_DIR, f"{name}_synthetic.csv")
            df.to_csv(filepath)
            print(f"  Saved to {filepath}")
            print(f"  Shape: {df.shape}")
    
    return datasets


# Mapping from dataset name to real data filename (all use OT as target column)
REAL_DATA_FILES = {
    'ETTh1':       'ETTh1.csv',
    'ETTh2':       'ETTh2.csv',
    'ETTm1':       'ETTm1.csv',
    'ETTm2':       'ETTm2.csv',
    'Electricity': 'electricity.csv',
    'Exchange':    'exchange_rate.csv',
    'Traffic':     'traffic.csv',
    'Weather':     'weather.csv',
    'ILI':         'ili.csv',
}

def load_synthetic_dataset(name: str) -> Tuple[pd.DataFrame, dict]:
    """Load dataset — uses real data if available, otherwise generates synthetic."""
    # Prefer real data
    if name in REAL_DATA_FILES:
        real_path = os.path.join(DATA_DIR, REAL_DATA_FILES[name])
        if os.path.exists(real_path):
            df_raw = pd.read_csv(real_path, index_col=0, parse_dates=True)
            # Use same target column as classical data_loader.py for consistency
            TARGET_COLS = {
                'ETTh1': 'OT', 'ETTh2': 'OT', 'ETTm1': 'OT', 'ETTm2': 'OT',
                'Exchange': 'OT',
                'Electricity': 'MT_001',  # first meter — but electricity.csv uses numeric cols
                'Traffic': '0',           # first sensor
                'Weather': 'T (degC)',    # temperature
            }
            target = TARGET_COLS.get(name, 'OT')
            if target not in df_raw.columns:
                target = 'OT' if 'OT' in df_raw.columns else df_raw.columns[0]
            df = df_raw[[target]].rename(columns={target: 'value'})
            config = DATASETS.get(name)
            return df, config

    # M4-Hourly: synthetic generates 50 series — average to single univariate series
    if name == 'M4-Hourly':
        filepath = os.path.join(DATA_DIR, f"{name}_synthetic.csv")
        if not os.path.exists(filepath):
            print(f"Generating synthetic dataset: {name}")
            df_multi = GENERATORS[name](seed=42)
            df_multi.to_csv(filepath)
        else:
            df_multi = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df = df_multi.mean(axis=1).to_frame(name='value')
        config = DATASETS.get(name)
        return df, config

    # Fall back to synthetic
    filepath = os.path.join(DATA_DIR, f"{name}_synthetic.csv")
    if not os.path.exists(filepath):
        print(f"Generating synthetic dataset: {name}")
        if name not in GENERATORS:
            raise ValueError(f"Unknown dataset: {name}")
        df = GENERATORS[name](seed=42)
        df.to_csv(filepath)
    else:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Get config
    config = DATASETS.get(name, None)
    if config is None:
        # Create default config
        config = {
            'horizon': 24,
            'seasonal_period': 24,
            'freq': 'H'
        }
    else:
        config = {
            'horizon': config.horizon,
            'seasonal_period': config.seasonal_period,
            'freq': config.freq
        }
    
    return df, config


if __name__ == "__main__":
    # Generate all datasets
    print("Generating synthetic datasets...")
    print("=" * 60)
    
    datasets = generate_all_datasets(seed=42, save=True)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    for name, df in datasets.items():
        print(f"\n{name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Columns: {list(df.columns)[:5]}...")
        
        # Quick stats
        target = df.iloc[:, 0]
        print(f"  Target mean: {target.mean():.2f}")
        print(f"  Target std: {target.std():.2f}")
