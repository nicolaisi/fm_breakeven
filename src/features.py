"""
Dataset feature extraction for break-even prediction.

Features:
- ACF decay (τ_acf): Lag where |ACF| < 0.5
- Hurst exponent (H): Long-range dependence measure
- Noise ratio (σ_r): std(diff) / std(original)
- Seasonal strength (S): From STL decomposition
- Spectral entropy (Φ): Entropy of normalized periodogram
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats
from scipy.fft import fft


def compute_acf(y: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Compute autocorrelation function."""
    n = len(y)
    y = y - np.mean(y)
    
    acf = np.correlate(y, y, mode='full')
    acf = acf[n-1:]  # Take positive lags
    acf = acf / acf[0]  # Normalize
    
    return acf[:min(max_lag, n)]


def acf_decay_lag(y: np.ndarray, threshold: float = 0.5, max_lag: int = 100) -> float:
    """Find smallest lag where |ACF| < threshold.
    
    Returns:
        τ_acf: Lag where ACF first drops below threshold
    """
    acf = compute_acf(y, max_lag)
    
    for lag, val in enumerate(acf):
        if abs(val) < threshold:
            return float(lag)
    
    return float(max_lag)


def hurst_exponent(y: np.ndarray, max_lag: int = 100) -> float:
    """Estimate Hurst exponent via R/S analysis.
    
    H = 0.5: Random walk (no memory)
    H > 0.5: Persistent (positive autocorrelation)
    H < 0.5: Anti-persistent (negative autocorrelation)
    """
    n = len(y)
    if n < 20:
        return 0.5  # Default for very short series
    
    # Use range of window sizes
    lags = []
    rs_values = []
    
    for lag in range(10, min(max_lag, n // 4)):
        # Compute R/S for this window size
        rs_list = []
        for start in range(0, n - lag, lag):
            window = y[start:start + lag]
            if len(window) < 2:
                continue
            
            # Mean-adjusted cumulative sum
            mean_adj = window - np.mean(window)
            cumsum = np.cumsum(mean_adj)
            
            # Range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Standard deviation
            S = np.std(window, ddof=1)
            
            if S > 0:
                rs_list.append(R / S)
        
        if rs_list:
            lags.append(lag)
            rs_values.append(np.mean(rs_list))
    
    if len(lags) < 2:
        return 0.5
    
    # Fit log-log regression
    log_lags = np.log(lags)
    log_rs = np.log(rs_values)
    
    slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
    
    # Hurst exponent is the slope
    return np.clip(slope, 0.0, 1.0)


def noise_ratio(y: np.ndarray) -> float:
    """Compute noise ratio: std(diff) / std(original).
    
    Higher values indicate noisier series dominated by innovations.
    Value near sqrt(2) ≈ 1.41 suggests random walk.
    """
    if len(y) < 2:
        return 1.0
    
    diff_std = np.std(np.diff(y))
    orig_std = np.std(y)
    
    if orig_std == 0:
        return 1.0
    
    return diff_std / orig_std


def seasonal_strength(y: np.ndarray, period: int) -> float:
    """Estimate seasonal strength via variance ratio.
    
    Uses a simple seasonal decomposition approach.
    S near 1: Strong seasonality
    S near 0: Weak seasonality
    """
    n = len(y)
    if n < 2 * period:
        return 0.0
    
    # Simple seasonal decomposition
    # Compute seasonal means
    n_complete = (n // period) * period
    y_trimmed = y[:n_complete].reshape(-1, period)
    seasonal_means = np.mean(y_trimmed, axis=0)
    
    # Extend seasonal component
    n_repeats = n // period + 1
    seasonal = np.tile(seasonal_means, n_repeats)[:n]
    
    # Deseasonalized
    deseasonalized = y - seasonal
    
    # Trend via moving average
    window = min(period, n // 2)
    if window < 1:
        window = 1
    trend = np.convolve(y, np.ones(window)/window, mode='same')
    
    # Remainder
    remainder = y - trend - seasonal
    
    # Seasonal strength
    var_remainder = np.var(remainder)
    var_detrended = np.var(y - trend)
    
    if var_detrended == 0:
        return 0.0
    
    strength = max(0, 1 - var_remainder / var_detrended)
    return strength


def spectral_entropy(y: np.ndarray) -> float:
    """Compute spectral entropy of normalized periodogram.
    
    Φ near 0: Power concentrated at few frequencies (predictable)
    Φ near 1: Power spread uniformly (unpredictable/white noise)
    """
    n = len(y)
    if n < 4:
        return 0.5
    
    # Compute periodogram
    y_centered = y - np.mean(y)
    fft_vals = fft(y_centered)
    power = np.abs(fft_vals[:n//2])**2
    
    # Normalize to get probability distribution
    total_power = np.sum(power)
    if total_power == 0:
        return 0.5
    
    p = power / total_power
    
    # Remove zeros for log computation
    p = p[p > 0]
    
    # Compute entropy
    entropy = -np.sum(p * np.log(p))
    
    # Normalize by maximum entropy (uniform distribution)
    max_entropy = np.log(len(p))
    if max_entropy == 0:
        return 0.5
    
    return entropy / max_entropy


def extract_all_features(
    y: np.ndarray,
    seasonal_period: int = 24
) -> Dict[str, float]:
    """Extract all dataset features.
    
    Args:
        y: Time series values
        seasonal_period: Expected seasonal period
    
    Returns:
        Dictionary with feature names and values
    """
    features = {
        'tau_acf': acf_decay_lag(y, threshold=0.5),
        'hurst_H': hurst_exponent(y),
        'noise_ratio': noise_ratio(y),
        'seasonal_strength': seasonal_strength(y, seasonal_period),
        'spectral_entropy': spectral_entropy(y),
    }
    
    return features


def predict_breakeven(features: Dict[str, float]) -> float:
    """Predict break-even point from features.
    
    Uses regression coefficients from paper:
    log(n*) = 3.21 + 0.094*tau_acf + 2.87*H + 1.42*sigma_r - 1.18*S + 1.56*Phi
    
    Note: These are placeholder coefficients - will be fitted from actual experiments.
    """
    log_n_star = (
        3.21 
        + 0.094 * features['tau_acf']
        + 2.87 * features['hurst_H']
        + 1.42 * features['noise_ratio']
        - 1.18 * features['seasonal_strength']
        + 1.56 * features['spectral_entropy']
    )
    
    return np.exp(log_n_star)


def recommend_model(
    y: np.ndarray,
    n_available: int,
    seasonal_period: int = 24
) -> Tuple[str, Dict[str, float]]:
    """Recommend model class based on data characteristics.
    
    Args:
        y: Time series for feature extraction
        n_available: Available training samples
        seasonal_period: Expected seasonal period
    
    Returns:
        Tuple of (recommendation, features)
    """
    features = extract_all_features(y, seasonal_period)
    n_star = predict_breakeven(features)
    
    if n_available > 1.5 * n_star:
        recommendation = "Foundation model (fine-tuned) - sufficient data for FM advantage"
    elif n_available < 0.5 * n_star:
        recommendation = "Classical method (XGBoost) - insufficient data for FM"
    else:
        recommendation = "Uncertain - evaluate both model types"
    
    return recommendation, features


if __name__ == "__main__":
    # Test feature extraction
    np.random.seed(42)
    
    print("=" * 60)
    print("Testing Feature Extraction")
    print("=" * 60)
    
    # Generate different types of series
    n = 1000
    t = np.arange(n)
    
    test_cases = {
        "Seasonal + Trend": 10 + 0.01*t + 5*np.sin(2*np.pi*t/24) + np.random.normal(0, 1, n),
        "Random Walk": np.cumsum(np.random.normal(0, 1, n)),
        "White Noise": np.random.normal(0, 1, n),
        "Strong AR(1)": None,  # Will generate below
    }
    
    # Generate AR(1)
    ar1 = np.zeros(n)
    ar1[0] = np.random.normal(0, 1)
    for i in range(1, n):
        ar1[i] = 0.95 * ar1[i-1] + np.random.normal(0, 1)
    test_cases["Strong AR(1)"] = ar1
    
    for name, y in test_cases.items():
        print(f"\n{name}:")
        features = extract_all_features(y, seasonal_period=24)
        for feat_name, value in features.items():
            print(f"  {feat_name}: {value:.3f}")
        
        n_star = predict_breakeven(features)
        print(f"  Predicted n*: {n_star:.0f}")
        
        rec, _ = recommend_model(y, n_available=500, seasonal_period=24)
        print(f"  Recommendation (n=500): {rec}")
