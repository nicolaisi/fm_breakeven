"""
Model wrappers for forecasting experiments.

Classical models (CPU):
- Naive (Seasonal Naive)
- ETS (Exponential Smoothing)
- ARIMA (Auto-ARIMA)
- XGBoost (with lag features)

Foundation models (GPU required):
- Chronos (Amazon) - zero-shot and fine-tuned
- Moirai (Salesforce) - zero-shot and fine-tuned
- Lag-Llama - zero-shot and fine-tuned
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Base Classes
# ============================================================================

class BaseForecaster(ABC):
    """Abstract base class for forecasting models."""
    
    def __init__(self, horizon: int, seasonal_period: int, **kwargs):
        self.horizon = horizon
        self.seasonal_period = seasonal_period
        self.is_fitted = False
        self.model = None
    
    @abstractmethod
    def fit(self, y: np.ndarray) -> 'BaseForecaster':
        """Fit the model on training data."""
        pass
    
    @abstractmethod
    def predict(self) -> np.ndarray:
        """Generate forecasts for the horizon."""
        pass
    
    def predict_from(self, history: np.ndarray) -> np.ndarray:
        """Forecast `horizon` steps after the end of `history`, using the
        parameters estimated in fit(). Override in subclasses."""
        raise NotImplementedError

    def fit_predict(self, y: np.ndarray) -> np.ndarray:
        """Fit and predict in one call."""
        self.fit(y)
        return self.predict()


# ============================================================================
# Classical Models (CPU)
# ============================================================================

class NaiveForecaster(BaseForecaster):
    """Seasonal Naive: repeat last seasonal cycle."""
    
    def __init__(self, horizon: int, seasonal_period: int, **kwargs):
        super().__init__(horizon, seasonal_period)
        self.last_season = None
    
    def fit(self, y: np.ndarray) -> 'NaiveForecaster':
        sp = min(self.seasonal_period, len(y))
        self.last_season = y[-sp:]
        self.is_fitted = True
        return self
    
    def predict(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        forecasts = []
        sp = len(self.last_season)
        for h in range(self.horizon):
            forecasts.append(self.last_season[h % sp])
        
        return np.array(forecasts)


    def predict_from(self, history: np.ndarray) -> np.ndarray:
        sp = min(self.seasonal_period, len(history))
        last_season = history[-sp:]
        return np.array([last_season[h % sp] for h in range(self.horizon)])


class ETSForecaster(BaseForecaster):
    """Exponential Smoothing with automatic model selection.

    By default training data is capped at the last MAX_TRAIN_SAMPLES points
    (max_train_samples=None removes the cap), and season_length at
    MAX_SEASON_LENGTH, to keep fitting tractable on high-frequency datasets
    like Weather (sp=144) and ETTm (sp=96).
    """

    MAX_TRAIN_SAMPLES = 200
    MAX_SEASON_LENGTH = 24
    MAX_CONTEXT = 2000  # history passed to forward() at prediction time

    def __init__(self, horizon: int, seasonal_period: int,
                 max_train_samples: Optional[int] = MAX_TRAIN_SAMPLES, **kwargs):
        super().__init__(horizon, seasonal_period)
        self._sf = None
        # None or 0 = no cap (fit on the whole training fraction)
        self.max_train_samples = max_train_samples or None

    def fit(self, y: np.ndarray) -> 'ETSForecaster':
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import AutoETS
            import pandas as pd
        except ImportError:
            raise ImportError("statsforecast required: pip install statsforecast")

        # Cap training data (optional) and season_length for tractability
        if self.max_train_samples and len(y) > self.max_train_samples:
            y = y[-self.max_train_samples:]
        effective_sp = min(self.seasonal_period, self.MAX_SEASON_LENGTH)

        df = pd.DataFrame({
            'unique_id': ['series'] * len(y),
            'ds': pd.date_range('2020-01-01', periods=len(y), freq='h'),
            'y': y
        })

        self._sf = StatsForecast(
            models=[AutoETS(season_length=effective_sp)],
            freq='h',
            n_jobs=1
        )
        self._sf.fit(df)
        self.is_fitted = True
        return self
    
    def predict(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        forecasts = self._sf.predict(h=self.horizon)
        return forecasts['AutoETS'].values


    def predict_from(self, history: np.ndarray) -> np.ndarray:
        """Apply the fitted model (no re-estimation) to recent history."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        tail = history[-self.MAX_CONTEXT:] if len(history) > self.MAX_CONTEXT else history
        model = self._sf.fitted_[0, 0]
        out = model.forward(y=np.asarray(tail, dtype=float), h=self.horizon)
        return np.asarray(out['mean'])


class ARIMAForecaster(BaseForecaster):
    """Auto-ARIMA with automatic order selection.
    
    By default training data is capped at the last MAX_TRAIN_SAMPLES points
    (max_train_samples=None removes the cap), and season_length at 24, to keep
    fitting tractable on high-frequency datasets like ETTm (sp=96).
    """

    MAX_TRAIN_SAMPLES = 200
    MAX_CONTEXT = 2000  # history passed to forward() at prediction time
    
    def __init__(self, horizon: int, seasonal_period: int,
                 max_train_samples: Optional[int] = MAX_TRAIN_SAMPLES, **kwargs):
        super().__init__(horizon, seasonal_period)
        self._sf = None
        # None or 0 = no cap (fit on the whole training fraction)
        self.max_train_samples = max_train_samples or None
    
    def fit(self, y: np.ndarray) -> 'ARIMAForecaster':
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA
            import pandas as pd
        except ImportError:
            raise ImportError("statsforecast required: pip install statsforecast")
        
        # Cap training data (optional) to keep fitting tractable
        if self.max_train_samples and len(y) > self.max_train_samples:
            y = y[-self.max_train_samples:]
        
        df = pd.DataFrame({
            'unique_id': ['series'] * len(y),
            'ds': pd.date_range('2020-01-01', periods=len(y), freq='h'),
            'y': y
        })
        
        effective_sp = min(self.seasonal_period, 24)

        self._sf = StatsForecast(
            models=[AutoARIMA(season_length=effective_sp)],
            freq='h',
            n_jobs=1
        )
        self._sf.fit(df)
        self.is_fitted = True
        return self
    
    def predict(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        forecasts = self._sf.predict(h=self.horizon)
        return forecasts['AutoARIMA'].values


    def predict_from(self, history: np.ndarray) -> np.ndarray:
        """Apply the fitted model (no re-estimation) to recent history."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        tail = history[-self.MAX_CONTEXT:] if len(history) > self.MAX_CONTEXT else history
        model = self._sf.fitted_[0, 0]
        out = model.forward(y=np.asarray(tail, dtype=float), h=self.horizon)
        return np.asarray(out['mean'])


class XGBoostForecaster(BaseForecaster):
    """XGBoost on lag, rolling-window and calendar features; recursive forecasting.

    feature_set="paper" (default), as described in the paper, with m = seasonal period:
      - lags 1..min(2m, 48)
      - rolling mean and std over the last w values, w in {m/2, m, 2m}
      - calendar indicators: position in the seasonal cycle (t mod m), plus
        day of week for sub-daily data ((t // m) mod 7, m = steps per day) and
        for daily data whose period is not already a week.
        Positions are taken from the absolute time index t of each point. Most
        datasets (Monash, M4) carry synthetic dates, so the index is used
        instead of timestamps; for a tree model the two are equivalent up to
        a relabelling of the categories.
      On short training sets, lags and windows are capped at n_train // 2 so
      at least half of the samples yield training rows.

    feature_set="legacy" reproduces the original runs: n_lags lags (default 24;
    the classical runner used 192 for 10/15-min data) plus mean/std of that window.
    """

    MAX_LAGS = 48

    def __init__(
        self,
        horizon: int,
        seasonal_period: int,
        n_lags: Optional[int] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        feature_set: str = "paper",
        freq: str = "",
        **kwargs
    ):
        super().__init__(horizon, seasonal_period)
        if feature_set not in ("paper", "legacy"):
            raise ValueError(f"feature_set must be 'paper' or 'legacy', got {feature_set!r}")
        self.feature_set = feature_set
        self.freq = str(freq)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        m = max(1, int(seasonal_period))
        if feature_set == "legacy":
            self.n_lags = n_lags or 24
            self.windows = [self.n_lags]
            self.calendar = []
        else:
            self.n_lags = n_lags or min(2 * m, self.MAX_LAGS)
            self.windows = sorted({max(2, w) for w in (m // 2, m, 2 * m)})
            self.calendar = self._calendar_cycles(m, self.freq)
        self._train_tail = None   # last `lookback` training values
        self._train_end = None    # absolute index of the point after them

    @staticmethod
    def _calendar_cycles(m: int, freq: str):
        """(divisor, modulus) pairs; feature = (t // divisor) % modulus."""
        f = freq.lower().lstrip("0123456789")
        cycles = [(1, m)] if m > 1 else []
        if f in ("h", "min", "t", "s") and m > 1:        # sub-daily: m = steps per day
            cycles.append((m, 7))
        elif f == "d" and m not in (5, 7):                 # daily, period not a week
            cycles.append((1, 7))
        return cycles

    @property
    def lookback(self) -> int:
        return max([self.n_lags] + self.windows)

    def _features_at(self, hist: np.ndarray, t: int) -> np.ndarray:
        """Features for predicting the value at absolute index t from the
        values `hist` that precede it (hist[-1] is at index t-1)."""
        feats = [hist[-1:-self.n_lags - 1:-1]]
        for w in self.windows:
            seg = hist[-w:]
            feats.append([seg.mean(), seg.std()])
        if self.calendar:
            feats.append([(t // d) % k for d, k in self.calendar])
        return np.concatenate([np.asarray(f, dtype=float) for f in feats])

    def _create_features(self, y: np.ndarray, t0: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Training matrix (vectorised). y[0] sits at absolute index t0."""
        from numpy.lib.stride_tricks import sliding_window_view
        y = np.asarray(y, dtype=float)
        n, L = len(y), self.lookback
        idx = np.arange(L, n)                               # positions in y being predicted
        lag_win = sliding_window_view(y, self.n_lags)       # lag_win[j] = y[j:j+n_lags]
        cols = [lag_win[idx - self.n_lags][:, ::-1]]        # lag 1 first
        c1 = np.concatenate([[0.0], np.cumsum(y)])
        c2 = np.concatenate([[0.0], np.cumsum(y * y)])
        for w in self.windows:
            s1 = c1[idx] - c1[idx - w]
            s2 = c2[idx] - c2[idx - w]
            mean = s1 / w
            var = np.maximum(s2 / w - mean ** 2, 0.0)
            cols.append(np.column_stack([mean, np.sqrt(var)]))
        if self.calendar:
            t = idx + t0
            cols.append(np.column_stack([(t // d) % k for d, k in self.calendar]))
        return np.column_stack(cols), y[idx]

    def fit(self, y: np.ndarray, t0: int = 0) -> 'XGBoostForecaster':
        """Fit on y; t0 is the absolute time index of y[0] (for calendar features)."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost required: pip install xgboost")

        y = np.asarray(y, dtype=float)
        if self.feature_set == "paper":
            cap = max(2, len(y) // 2)
            self.n_lags = min(self.n_lags, cap)
            self.windows = sorted({min(w, cap) for w in self.windows})
        if len(y) <= self.lookback:
            raise ValueError(f"XGBoost needs more than {self.lookback} samples, got {len(y)}")

        X, targets = self._create_features(y, t0)
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            verbosity=0
        )
        self.model.fit(X, targets)
        self._train_tail = y[-self.lookback:].copy()
        self._train_end = t0 + len(y)
        self.is_fitted = True
        return self

    def _recursive(self, hist: np.ndarray, t_next: int) -> np.ndarray:
        hist = list(np.asarray(hist, dtype=float)[-self.lookback:])
        out = []
        for k in range(self.horizon):
            x = self._features_at(np.asarray(hist), t_next + k)
            pred = float(self.model.predict(x.reshape(1, -1))[0])
            out.append(pred)
            hist.append(pred)
            hist.pop(0)
        return np.array(out)

    def predict(self) -> np.ndarray:
        """Forecast the horizon right after the training data."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self._recursive(self._train_tail, self._train_end)

    def predict_from(self, history: np.ndarray) -> np.ndarray:
        """Recursive forecast after `history`, which must start at absolute
        index 0 (as the contexts from load_and_prepare do)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        h = np.asarray(history, dtype=float)
        if len(h) < self.lookback:
            h = np.concatenate([np.full(self.lookback - len(h), h[0]), h])
        return self._recursive(h, len(history))


# ============================================================================
# Foundation Models (GPU)
# ============================================================================

class ChronosForecaster(BaseForecaster):
    """
    Chronos foundation model (Amazon).
    
    Requires GPU and: pip install chronos-forecasting torch
    """
    
    def __init__(
        self,
        horizon: int,
        seasonal_period: int,
        model_size: str = "base",  # "small", "base", "large"
        fine_tune: bool = False,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(horizon, seasonal_period)
        self.model_size = model_size
        self.fine_tune = fine_tune
        self.device = device
        self.pipeline = None
        self._y_train = None
        
        # Model paths
        self.model_paths = {
            "small": "amazon/chronos-bolt-small",
            "base": "amazon/chronos-bolt-base",
            "large": "amazon/chronos-bolt-large",
        }
    
    def fit(self, y: np.ndarray) -> 'ChronosForecaster':
        try:
            from chronos import ChronosPipeline
            import torch
        except ImportError:
            raise ImportError(
                "Chronos required: pip install chronos-forecasting torch"
            )
        
        model_path = self.model_paths.get(self.model_size, self.model_size)
        
        # Load model
        self.pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        
        self._y_train = y.copy()
        
        if self.fine_tune:
            # Fine-tuning would go here
            # For Chronos, this typically uses AutoGluon-TimeSeries
            # or custom training loop
            pass
        
        self.is_fitted = True
        return self
    
    def predict(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        import torch
        
        context = torch.tensor(self._y_train, dtype=torch.float32).unsqueeze(0)
        
        # Generate forecasts
        forecast = self.pipeline.predict(
            context,
            self.horizon,
            num_samples=20,
        )
        
        # Return median forecast
        return np.median(forecast[0].numpy(), axis=0)


class MoiraiForecaster(BaseForecaster):
    """
    Moirai foundation model (Salesforce).
    
    Requires GPU and: pip install uni2ts torch
    """
    
    def __init__(
        self,
        horizon: int,
        seasonal_period: int,
        model_size: str = "base",  # "small", "base", "large"
        fine_tune: bool = False,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(horizon, seasonal_period)
        self.model_size = model_size
        self.fine_tune = fine_tune
        self.device = device
        self._y_train = None
        
        self.model_paths = {
            "small": "Salesforce/moirai-1.0-R-small",
            "base": "Salesforce/moirai-1.0-R-base",
            "large": "Salesforce/moirai-1.0-R-large",
        }
    
    def fit(self, y: np.ndarray) -> 'MoiraiForecaster':
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            import torch
        except ImportError:
            raise ImportError(
                "Moirai required: pip install uni2ts torch"
            )
        
        model_path = self.model_paths.get(self.model_size, self.model_size)
        
        # Load pretrained model
        self.model = MoiraiForecast.load_from_checkpoint(
            prediction_length=self.horizon,
            context_length=min(512, len(y)),
            patch_size="auto",
            num_samples=20,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        
        self._y_train = y.copy()
        
        if self.fine_tune:
            # Fine-tuning with LoRA would go here
            pass
        
        self.is_fitted = True
        return self
    
    def predict(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        import torch
        
        # Prepare input
        context = torch.tensor(self._y_train, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        
        # Generate forecast
        forecast = self.model(context)
        
        return np.median(forecast.numpy(), axis=1).flatten()[:self.horizon]


class LagLlamaForecaster(BaseForecaster):
    """
    Lag-Llama foundation model.
    
    Requires GPU. See: https://github.com/time-series-foundation-models/lag-llama
    """
    
    def __init__(
        self,
        horizon: int,
        seasonal_period: int,
        fine_tune: bool = False,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(horizon, seasonal_period)
        self.fine_tune = fine_tune
        self.device = device
        self._y_train = None
    
    def fit(self, y: np.ndarray) -> 'LagLlamaForecaster':
        try:
            from lag_llama.gluon.estimator import LagLlamaEstimator
            import torch
        except ImportError:
            raise ImportError(
                "Lag-Llama required. See: "
                "https://github.com/time-series-foundation-models/lag-llama"
            )
        
        self.estimator = LagLlamaEstimator(
            prediction_length=self.horizon,
            context_length=min(512, len(y)),
            num_samples=20,
            device=self.device,
        )
        
        self._y_train = y.copy()
        
        if self.fine_tune:
            # Fine-tuning would go here
            pass
        
        self.is_fitted = True
        return self
    
    def predict(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Generate forecast
        # Note: Actual implementation depends on Lag-Llama API
        # This is a placeholder structure
        
        import torch
        context = torch.tensor(self._y_train, dtype=torch.float32)
        
        # Placeholder - actual prediction code depends on Lag-Llama version
        forecast = np.zeros(self.horizon)
        
        return forecast


# ============================================================================
# Model Registry and Factory
# ============================================================================

CLASSICAL_MODELS = {
    'Naive': NaiveForecaster,
    'ETS': ETSForecaster,
    'ARIMA': ARIMAForecaster,
    'XGBoost': XGBoostForecaster,
}

FOUNDATION_MODELS = {
    'Chronos-S': lambda **kw: ChronosForecaster(model_size='small', **kw),
    'Chronos-B': lambda **kw: ChronosForecaster(model_size='base', **kw),
    'Chronos-L': lambda **kw: ChronosForecaster(model_size='large', **kw),
    'Chronos-S-FT': lambda **kw: ChronosForecaster(model_size='small', fine_tune=True, **kw),
    'Chronos-B-FT': lambda **kw: ChronosForecaster(model_size='base', fine_tune=True, **kw),
    'Chronos-L-FT': lambda **kw: ChronosForecaster(model_size='large', fine_tune=True, **kw),
    'Moirai-S': lambda **kw: MoiraiForecaster(model_size='small', **kw),
    'Moirai-B': lambda **kw: MoiraiForecaster(model_size='base', **kw),
    'Moirai-L': lambda **kw: MoiraiForecaster(model_size='large', **kw),
    'Moirai-S-FT': lambda **kw: MoiraiForecaster(model_size='small', fine_tune=True, **kw),
    'Moirai-B-FT': lambda **kw: MoiraiForecaster(model_size='base', fine_tune=True, **kw),
    'Moirai-L-FT': lambda **kw: MoiraiForecaster(model_size='large', fine_tune=True, **kw),
    'LagLlama': LagLlamaForecaster,
    'LagLlama-FT': lambda **kw: LagLlamaForecaster(fine_tune=True, **kw),
}

ALL_MODELS = {**CLASSICAL_MODELS, **FOUNDATION_MODELS}


def get_model(name: str, horizon: int, seasonal_period: int, **kwargs) -> BaseForecaster:
    """
    Get model instance by name.
    
    Args:
        name: Model name (e.g., 'XGBoost', 'Chronos-B', 'Moirai-L-FT')
        horizon: Forecast horizon
        seasonal_period: Seasonal period
        **kwargs: Additional model parameters
    
    Returns:
        Forecaster instance
    """
    if name not in ALL_MODELS:
        raise ValueError(
            f"Unknown model: {name}. "
            f"Available: {list(ALL_MODELS.keys())}"
        )
    
    model_cls = ALL_MODELS[name]
    
    if callable(model_cls) and not isinstance(model_cls, type):
        # It's a lambda factory
        return model_cls(horizon=horizon, seasonal_period=seasonal_period, **kwargs)
    else:
        # It's a class
        return model_cls(horizon=horizon, seasonal_period=seasonal_period, **kwargs)


def list_models(include_foundation: bool = True) -> Dict[str, list]:
    """List available models."""
    result = {'classical': list(CLASSICAL_MODELS.keys())}
    if include_foundation:
        result['foundation'] = list(FOUNDATION_MODELS.keys())
    return result


if __name__ == "__main__":
    # Test classical models
    print("Testing Classical Models")
    print("=" * 50)
    
    np.random.seed(42)
    t = np.arange(500)
    y = 10 + 0.01*t + 5*np.sin(2*np.pi*t/24) + np.random.normal(0, 1, 500)
    
    y_train = y[:-24]
    y_test = y[-24:]
    
    for name in CLASSICAL_MODELS.keys():
        print(f"\n{name}:")
        try:
            model = get_model(name, horizon=24, seasonal_period=24)
            model.fit(y_train)
            preds = model.predict()
            mae = np.mean(np.abs(preds - y_test))
            print(f"  MAE: {mae:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 50)
    print("Foundation models require GPU. Skipping tests.")
    print("Available foundation models:", list(FOUNDATION_MODELS.keys()))
