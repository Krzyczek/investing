import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def make_price(n=1600, start='2016-01-01', seed=0, drift=None):
    """Synthetic daily OHLC with alternating up/down regimes (deterministic)."""
    rng = np.random.default_rng(seed)
    if drift is None:
        regime = np.repeat(rng.choice([-1.0, 1.0], size=n // 80 + 1), 80)[:n]
        drift = 0.004 * regime
    rets = drift + rng.normal(0, 0.02, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.01, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range(start, periods=n, freq='D', name='time')
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close}, index=idx)
    df['return'] = df['close'].pct_change(fill_method=None)
    return df


@pytest.fixture
def price():
    return make_price()


@pytest.fixture
def data_dir(tmp_path, price):
    """Folder with a synthetic 'SYN-USD.csv' (+ a decoy file)."""
    price.drop(columns='return').to_csv(tmp_path / 'SYN-USD.csv')
    make_price(seed=5).drop(columns='return').to_csv(tmp_path / 'ZZZ-USD.csv')
    return str(tmp_path)
