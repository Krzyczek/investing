import numpy as np
import pandas as pd

import indicators
import tpi
from conftest import make_price


def _signal(df, period=14, threshold=20):
    return tpi.COMPONENTS['adx']['signal'](df, {'adx_period': period, 'threshold': threshold})


def test_adx_exposes_directional_indices(price):
    a = indicators.adx(price, 14)
    a.calculate()
    assert len(a.plus_di) == len(a.minus_di) == len(price)


def test_strong_uptrend_votes_long():
    df = make_price(n=300, drift=np.full(300, 0.01), seed=1)
    sig = _signal(df)
    assert (sig[100:] == 1).mean() > 0.9


def test_strong_downtrend_votes_short():
    # before the fix ADX > threshold voted +1 here as well
    df = make_price(n=300, drift=np.full(300, -0.01), seed=1)
    sig = _signal(df)
    assert (sig[100:] == -1).mean() > 0.9
    assert (sig[100:] == 1).sum() == 0 or (sig[100:] == 1).mean() < 0.05


def test_rule_matches_definition_and_warmup_is_zero(price):
    a = indicators.adx(price, 14)
    a.calculate()
    sig = _signal(price, 14, 20)
    adx, pdi, mdi = a.adx_values.to_numpy(), a.plus_di.to_numpy(), a.minus_di.to_numpy()
    expected = np.where((adx > 20) & (pdi > mdi), 1, np.where((adx > 20) & (mdi > pdi), -1, 0))
    assert np.array_equal(sig, expected)
    warm = np.isnan(adx)
    assert warm[:13].all() and (sig[warm] == 0).all()   # ADX defined from bar period-1
    # weak trend (ADX below threshold) -> 0, never -1 just for being weak
    weak = ~warm & (adx <= 20)
    assert (sig[weak] == 0).all()
