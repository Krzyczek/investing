"""Zero-score rule: TPI score exactly 0 = HOLD the previous position (no flip,
no exit); flat until the first non-zero score; one-bar lag unchanged."""
import numpy as np
import pandas as pd
import pytest

import evaluation_test as et
import tpi
from conftest import make_price

pos = tpi.position_from_score


def test_zero_after_long_keeps_long():
    assert list(pos([1, 0, 0], 'long_short')) == [1, 1, 1]


def test_zero_after_short_keeps_short():
    assert list(pos([-0.5, 0, 0.0], 'long_short')) == [-1, -1, -1]


def test_zero_on_first_bars_is_flat_until_first_nonzero():
    assert list(pos([0, 0, -1, 0], 'long_short')) == [0, 0, -1, -1]
    assert list(pos([0, 0, 0.2, 0], 'long_short')) == [0, 0, 1, 1]


def test_run_of_zeros_then_flip():
    assert list(pos([1, 0, 0, 0, -1, 0], 'long_short')) == [1, 1, 1, 1, -1, -1]


def test_long_only_same_hold_rule():
    assert list(pos([1, 0, 0], 'long_only')) == [1, 1, 1]          # 0 after long keeps long
    assert list(pos([-1, 0, 0], 'long_only')) == [0, 0, 0]         # 0 after flat stays flat
    assert list(pos([0, 0, 1], 'long_only')) == [0, 0, 1]          # flat until first non-zero
    assert list(pos([1, 0, -1, 0, 1], 'long_only')) == [1, 1, 0, 0, 1]   # negative exits


def test_float_residue_counts_as_zero_but_small_real_scores_do_not():
    assert list(pos([1, 1e-17, -1e-16], 'long_short')) == [1, 1, 1]
    assert list(pos([1, -1 / 6], 'long_short')) == [1, -1]


def test_nan_score_holds():
    assert list(pos([np.nan, 1, np.nan], 'long_short')) == [0, 1, 1]


def test_group_averaging_zero_holds():
    # trend group +1, oscillator group -1 -> score exactly 0 -> hold
    df = make_price(n=5)
    t = tpi.tpi(df)
    tpi.COMPONENTS['_zt_a'] = {'group': 'g1', 'signal': lambda d, p: np.array([1, 1, 1, -1, 1.0]),
                               'params': {}}
    tpi.COMPONENTS['_zt_b'] = {'group': 'g2', 'signal': lambda d, p: np.array([1, -1, -1, -1, 1.0]),
                               'params': {}}
    try:
        t.calculate_tpi({}, 'long_short', ['_zt_a', '_zt_b'])
    finally:
        del tpi.COMPONENTS['_zt_a'], tpi.COMPONENTS['_zt_b']
    assert list(t.tpi_score) == [1, 0, 0, -1, 1]
    assert list(t.signal) == [1, 1, 1, -1, 1]


@pytest.fixture
def fake_component():
    """Component whose score is a fixed array set per test."""
    box = {}
    tpi.COMPONENTS['_fake'] = {'group': 'x', 'signal': lambda d, p: box['score'], 'params': {}}
    yield box
    del tpi.COMPONENTS['_fake']


@pytest.mark.parametrize('mode', ['long_short', 'long_only'])
def test_backtest_holds_through_zero_run_with_one_bar_lag(fake_component, mode):
    n = 12
    df = make_price(n=n, start='2020-01-01', seed=3)
    score = np.array([0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0], dtype=float)
    fake_component['score'] = score
    m = et.run_backtest(df, 10000, 'crypto', 0.0, mode, 0.0, {}, components=['_fake'],
                        start='2020-01-01')
    held = pos(score, mode)
    if mode == 'long_short':
        assert list(held) == [0, 0, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1]
    else:
        assert list(held) == [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    lagged = pd.Series(held, index=df.index).shift(1).fillna(0)       # lag unchanged
    expected = (np.prod(1 + df['return'].fillna(0) * lagged) - 1) * 100
    assert m['total_return'] == pytest.approx(round(expected, 3), abs=1e-3)
    assert m['num_trades'] == et.count_trades(lagged)
    assert m['num_trades'] == (3 if mode == 'long_short' else 2)
