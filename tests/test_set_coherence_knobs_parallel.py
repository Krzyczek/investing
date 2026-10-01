"""1 Oct 2026: KNOB covers every POOL-READY member (no ad-hoc knobs), a member without a
knob fails loudly, and parallel scoring (n_jobs) gives exactly the serial result."""
import numpy as np
import pandas as pd
import pytest

import set_coherence as sc
import tpi

NEW = {'obv': 'obv_ema_length', 'linreg': 'linreg_length', 'ehlers_itrend': 'itrend_alpha',
       'ichimoku': 'ichimoku_base', 'regime_gate': ('regime_fast', 'regime_slow')}
BASE = ('ema_cross', 'parabolic_sar', 'supertrend', 'adx', 'aroon')


def test_knob_has_new_pool_members():
    for c, k in NEW.items():
        assert sc.KNOB[c] == k
    # every POOL-READY key has a knob (base 5 + 16 branch members = 21)
    pool_ready = set(BASE) | {'donchian', 'keltner', 'rsi50', 'roc', 'hull', 'tema', 'vortex', 'cci',
                              'kama', 'macd', 'bollinger'} | set(NEW)
    assert len(pool_ready) == 21 and pool_ready <= set(sc.KNOB)


def test_registered_knobs_name_registered_params():
    for c in tpi.COMPONENTS:
        assert c in sc.KNOB, f'{c} registered but has no KNOB entry'
        keys = sc.KNOB[c] if isinstance(sc.KNOB[c], tuple) else (sc.KNOB[c],)
        assert set(keys) <= set(tpi.COMPONENTS[c]['params'])


def test_member_without_knob_raises(price, monkeypatch):
    monkeypatch.setitem(tpi.COMPONENTS, 'noknob', {**tpi.COMPONENTS['ema_cross']})
    with pytest.raises(ValueError, match='no speed knob'):
        sc.horizon_match(price, ['ema_cross', 'noknob'], 20, str(price.index[0].date()))


def test_regime_gate_knob_is_matched_on_2d_ema_pair(price, monkeypatch):
    # stand-in with regime_gate's param names (the real one lives on ind-regime-gate):
    # EMA-cross vote on (regime_fast, regime_slow), gate params unused here
    def sig(df, p):
        return tpi.COMPONENTS['ema_cross']['signal'](df, {'fast_ma': p['regime_fast'],
                                                          'slow_ma': p['regime_slow']})
    monkeypatch.setitem(tpi.COMPONENTS, 'regime_gate', {
        'group': 'perpetual', 'signal': sig,
        'params': {'regime_fast': ('int', 5, 30, 1), 'regime_slow': ('int', 5, 40, 1),
                   'regime_hurst_window': ('int', 50, 200, 25),
                   'regime_hurst_threshold': ('float', 0.40, 0.70, 0.01),
                   'regime_adf_pvalue': ('float', 0.0, 0.50, 0.01)}})
    monkeypatch.setattr(tpi, 'CONSTRAINTS', tpi.CONSTRAINTS + [
        ('regime_gate', lambda p: p['regime_fast'] < p['regime_slow'])])
    tab, params = sc.horizon_match(price, ['regime_gate'], 30, str(price.index[0].date()))
    row = tab.iloc[0]
    assert row.param == 'regime_fast/regime_slow' and '/4 neighbours' in str(row.plateau)
    p = params['regime_gate']
    assert p['regime_fast'] < p['regime_slow']
    tb = sc.textbook('regime_gate')          # gate params held at textbook (mid-range here)
    assert all(p[k] == tb[k] for k in ('regime_hurst_window', 'regime_hurst_threshold', 'regime_adf_pvalue'))


def _synthetic_pool(n_members=9, T=600, target=10, holds=(10, 20)):
    rng = np.random.default_rng(7)
    t = np.arange(T)
    base = np.where((t // 60) % 2 == 0, 1.0, -1.0)
    rows = []
    for i in range(n_members):
        r = np.roll(base, int(rng.integers(0, 25)))
        flip = rng.random(T) < 0.05 * (i % 3)
        rows.append(np.where(flip, -r, r) * (rng.random(T) > 0.03))
    return sc.Pool([f'm{i}' for i in range(n_members)], np.vstack(rows), target, holds=holds)


@pytest.mark.parametrize('n_jobs,chunk', [(2, 7), (3, 50), (None, 13)])
def test_parallel_score_all_identical_to_serial(n_jobs, chunk):
    P = _synthetic_pool()
    serial = sc.score_all(P, sizes=(5, 6, 7), holds=(10, 20))
    par = sc._score_pools([P], (5, 6, 7), (10, 20), n_jobs=n_jobs, chunk=chunk)[0]
    pd.testing.assert_frame_equal(serial, par)
    assert len(serial) == (126 + 84 + 36) * 2


def test_parallel_rank_pool_identical_to_serial(price):
    pool = list(BASE)
    kw = dict(targets=(20, 30), holds=(5, 10), sizes=(4, 5), start=str(price.index[0].date()))
    a = sc.rank_pool(price, pool, n_jobs=1, **kw)
    b = sc.rank_pool(price, pool, n_jobs=2, **kw)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
