import numpy as np
import pandas as pd
import pytest

import coherence
import evaluation_test as et
import tpi


def _perturb_future(price, cut):
    """Scramble every bar after position `cut`."""
    fut = price.copy()
    rng = np.random.default_rng(42)
    f = rng.uniform(0.5, 1.5, size=len(fut) - cut - 1)
    for c in ('open', 'high', 'low', 'close'):
        fut.iloc[cut + 1:, fut.columns.get_loc(c)] *= f
    fut['return'] = fut['close'].pct_change(fill_method=None)
    return fut


@pytest.mark.parametrize('component', list(tpi.COMPONENTS))
def test_component_signal_ignores_future_bars(price, component):
    cut = 1000
    fut = _perturb_future(price, cut)
    a = np.asarray(tpi.COMPONENTS[component]['signal'](price, coherence.TEXTBOOK_PARAMS), float)
    b = np.asarray(tpi.COMPONENTS[component]['signal'](fut, coherence.TEXTBOOK_PARAMS), float)
    assert np.array_equal(a[:cut + 1], b[:cut + 1])
    assert not np.array_equal(a, b)   # the perturbation did change the future


@pytest.mark.parametrize('mode', ['long_only', 'long_short'])
def test_tpi_and_in_sample_backtest_ignore_future_bars(price, mode):
    cut = 1000
    end = price.index[cut].strftime('%Y-%m-%d')
    fut = _perturb_future(price, cut)
    t1, t2 = tpi.tpi(price), tpi.tpi(fut)
    t1.calculate_tpi(coherence.TEXTBOOK_PARAMS, mode)
    t2.calculate_tpi(coherence.TEXTBOOK_PARAMS, mode)
    assert np.array_equal(t1.signal[:cut + 1], t2.signal[:cut + 1])
    m1 = et.run_backtest(price, 10000, 'crypto', 0.0, mode, 0.0, coherence.TEXTBOOK_PARAMS,
                         start='2018-01-01', end=end)
    m2 = et.run_backtest(fut, 10000, 'crypto', 0.0, mode, 0.0, coherence.TEXTBOOK_PARAMS,
                         start='2018-01-01', end=end)
    assert m1 == m2
