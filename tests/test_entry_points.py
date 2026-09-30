import importlib
import time
import warnings

import pytest

import evaluation_test as et
import final_selection


def test_evaluation_is_import_safe_and_deprecated():
    t0 = time.time()
    with pytest.warns(DeprecationWarning):
        import evaluation
        importlib.reload(evaluation)
    assert time.time() - t0 < 10     # no study at import time


def test_candidates_to_frame():
    m = {'sharpe': 1.0, 'sortino': 2.0, 'omega': 1.2, 'calmar': 0.5, 'alpha': 3.0,
         'max_dd': 0.3, 'profit_factor': 2.5, 'pct_profitable': 0.4, 'num_trades': 50}
    df = final_selection.candidates_to_frame([
        {'front': 1, 'candidate_idx': 1, 'overall_cov': 0.05, 'cov_class': '1st Class',
         'base_metrics': m, 'params': {'aroon_length': 14}}])
    assert list(df[['sharpe', 'sortino', 'omega', 'calmar', 'alpha']].iloc[0]) == [1.0, 2.0, 1.2, 0.5, 3.0]


def test_holdout_report_window(data_dir, price):
    p = {'supertrend_atr_period': 10, 'supertrend_factor': 3.0}
    r = et.holdout_report(10000, 'crypto', p, '2019-12-31', 0.0, 'long_only', 'alt',
                          ['supertrend'], ticker='SYN-USD', data_dir=data_dir, verbose=False)
    assert str(r['holdout_start'].date()) == '2020-01-01'
    assert r['holdout_end'] == price.index[-1]
    assert r['metrics'] is not None and set(r['colors']) == set(et.TABLE_METRICS)
