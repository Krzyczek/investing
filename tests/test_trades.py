import numpy as np
import optuna
import pandas as pd
import pytest

import evaluation_test as et
import optuna_testing as ot
import tpi


@pytest.mark.parametrize('pos,expected', [
    ([0, 0, 0], 0),
    ([0, 1, 1, 0, 0, 1], 2),          # long, flat, long -> 2 trades
    ([1, 1, -1, -1, 1], 3),           # long -> short -> long (no flat) -> 3
    ([0, 1, 0, -1, -1, 0], 2),
    ([1, 1, 1, 1], 1),
])
def test_count_trades_is_non_flat_segments(pos, expected):
    assert et.count_trades(pos) == expected


def test_run_backtest_trade_count_matches_segments(price):
    params = dict(fast_ma=10, slow_ma=30)
    for mode in ('long_only', 'long_short'):
        m = et.run_backtest(price, 10000, 'crypto', 0.0, mode, 0.0, params,
                            components=['ema_cross'], start='2018-01-01')
        t = tpi.tpi(price.copy())
        t.calculate_tpi(params, mode, ['ema_cross'])
        pos = pd.Series(t.signal, index=price.index).loc['2018-01-01':].shift(1).fillna(0)
        segments = [g for _, g in pos.groupby((pos != pos.shift()).cumsum()) if g.iloc[0] != 0]
        assert m['num_trades'] == len(segments) == et.count_trades(pos)


def _run_study_with_fake_backtest(monkeypatch, data_dir, table, num_trades, close_dd=0.3):
    def fake_backtest(*args, **kwargs):
        return {'max_dd': 0.3, 'sortino': 1.0, 'sharpe': 1.0, 'profit_factor': 2.0,
                'pct_profitable': 0.4, 'num_trades': num_trades, 'omega': 1.2,
                'calmar': 0.5, 'alpha': 0.0, 'close_max_dd': close_dd}
    monkeypatch.setattr(et, 'run_backtest', fake_backtest)
    s = ot.instrument_strategy('SYN-USD', 'crypto', 10000, 0.0, data_dir=data_dir)
    study = s.strategy_evaluation(table=table, components=['supertrend'], n_trials=3,
                                  n_jobs=1, seed=0)
    return [ot.is_feasible(t) for t in study.trials]


@pytest.mark.parametrize('table,n,feasible', [
    ('main', 103, True),    # yellow/green trade counts are no longer thrown out
    ('main', 105, True),
    ('main', 40, True),     # yellow
    ('main', 39, False),    # red
    ('main', 106, False),   # red
    ('alt', 95, True),
    ('alt', 30, True),
    ('alt', 96, False),
    ('alt', 29, False),
])
def test_optuna_filter_rejects_only_red_band(monkeypatch, data_dir, table, n, feasible):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    assert _run_study_with_fake_backtest(monkeypatch, data_dir, table, n) == [feasible] * 3


def test_optuna_drawdown_rule_still_applies(monkeypatch, data_dir):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    assert _run_study_with_fake_backtest(monkeypatch, data_dir, 'main', 60, close_dd=0.61) == [False] * 3
