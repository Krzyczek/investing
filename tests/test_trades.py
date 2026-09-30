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


# every table metric yellow (no red) -> feasible unless something is overridden
YELLOW = {'max_dd': 0.3, 'sortino': 2.0, 'sharpe': 1.0, 'profit_factor': 2.0,
          'pct_profitable': 0.4, 'num_trades': 60, 'omega': 1.2,
          'calmar': 0.5, 'alpha': 0.0, 'close_max_dd': 0.3}


def _run_fake_study(monkeypatch, data_dir, table='main', n_trials=3, **override):
    def fake_backtest(*args, **kwargs):
        return {**YELLOW, **override}
    monkeypatch.setattr(et, 'run_backtest', fake_backtest)
    s = ot.instrument_strategy('SYN-USD', 'crypto', 10000, 0.0, data_dir=data_dir)
    return s.strategy_evaluation(table=table, components=['supertrend'], n_trials=n_trials,
                                 n_jobs=1, seed=0)


def _run_study_with_fake_backtest(monkeypatch, data_dir, table, num_trades, close_dd=0.3):
    study = _run_fake_study(monkeypatch, data_dir, table, num_trades=num_trades,
                            close_max_dd=close_dd)
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


def test_all_yellow_is_feasible(monkeypatch, data_dir):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _run_fake_study(monkeypatch, data_dir)
    assert all(ot.is_feasible(t) for t in study.trials)
    assert all(t.values == [2.0, 0.5] for t in study.trials)


@pytest.mark.parametrize('metric,value', [
    ('max_dd', 0.41), ('sortino', 1.99), ('sharpe', 0.99), ('profit_factor', 1.99),
    ('pct_profitable', 0.34), ('omega', 1.09), ('num_trades', 106),
])
@pytest.mark.parametrize('table', ['main', 'alt'])
def test_any_red_metric_makes_trial_infeasible(monkeypatch, data_dir, metric, value, table):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _run_fake_study(monkeypatch, data_dir, table, **{metric: value})
    for t in study.trials:
        assert not ot.is_feasible(t)
        assert t.values == ot.PENALTY_VALUES
        red = {k for k, v in t.constraints.items() if v > 0}
        assert red == {f'red_{metric}'}           # only that metric is violated
        assert t.user_attrs['colors'][metric] == 'red'
    fronts = ot.instrument_strategy.get_all_pareto_fronts(None, study)
    assert fronts == []


def test_red_violation_is_graded(monkeypatch, data_dir):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    near = _run_fake_study(monkeypatch, data_dir, sortino=1.9).trials[0].constraints['red_sortino']
    far = _run_fake_study(monkeypatch, data_dir, sortino=0.5).trials[0].constraints['red_sortino']
    assert near == pytest.approx(0.05) and far == pytest.approx(0.75)


def test_liquidated_trial_violates_everything(monkeypatch, data_dir):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    monkeypatch.setattr(et, 'run_backtest', lambda *a, **k: None)
    s = ot.instrument_strategy('SYN-USD', 'crypto', 10000, 0.0, data_dir=data_dir)
    t = s.strategy_evaluation(components=['supertrend'], n_trials=1, n_jobs=1, seed=0).trials[0]
    assert t.constraints['liquidation'] == 1.0
    assert all(t.constraints[f'red_{m}'] == 1.0 for m in et.TABLE_METRICS)
