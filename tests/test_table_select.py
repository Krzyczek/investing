"""Automatic colour-table choice by history length (select_table / resolve_table)."""
import numpy as np
import pandas as pd

import evaluation_test as et
import optuna_testing
from conftest import make_price


def _span(first, last):
    n = (pd.Timestamp(last) - pd.Timestamp(first)).days + 1
    return make_price(n=n, start=first)


def test_main_when_history_starts_before_2018():
    c = et.select_table(_span('2015-06-01', '2025-01-01'))
    assert c['table'] == 'main' and not c['skip'] and c['start'] == '2018-01-01'


def test_boundary_first_bar_exactly_2018_01_01_is_main():
    c = et.select_table(_span('2018-01-01', '2025-01-01'))
    assert c['table'] == 'main' and c['start'] == '2018-01-01'
    assert et.select_table(_span('2018-01-02', '2025-01-01'))['table'] == 'alt'


def test_alt_after_2018_with_3_years_and_warmup_start():
    p = _span('2020-04-10', '2024-01-01')
    c = et.select_table(p)
    assert c['table'] == 'alt' and not c['skip']
    assert c['start'] == p.index[et.ALT_WARMUP_BARS]      # 200 warm-up bars
    assert c['start'] > pd.Timestamp('2020-04-10')


def test_three_years_counted_first_valid_to_last_bar():
    assert et.select_table(_span('2021-01-01', '2024-01-01'))['table'] == 'alt'   # exactly 3y
    c = et.select_table(_span('2021-01-01', '2023-12-31'))                       # one day short
    assert c['skip'] and c['table'] is None and 'skipped' in c['reason']


def test_first_valid_bar_ignores_leading_nan_rows():
    p = _span('2017-06-01', '2025-01-01')
    p.loc[:'2018-03-01', ['open', 'high', 'low', 'close']] = np.nan
    c = et.select_table(p)
    assert c['table'] == 'alt' and c['first_bar'] == pd.Timestamp('2018-03-02')


def test_under_3_years_skips_without_raising(tmp_path, capsys):
    _span('2023-01-01', '2024-06-30').drop(columns='return').to_csv(tmp_path / 'NEW-USD.csv')
    s = optuna_testing.instrument_strategy('NEW-USD', 'crypto', 10000, 0.0,
                                           data_dir=str(tmp_path))
    res = s.strategy_evaluation(n_trials=2, n_jobs=1, seed=1, components=['supertrend'])
    assert isinstance(res, et.AssetSkipped) and list(res) == [] and s.study is None
    r = et.parameter_robustness_test(10000, 'crypto', [], 0.0, ticker='NEW-USD',
                                     data_dir=str(tmp_path))
    assert isinstance(r, et.AssetSkipped)
    assert '[colour table] NEW-USD: SKIP' in capsys.readouterr().out


def test_explicit_table_overrides_rule(capsys):
    p = _span('2023-01-01', '2024-06-30')             # would be SKIP
    c = et.resolve_table(p, 'main', ticker='X')
    assert c['table'] == 'main' and not c['skip'] and c['source'] == 'explicit'
    c = et.resolve_table(_span('2015-01-01', '2025-01-01'), 'alt', ticker='X')
    assert c['table'] == 'alt'
    c = et.resolve_table(_span('2015-01-01', '2025-01-01'), 'auto', ticker='X')
    assert c['table'] == 'main' and c['source'] == 'auto'
    assert '[colour table] X:' in capsys.readouterr().out
