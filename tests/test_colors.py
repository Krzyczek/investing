import numpy as np
import pytest

import evaluation_test as et


@pytest.mark.parametrize('table', ['main', 'alt'])
@pytest.mark.parametrize('name,value,expected', [
    # Intra-trade max DD: red >40%, yellow 25-40%, green <25%
    ('max_dd', 0.2499, 'green'), ('max_dd', 0.25, 'yellow'), ('max_dd', 0.40, 'yellow'),
    ('max_dd', 0.4001, 'red'),
    # Sortino: red <2, yellow 2-2.9, green >2.90
    ('sortino', 1.999, 'red'), ('sortino', 2.0, 'yellow'), ('sortino', 2.9, 'yellow'),
    ('sortino', 2.9001, 'green'),
    # Sharpe: red <1, yellow 1-2, green >2
    ('sharpe', 0.999, 'red'), ('sharpe', 1.0, 'yellow'), ('sharpe', 2.0, 'yellow'),
    ('sharpe', 2.001, 'green'),
    # Profit factor: red <2, yellow 2-4, green >4
    ('profit_factor', 1.99, 'red'), ('profit_factor', 2.0, 'yellow'),
    ('profit_factor', 4.0, 'yellow'), ('profit_factor', 4.01, 'green'),
    ('profit_factor', np.inf, 'green'),
    # % profitable: red <35%, yellow 35-50%, green >50%
    ('pct_profitable', 0.3499, 'red'), ('pct_profitable', 0.35, 'yellow'),
    ('pct_profitable', 0.50, 'yellow'), ('pct_profitable', 0.5001, 'green'),
    # Omega: red <1.1, yellow 1.1-1.31, green >1.31
    ('omega', 1.0999, 'red'), ('omega', 1.1, 'yellow'), ('omega', 1.31, 'yellow'),
    ('omega', 1.3101, 'green'),
    ('sortino', np.nan, 'red'),
])
def test_shared_metrics_identical_in_both_tables(table, name, value, expected):
    assert et.classify_metric(name, value, table) == expected


@pytest.mark.parametrize('n,expected', [
    (39, 'red'), (40, 'yellow'), (44, 'yellow'), (45, 'green'), (105, 'green'), (106, 'red'),
    (0, 'red'),
])
def test_trades_main_table(n, expected):
    assert et.classify_metric('num_trades', n) == expected          # default table
    assert et.classify_metric('num_trades', n, 'main') == expected


@pytest.mark.parametrize('n,expected', [
    (29, 'red'), (30, 'yellow'), (34, 'yellow'), (35, 'green'), (95, 'green'), (96, 'red'),
    (105, 'red'), (40, 'green'),
])
def test_trades_alt_table(n, expected):
    assert et.classify_metric('num_trades', n, 'alt') == expected


def test_unknown_table_rejected():
    with pytest.raises(ValueError):
        et.classify_metric('sharpe', 1.5, 'auto')


def test_column_verdict_uses_table():
    m = {'max_dd': 0.1, 'sortino': 3.5, 'sharpe': 2.5, 'profit_factor': 5,
         'pct_profitable': 0.6, 'num_trades': 100, 'omega': 1.5}
    assert et.column_verdict(m, 'main')[0] is True
    ok, greens, reds, colors = et.column_verdict(m, 'alt')
    assert ok is False and reds == 1 and colors['num_trades'] == 'red'


VALUES = {
    'max_dd': [0.0, 0.2499, 0.25, 0.40, 0.4001, 0.9, np.nan],
    'sortino': [-3.0, 1.999, 2.0, 2.9, 2.9001, np.inf, np.nan],
    'sharpe': [-1.0, 0.999, 1.0, 2.0, 2.001, np.nan],
    'profit_factor': [0.0, 1.99, 2.0, 4.0, 4.01, np.inf, np.nan],
    'pct_profitable': [0.0, 0.3499, 0.35, 0.50, 0.5001],
    'num_trades': [0, 29, 30, 34, 35, 39, 40, 44, 45, 95, 96, 105, 106, 300],
    'omega': [0.5, 1.0999, 1.1, 1.31, 1.3101, np.nan],
}


@pytest.mark.parametrize('table', ['main', 'alt'])
@pytest.mark.parametrize('name', list(VALUES))
def test_red_violations_positive_exactly_when_red(table, name):
    base = {'max_dd': 0.1, 'sortino': 3.5, 'sharpe': 2.5, 'profit_factor': 5,
            'pct_profitable': 0.6, 'num_trades': 60, 'omega': 1.5}
    assert all(v == 0 for v in et.red_violations(base, table).values())
    for value in VALUES[name]:
        v = et.red_violations({**base, name: value}, table)
        is_red = et.classify_metric(name, value, table) == 'red'
        assert (v[name] > 0) == is_red, (name, value, table, v[name])
        assert np.isfinite(v[name])
        assert all(v[k] == 0 for k in v if k != name)


def test_red_violation_grades():
    base = {'max_dd': 0.1, 'sortino': 3.5, 'sharpe': 2.5, 'profit_factor': 5,
            'pct_profitable': 0.6, 'num_trades': 60, 'omega': 1.5}
    g = lambda **kw: et.red_violations({**base, **kw})
    assert g(max_dd=0.5)['max_dd'] == pytest.approx(0.25)
    assert g(sortino=1.5)['sortino'] == pytest.approx(0.25)
    assert g(num_trades=30)['num_trades'] == pytest.approx(0.25)
    assert g(num_trades=126)['num_trades'] == pytest.approx(0.2)
    assert et.red_violations({**base, 'num_trades': 96}, 'alt')['num_trades'] == pytest.approx(1 / 95)
