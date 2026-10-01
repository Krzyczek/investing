"""Final hold-out acceptance gate: robust in-sample AND hold-out total return > 0
AND hold-out Sortino > 0. Nothing accepted -> 'no accepted TPI', no substitute."""
import math

import pytest

import evaluation_test as et


def _ho(total_return, sortino):
    return {'skipped': False, 'metrics': {'total_return': total_return, 'sortino': sortino}}


def test_pass_case():
    v = et.accept_tpi(True, _ho(12.5, 0.8))
    assert v['verdict'] == 'ACCEPTED' and v['accepted']
    assert v['holdout_total_return'] == 12.5 and v['holdout_sortino'] == 0.8


@pytest.mark.parametrize('tr', [0.0, -5.0])
def test_fail_total_return_not_positive(tr):
    v = et.accept_tpi(True, _ho(tr, 1.0))
    assert v['verdict'] == 'REJECTED' and 'total return' in v['reason']
    assert 'Sortino' not in v['reason']


@pytest.mark.parametrize('so', [0.0, -0.3, math.nan])
def test_fail_sortino_not_positive(so):
    v = et.accept_tpi(True, _ho(3.0, so))
    assert v['verdict'] == 'REJECTED' and 'Sortino' in v['reason']
    assert 'total return' not in v['reason']


def test_fail_in_sample_robustness_even_with_good_holdout():
    v = et.accept_tpi(False, _ho(50.0, 3.0))
    assert v['verdict'] == 'REJECTED' and 'robustness' in v['reason']


def test_fail_no_holdout_or_liquidated():
    assert et.accept_tpi(True, None)['verdict'] == 'REJECTED'
    assert et.accept_tpi(True, {'skipped': False, 'metrics': None})['verdict'] == 'REJECTED'


def _cand(n):
    return {'front': 1, 'candidate_idx': n, 'components': ['supertrend'],
            'params': {'supertrend_atr_period': 10 + n, 'supertrend_factor': 3.0}}


def test_final_acceptance_keeps_only_accepted(monkeypatch):
    fake = {11: _ho(5.0, 1.0), 12: _ho(-1.0, 1.0), 13: _ho(5.0, -1.0)}
    monkeypatch.setattr(et, 'holdout_report',
                        lambda *a, **k: fake[a[2]['supertrend_atr_period']])
    out = et.final_acceptance(10000, 'crypto', [_cand(1), _cand(2), _cand(3)],
                              '2024-12-31', ticker='X')
    assert [c['candidate_idx'] for c in out['accepted']] == [1]
    assert [r['verdict'] for r in out['results']] == ['ACCEPTED', 'REJECTED', 'REJECTED']


def test_nothing_accepted_reports_no_accepted_tpi_without_substitute(monkeypatch, capsys):
    monkeypatch.setattr(et, 'holdout_report', lambda *a, **k: _ho(-1.0, -1.0))
    out = et.final_acceptance(10000, 'crypto', [_cand(1)], '2024-12-31', ticker='X')
    assert out['accepted'] == [] and 'no accepted TPI' in out['message']
    out = et.final_acceptance(10000, 'crypto', [], '2024-12-31', ticker='X')
    assert out['accepted'] == [] and out['results'] == []
    assert 'no accepted TPI' in out['message'] and 'robustness' in out['message']
    assert 'no accepted TPI' in capsys.readouterr().out


def test_real_holdout_feeds_the_gate(data_dir):
    p = {'supertrend_atr_period': 10, 'supertrend_factor': 3.0}
    ho = et.holdout_report(10000, 'crypto', p, '2019-12-31', 0.0, 'long_only', 'alt',
                           ['supertrend'], ticker='SYN-USD', data_dir=data_dir, verbose=False)
    v = et.accept_tpi(True, ho)
    assert v['holdout_total_return'] == ho['metrics']['total_return']
    assert v['holdout_sortino'] == ho['metrics']['sortino']
    assert v['verdict'] == ('ACCEPTED' if (v['holdout_total_return'] > 0
                                           and v['holdout_sortino'] > 0) else 'REJECTED')
