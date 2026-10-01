"""PSAR tuning search space, changed 30 Sep 2026 (acceleration widened to
0.0004-0.1 step 0.0001: every target 60/70/80 is reached within +-1 trade)."""
import numpy as np
import optuna
import pytest

import tpi

PSAR = tpi.COMPONENTS['parabolic_sar']['params']


def test_psar_new_bounds():
    assert PSAR['parabolic_sar_acceleration'] == ('float', 0.0004, 0.1, 0.0001)
    # unchanged
    assert PSAR['parabolic_sar_start'] == ('float', 0.0, 1.0, 0.01)
    assert PSAR['parabolic_sar_maximum'] == ('float', 0.1, 1.0, 0.1)


def test_psar_old_grid_is_contained_in_new_grid():
    lo, hi, step = PSAR['parabolic_sar_acceleration'][1:]
    new = {round(lo + i * step, 6) for i in range(int(round((hi - lo) / step)) + 1)}
    old = {round(0.01 * i, 6) for i in range(1, 11)}
    assert old <= new and min(new) == 0.0004 and max(new) == 0.1 and len(new) == 997
    assert {0.0004, 0.001, 0.0013} <= new          # the values that reach 60 / 70 / 80 (+-1)


def test_psar_suggest_params_stays_on_grid_and_start_le_max_enforced():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    seen_low = False
    n_invalid = 0
    for _ in range(400):
        trial = study.ask()
        p = tpi.suggest_params(trial, ['parabolic_sar'])
        a = p['parabolic_sar_acceleration']
        assert 0.0004 <= a <= 0.1
        assert abs(a / 0.0001 - round(a / 0.0001)) < 1e-9
        seen_low |= a < 0.01
        valid = tpi.params_valid(p, ['parabolic_sar'])
        assert valid == (p['parabolic_sar_start'] <= p['parabolic_sar_maximum'])
        n_invalid += not valid
        study.tell(trial, 0.0)
    assert seen_low and n_invalid > 0      # the constraint is live in the widened space


@pytest.mark.parametrize('start,maximum,ok', [(0.02, 0.2, True), (0.2, 0.2, True),
                                              (0.21, 0.2, False), (0.0, 0.1, True)])
def test_psar_start_le_maximum(start, maximum, ok):
    p = {'parabolic_sar_start': start, 'parabolic_sar_acceleration': 0.0004,
         'parabolic_sar_maximum': maximum}
    assert tpi.params_valid(p, ['parabolic_sar']) is ok


def test_psar_lowest_acceleration_gives_a_valid_signal(price):
    p = {'parabolic_sar_start': 0.02, 'parabolic_sar_acceleration': 0.0004,
         'parabolic_sar_maximum': 0.2}
    sig = np.asarray(tpi.COMPONENTS['parabolic_sar']['signal'](price, p))
    assert len(sig) == len(price) and set(np.unique(sig)) <= {-1, 0, 1}
