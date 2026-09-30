import numpy as np
import optuna
import pytest

import tpi


def test_default_is_all_components():
    assert tpi.resolve_components() == list(tpi.COMPONENTS)
    assert set(tpi.param_space()) == {p for c in tpi.COMPONENTS.values() for p in c['params']}


def test_subset_param_space():
    assert set(tpi.param_space(['supertrend'])) == {'supertrend_atr_period', 'supertrend_factor'}
    assert set(tpi.param_space('ema_cross')) == {'fast_ma', 'slow_ma'}
    assert set(tpi.param_space(['aroon', 'adx'])) == {'aroon_length', 'adx_period', 'threshold'}
    with pytest.raises(KeyError):
        tpi.param_space(['nope'])


def test_suggest_params_only_samples_selected_components():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    params = tpi.suggest_params(trial, ['supertrend'])
    assert set(params) == {'supertrend_atr_period', 'supertrend_factor'}
    assert set(trial.params) == {'supertrend_atr_period', 'supertrend_factor_scaled'}
    assert tpi.params_from_trial(trial, ['supertrend']) == params


def test_constraints_restricted_to_selected_components():
    st = {'supertrend_atr_period': 10, 'supertrend_factor': 3.0}
    assert tpi.params_valid(st, ['supertrend'])              # no KeyError on fast_ma
    assert not tpi.params_valid({'fast_ma': 30, 'slow_ma': 20}, ['ema_cross'])
    assert tpi.params_valid({'fast_ma': 20, 'slow_ma': 30}, ['ema_cross'])


def test_parabolic_sar_start_le_maximum():
    base = {'parabolic_sar_acceleration': 0.02}
    assert tpi.params_valid({**base, 'parabolic_sar_start': 0.2, 'parabolic_sar_maximum': 0.2},
                            ['parabolic_sar'])
    assert not tpi.params_valid({**base, 'parabolic_sar_start': 0.21, 'parabolic_sar_maximum': 0.2},
                                ['parabolic_sar'])


def test_single_component_and_single_group_aggregation(price):
    p = {'supertrend_atr_period': 10, 'supertrend_factor': 3.0,
         'fast_ma': 12, 'slow_ma': 26, 'aroon_length': 14, 'adx_period': 14, 'threshold': 25}
    t = tpi.tpi(price)
    t.calculate_tpi(p, 'long_short', ['supertrend'])
    sig = tpi.COMPONENTS['supertrend']['signal'](price, p)
    assert np.array_equal(t.tpi_score, sig)
    # one group (two perpetual components): plain mean of the two
    t.calculate_tpi(p, 'long_only', ['supertrend', 'ema_cross'])
    ema = tpi.COMPONENTS['ema_cross']['signal'](price, p)
    assert np.allclose(t.tpi_score, (sig + ema) / 2)
    # two groups: mean of group means
    t.calculate_tpi(p, 'long_only', ['supertrend', 'ema_cross', 'aroon'])
    ar = tpi.COMPONENTS['aroon']['signal'](price, p)
    assert np.allclose(t.tpi_score, ((sig + ema) / 2 + ar) / 2)


def test_unknown_mode_rejected(price):
    with pytest.raises(ValueError):
        tpi.tpi(price).calculate_tpi({'aroon_length': 14}, 'short_only', ['aroon'])
