"""params_from_trial must invert suggest_params for every registered component,
including float steps that are not 1/scale (e.g. PSAR acceleration step 0.0005)."""
import optuna
import tpi


def test_params_from_trial_inverts_suggest_params():
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    comps = list(tpi.COMPONENTS)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    suggested = []

    def objective(trial):
        suggested.append(tpi.suggest_params(trial, comps))
        return 0.0

    study.optimize(objective, n_trials=50)
    for trial, params in zip(study.trials, suggested):
        back = tpi.params_from_trial(trial, comps)
        for k, v in params.items():
            assert abs(back[k] - v) < 1e-12, (k, v, back[k])
            lo, hi = tpi.param_space(comps)[k][1:3]
            assert lo - 1e-12 <= back[k] <= hi + 1e-12


import pytest


@pytest.mark.parametrize('spec', [('float', 0.0005, 0.1, 0.0005),   # PSAR range at eb6a9f1
                                  ('float', 0.0, 1.0, 0.05),
                                  ('float', 0.5, 5.0, 0.25),
                                  ('float', 0.0004, 0.1, 0.0001)])  # current PSAR range
def test_params_from_trial_roundtrip_for_steps_that_are_not_one_over_scale(monkeypatch, spec):
    """Fails on eb6a9f1 / 968129f (value * step) for the first three specs;
    the current PSAR step 0.0001 == 1/scale was not affected."""
    comp = dict(tpi.COMPONENTS['parabolic_sar'])
    comp['params'] = {**comp['params'], 'parabolic_sar_acceleration': spec}
    monkeypatch.setitem(tpi.COMPONENTS, 'parabolic_sar', comp)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
    for _ in range(40):
        trial = study.ask()
        p = tpi.suggest_params(trial, ['parabolic_sar'])
        back = tpi.params_from_trial(trial, ['parabolic_sar'])
        assert abs(back['parabolic_sar_acceleration'] - p['parabolic_sar_acceleration']) < 1e-12
        assert spec[1] - 1e-12 <= back['parabolic_sar_acceleration'] <= spec[2] + 1e-12
        study.tell(trial, 0.0)
