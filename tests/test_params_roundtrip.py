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
