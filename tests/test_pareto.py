import optuna
import pytest

import evaluation_test as et
import optuna_testing as ot
from conftest import make_price


def _study_with(trials):
    study = optuna.create_study(directions=['maximize', 'maximize'])
    for t in trials:
        study.add_trial(t)
    return study


def _trial(x, values, constraints=None):
    return optuna.trial.create_trial(
        params={'aroon_length': x},
        distributions={'aroon_length': optuna.distributions.IntDistribution(5, 100)},
        values=values, constraints=constraints)


def test_infeasible_trials_never_in_pareto_fronts():
    # the infeasible trial has the BEST values, it must still be excluded
    study = _study_with([
        _trial(10, [9.0, 9.0], {'num_trades': 0.5, 'liquidation': 0.0, 'max_drawdown': 0.0}),
        _trial(11, [1.0, 2.0], {'num_trades': 0.0, 'liquidation': 0.0, 'max_drawdown': 0.0}),
        _trial(12, [2.0, 1.0], {'num_trades': 0.0, 'liquidation': 0.0, 'max_drawdown': 0.0}),
        _trial(13, [0.5, 0.5], {'num_trades': 0.0, 'liquidation': 0.0, 'max_drawdown': 0.0}),
        _trial(14, [-1000, 0], {'num_trades': 0.0, 'liquidation': 1.0, 'max_drawdown': 0.4}),
    ])
    fronts = ot.instrument_strategy.get_all_pareto_fronts(None, study)
    numbers = [[t.params['aroon_length'] for t in f] for f in fronts]
    assert sorted(numbers[0]) == [11, 12] and numbers[1] == [13]
    assert all(ot.is_feasible(t) for f in fronts for t in f)
    assert {t.params['aroon_length'] for t in study.best_trials} == {11, 12}


def test_legacy_penalised_trials_without_constraints_are_excluded():
    study = _study_with([_trial(10, [-1000, 0]), _trial(11, [0.1, 0.1])])
    fronts = ot.instrument_strategy.get_all_pareto_fronts(None, study)
    assert [[t.params['aroon_length'] for t in f] for f in fronts] == [[11]]


def test_all_infeasible_gives_no_fronts():
    study = _study_with([_trial(10, [-1000, 0], {'num_trades': 1.0})])
    assert ot.instrument_strategy.get_all_pareto_fronts(None, study) == []


def test_real_study_fronts_and_robustness_skip_infeasible(tmp_path, capsys):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    make_price(n=3000).drop(columns='return').to_csv(tmp_path / 'SYN-USD.csv')
    data_dir = str(tmp_path)
    s = ot.instrument_strategy('SYN-USD', 'crypto', 10000, 0.0, data_dir=data_dir)
    study = s.strategy_evaluation(mode='long_only', components=['supertrend'],
                                  n_trials=40, n_jobs=1, seed=1)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert all(set(t.constraints) == {'num_trades', 'liquidation', 'max_drawdown'} for t in completed)
    bad = [t for t in completed if not ot.is_feasible(t)]
    good = [t for t in completed if ot.is_feasible(t)]
    assert bad and good          # this seed produces both kinds
    fronts = s.get_all_pareto_fronts(study)
    in_fronts = [t.number for f in fronts for t in f]
    assert in_fronts and all(ot.is_feasible(t) for f in fronts for t in f)
    assert not set(in_fronts) & {t.number for t in bad}
    # robustness test is handed an infeasible trial -> it is skipped, not tested
    if bad:
        res = et.parameter_robustness_test(10000, 'crypto', [[bad[0]]], 0.0, mode='long_only',
                                           ticker='SYN-USD', data_dir=data_dir)
        assert res == [] and 'SKIPPED (infeasible' in capsys.readouterr().out
