import buyhold_data
import data_import
import indicators
import investment_metrics as im
import numpy as np
import pandas as pd
import tpi
import optuna

# Wartości zwracane przez Optunę dla strategii odrzuconych przez filtry.
# Filtry: likwidacja, drawdown (close) > 60% oraz KAŻDA czerwona metryka
# wybranej tabeli kolorów (main/alt), w tym liczba transakcji.
# Od teraz takie próby są dodatkowo oznaczone jako NIEDOPUSZCZALNE przez
# trial.set_constraint(...), więc nie trafiają do frontów Pareto ani do testu
# odporności, a sampler NSGA-II nadal się na nich uczy (constrained domination).
PENALTY_VALUES = [-1000, 0]
MAX_CLOSE_DRAWDOWN = 0.60


def record_constraints(trial, violations: dict):
    """Store constraint violations on the trial (value > 0 == violated).
    Optuna 5: trial.set_constraint is read by NSGAIISampler (constrained
    domination) and by study.best_trials; the dict is also kept as a user
    attr so it is visible in trials_dataframe()."""
    for name, value in violations.items():
        trial.set_constraint(name, float(value))
    trial.set_user_attr('constraints', {k: float(v) for k, v in violations.items()})
    trial.set_user_attr('feasible', all(v <= 0 for v in violations.values()))


def is_feasible(trial) -> bool:
    """True for a COMPLETE trial that satisfied every filter. Trials from
    studies run before constraints were recorded are recognised by the
    [-1000, 0] penalty values."""
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return False
    constraints = getattr(trial, 'constraints', None) or {}
    if constraints:
        return all(v <= 0 for v in constraints.values())
    return not (trial.values is not None and list(trial.values) == PENALTY_VALUES)


INSTRUMENT_TYPES = ('crypto', 'stocks')   # crypto annualises at 365 days, stocks at 252
MODES = ('long_only', 'long_short')


def check_instrument_type(instrument_type):
    if instrument_type not in INSTRUMENT_TYPES:
        raise ValueError(f"instrument_type must be one of {INSTRUMENT_TYPES}, got {instrument_type!r}")
    return instrument_type


def check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return mode


class instrument_strategy():
    def __init__(self, instrument: str, instrument_type: str, deposit: int = 12000,
                 risk_free_rate: float = 0.03, data_dir: str = None, file_path: str = None):
        """instrument      - ticker whose CSV is loaded (e.g. 'BTC-USD', 'CDR.WA')
        instrument_type - 'crypto' (365-day annualisation) or 'stocks' (252)
        deposit         - starting cash (default 12000, as in the entry scripts)
        risk_free_rate  - annual rate as a decimal
        data_dir / file_path - where the CSV lives (see data_import.data_importer)"""
        self.instrument = instrument
        self.deposit = deposit
        self.instrument_type = check_instrument_type(instrument_type)
        self.risk_free_rate = risk_free_rate
        self.data_dir = data_dir
        self.file_path = file_path

    def load_price(self):
        dataframe = data_import.data_importer(self.instrument, data_dir=self.data_dir,
                                              file_path=self.file_path)
        dataframe.import_csv_file()
        return dataframe.df

    def strategy_evaluation(self, table: str = None, mode: str = 'long_short',
                            components=None, n_trials: int = 5000, n_jobs: int = -1,
                            seed: int = None, start: str = '2018-01-01', sampler=None,
                            in_sample_end: str = None):
        """Run the multi-objective (Sortino, Calmar) Optuna search.

        table      - Cobra table ('main'|'alt'): a trial with ANY red metric on
                     it is infeasible (graded constraint per metric).
                     None/'auto' (default) = evaluation_test.select_table by
                     history length; too little history -> no study, returns
                     an (empty) evaluation_test.AssetSkipped, also in self.skipped
        mode       - 'long_short' (default, the primary mode) or 'long_only'
                     (diagnostic flag only)
        components - TPI components to optimise (default: all registered)
        n_trials / n_jobs - Optuna budget; n_jobs=-1 uses all cores (threads)
        seed       - NSGA-II seed (None = not reproducible; with n_jobs != 1
                     thread scheduling can still change the order of trials)
        start      - first bar of the backtest window (indicators still use the
                     earlier history for warm-up)
        sampler    - optional custom Optuna sampler (overrides seed)
        in_sample_end - optional last in-sample date: the search only sees
                     bars up to it (hold-out = everything after, see
                     evaluation_test.holdout_report)
        Returns the study (also stored in self.study)."""
        check_mode(mode)
        components = tpi.resolve_components(components)
        # Import leniwy, bo evaluation_test importuje ten moduł.
        import evaluation_test

        price = self.load_price()
        self.price = price
        self.skipped = None
        self.study = None

        # wybór tabeli kolorów (main/alt) i ewentualne pominięcie aktywa
        choice = evaluation_test.resolve_table(price, table, start, self.instrument)
        self.table_choice = choice
        if choice['skip']:
            self.skipped = evaluation_test.AssetSkipped(self.instrument, choice['reason'], choice)
            return self.skipped
        table, start = choice['table'], choice['start']
        self.mode, self.table, self.components, self.start = mode, table, components, start
        self.in_sample_end = in_sample_end

        #price['return'] = price['close'].pct_change(fill_method=None)

        benchmark_metrics = buyhold_data.buyhold_benchmark(price.loc[start:in_sample_end], self.deposit, self.instrument_type, self.risk_free_rate)
        self.benchmark_metrics = benchmark_metrics
        self.benchmark_returns = benchmark_metrics['return']
        
        # Jedna definicja backtestu i liczby transakcji dla Optuny i testu
        # odporności: evaluation_test.run_backtest (te same kroki co wcześniej
        # tutaj: sygnał TPI na pełnej historii, przesunięcie o 1 dzień, kapitał
        # close/high/low, metryki).

        def objective(trial):
            params = tpi.suggest_params(trial, components)
            trial.set_user_attr('components', components)
            if not tpi.params_valid(params, components):
                raise optuna.TrialPruned()

            m = evaluation_test.run_backtest(price, self.deposit, self.instrument_type,
                                             self.risk_free_rate, mode,
                                             self.benchmark_returns, params,
                                             components=components, start=start,
                                             end=in_sample_end)
            if m is None:
                # strategy was liquidated on a short: metryki nieznane ->
                # każda metryka tabeli liczona jako czerwona (1.0)
                record_constraints(trial, {'liquidation': 1.0,
                                           'max_drawdown': 1.0 - MAX_CLOSE_DRAWDOWN,
                                           **{f'red_{k}': 1.0
                                              for k in evaluation_test.TABLE_METRICS}})
                return PENALTY_VALUES

            num_trades = m['num_trades']      # non-flat position segments
            trial.set_user_attr('num_trades', num_trades)
            # Naruszenia są stopniowane (0 = OK), żeby sampler wiedział, która
            # odrzucona próba była "bliżej" dopuszczalnej:
            #  - KAŻDA czerwona metryka wybranej tabeli (main/alt), w tym
            #    liczba transakcji: względna odległość od granicy czerwieni
            #    (evaluation_test.red_violations)
            #  - drawdown (close): nadwyżka ponad 60% od szczytu
            violations = {'liquidation': 0.0,
                          'max_drawdown': max(0.0, m['close_max_dd'] - MAX_CLOSE_DRAWDOWN)}
            violations.update({f'red_{k}': v for k, v in
                               evaluation_test.red_violations(m, table).items()})
            record_constraints(trial, violations)
            trial.set_user_attr('colors', {k: evaluation_test.classify_metric(k, m[k], table)
                                           for k in evaluation_test.TABLE_METRICS})
            if any(v > 0 for v in violations.values()):
                return PENALTY_VALUES   # reject (red metric / lost >60% from peak)

            return round(m['sortino'], 4), round(m['calmar'], 4)
        
        # Opcja 'maximize' mówi Optunie, że im większy wynik z return, tym lepiej
        # NSGA-II jawnie: w Optunie 5 domyślnym samplerem jest TPE, a
        # NSGA-II obsługuje ograniczenia (constrained domination)
        if sampler is None:
            sampler = optuna.samplers.NSGAIISampler(seed=seed)
        study = optuna.create_study(directions=['maximize','maximize'], sampler=sampler)
        # ustawienia badania zapisane w study (potrzebne w teście odporności)
        for key, value in {'instrument': self.instrument, 'instrument_type': self.instrument_type,
                           'mode': mode, 'table': table, 'table_reason': choice['reason'],
                           'table_source': choice['source'], 'components': components,
                           'start': str(start), 'in_sample_end': in_sample_end,
                           'deposit': self.deposit,
                           'risk_free_rate': self.risk_free_rate, 'seed': seed}.items():
            study.set_user_attr(key, value)

        print("Rozpoczynam poszukiwanie najlepszych parametrów...")
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs) # n_jobs=-1 używa wszystkich rdzeni procesora!
        self.study = study
        print("\n--- ZAKOŃCZONO OPTYMALIZACJĘ ---")
        best = study.best_trials   # constrained study -> feasible trials only
        lista=[]
        for trial in best:
                parametry = trial.params
                lista.append({'parametry':parametry})
        if not lista:
            lista=[{'parametry':0}]
            
        #print(f"Optymalne parametry: {study.best_params}")
        df_best_trials = pd.DataFrame(lista)
        self.best_trials = df_best_trials
        return study


    

    def get_all_pareto_fronts(self, study):
        def dominates(t1: optuna.trial.FrozenTrial, t2: optuna.trial.FrozenTrial, directions):
                """Checks if t1 dominates t2 across all study objectives."""
                better_in_at_least_one = False
                for v1, v2, direction in zip(t1.values, t2.values, directions):
                    if direction == optuna.study.StudyDirection.MINIMIZE:
                        if v1 > v2:
                            return False
                        if v1 < v2:
                            better_in_at_least_one = True
                    else:  # MAXIMIZE
                        if v1 < v2:
                            return False
                        if v1 > v2:
                            better_in_at_least_one = True
                return better_in_at_least_one

        
        # tylko próby DOPUSZCZALNE: odrzucone przez filtry ([-1000, 0]) nie
        # mogą tworzyć frontów Pareto
        completed = [t for t in study.trials if is_feasible(t)]

        # --- deduplicate by params ---
        seen = set()
        unique = []
        for t in completed:
            key = tuple(sorted(t.params.items()))  # dicts aren't hashable, tuples are
            if key not in seen:
                seen.add(key)
                unique.append(t)
        completed = unique
        # -----------------------------

        if not completed:
            return []

        S  = {t.number: [] for t in completed} #track dominated trials
        n  = {t.number: 0 for t in completed} #track domination counts
        fronts = [[]]

        for p in completed:
            for q in completed:
                if p.number == q.number:
                    continue
                if dominates(p, q, study.directions):
                    S[p.number].append(q)
                elif dominates(q, p, study.directions):
                    n[p.number] += 1

            if n[p.number] == 0:
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in S[p.number]:
                    n[q.number] -= 1
                    if n[q.number] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts


                  
        
    

