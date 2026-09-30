import optuna_testing
import pandas as pd
from optuna_testing import instrument_strategy as istr
import buyhold_data
import data_import
import tpi
import numpy as np
import investment_metrics as im


# --- Parameter search space (must mirror optuna_testing.objective) ---


N_SIDE = 3           # 3 step-deviations on each side -> 7 columns total

# The 7 metrics of the Cobra color table (used for green/red and CoV)
TABLE_METRICS = ('max_dd', 'sortino', 'sharpe', 'profit_factor',
                 'pct_profitable', 'num_trades', 'omega')

MIN_GREEN = 5        # "5/7 green metrics at least and NO RED" per column

# Two Cobra tables: 'main' and 'alt' ("with min 3 yr data"). They are
# identical except for the # of trades bands:
#   table -> (red below, green from, green up to; red above)
#   main: red <40 or >105, yellow 40-44, green 45-105
#   alt:  red <30 or >95,  yellow 30-34, green 35-95
# The table is always chosen explicitly (table='main'|'alt'); there is no
# automatic rule based on history length.
TRADE_COUNT_BANDS = {'main': (40, 45, 105),
                     'alt':  (30, 35, 95)}
COLOR_TABLES = tuple(TRADE_COUNT_BANDS)


def check_table(table):
    if table not in TRADE_COUNT_BANDS:
        raise ValueError(f"table must be one of {COLOR_TABLES}, got {table!r}")
    return table


def count_trades(position) -> int:
    """THE trade definition used everywhere (Optuna filter, robustness table,
    reports): the number of non-flat position segments. A segment is a run of
    consecutive bars with the same position; flat (0) runs are not trades.
    long -> short is two trades, long -> flat -> long is two trades."""
    pos = pd.Series(np.asarray(position, dtype=float)).fillna(0)
    segment_start = pos != pos.shift()
    return int((segment_start & (pos != 0)).sum())


def dedupe_fronts(pareto_fronts):
    seen = set()
    clean = []
    for front in pareto_fronts:
        kept = []
        for trial in front:
            key = tuple(sorted(trial.params.items()))
            if key not in seen:
                seen.add(key)
                kept.append(trial)
        clean.append(kept)
    return clean

def eval(deposit: int, instrument, safe_investment: float = 0.03, *,
         ticker: str, data_dir: str = None, file_path: str = None,
         mode: str = 'long_short', table: str = 'main', components=None,
         n_trials: int = 5000, n_jobs: int = -1, seed: int = None,
         start: str = '2018-01-01', in_sample_end: str = None):
    """Optuna search -> all Pareto fronts (feasible trials only).
    `instrument` is the instrument TYPE ('crypto' or 'stocks'); the data is
    chosen by `ticker` (+ optional data_dir / file_path). Before, this ran
    'CDR.WA' labelled with whatever type was passed."""

    strat_1 = optuna_testing.instrument_strategy(ticker, instrument, deposit, safe_investment,
                                                 data_dir=data_dir, file_path=file_path)
    strat_1.strategy_evaluation(table=table, mode=mode, components=components,
                                n_trials=n_trials, n_jobs=n_jobs, seed=seed,
                                start=start, in_sample_end=in_sample_end)

    study = strat_1.study

    pareto_fronts = strat_1.get_all_pareto_fronts(study)

    return pareto_fronts


def classify_metric(name, value, table='main'):
    """Return 'green', 'yellow' or 'red' according to the Cobra table
    (table='main' or 'alt'; they differ only in num_trades).
    max_dd is a POSITIVE fraction (0.25 == 25%), pct_profitable a fraction."""
    check_table(table)
    if not np.isfinite(value):
        # inf profit factor (no losing trades) is the best possible outcome
        if name == 'profit_factor' and value == np.inf:
            return 'green'
        return 'red'
    if name == 'max_dd':
        return 'red' if value > 0.40 else ('green' if value < 0.25 else 'yellow')
    if name == 'sortino':
        return 'red' if value < 2 else ('green' if value > 2.90 else 'yellow')
    if name == 'sharpe':
        return 'red' if value < 1 else ('green' if value > 2 else 'yellow')
    if name == 'profit_factor':
        return 'red' if value < 2 else ('green' if value > 4 else 'yellow')
    if name == 'pct_profitable':
        return 'red' if value < 0.35 else ('green' if value > 0.50 else 'yellow')
    if name == 'num_trades':
        red_below, green_from, green_to = TRADE_COUNT_BANDS[table]
        if value < red_below or value > green_to:
            return 'red'
        return 'green' if value >= green_from else 'yellow'
    if name == 'omega':
        return 'red' if value < 1.1 else ('green' if value > 1.31 else 'yellow')
    raise ValueError(f"Unknown metric {name}")


def column_verdict(metrics, table='main'):
    """Apply the guide's per-column rule: >=5/7 green and NO red."""
    colors = {m: classify_metric(m, metrics[m], table) for m in TABLE_METRICS}
    greens = sum(1 for c in colors.values() if c == 'green')
    reds = sum(1 for c in colors.values() if c == 'red')
    return (greens >= MIN_GREEN and reds == 0), greens, reds, colors


def run_backtest(price, deposit, instrument, safe_investment, mode,
                 benchmark_returns, params, components=None,
                 start='2018-01-01', end=None):
    """Run one backtest for a given parameter set.
    Returns a metrics dict (7 table metrics + calmar/alpha), or None if the
    strategy got liquidated (automatic robustness failure).

    components - TPI components used (default: all registered)
    start/end  - backtest window. Indicators are computed on all history
                 before `start` (warm-up) but on NOTHING after `end`: the
                 price is cut at `end` before the signal is calculated, so no
                 future bar can influence the window (no lookahead)."""
    
    test_df = price.copy()
    if end is not None:
        test_df = test_df.loc[:end]
    # --- OBLICZANIE SYGNAŁU (Z poprawkami znoszącymi wehikuł czasu) ---
    tpi_signal = tpi.tpi(test_df)
    tpi_signal.calculate_tpi(params, mode, components)
    # 1. Sygnał na koniec dzisiejszego dnia
    test_df['signal'] = tpi_signal.signal
    test_df = test_df.loc[test_df.index >= pd.Timestamp(start)]

    # 2. PRZESUNIĘCIE SYGNAŁU (Likwidacja wehikułu czasu)
    shifted_signal = test_df['signal'].shift(1).fillna(0)

    # 3. Zyski i Kapitał
    test_df['strat_return'] = test_df['return'] * shifted_signal
    test_df['equity'] = deposit * (1 + test_df['strat_return']).cumprod()

    # Liquidation check (same rule as in the optuna objective)
    if (1 + test_df['strat_return'] <= 0).any():
        return None

    # Close-based max drawdown (used by the optuna ">60% from peak" rule)
    running_peak = test_df['equity'].cummax()
    close_max_dd = abs(((test_df['equity'] - running_peak) / running_peak).min())

    # 4. Zwroty wewnątrzdzienne (High/Low)
    test_df['return_high'] = (test_df['high'] - test_df['close'].shift(1)) / test_df['close'].shift(1)
    test_df['return_low'] = (test_df['low'] - test_df['close'].shift(1)) / test_df['close'].shift(1)

    # 5. Kapitał High/Low z użyciem prawidłowego (przesuniętego) sygnału
    prev_equity = test_df['equity'].shift(1).fillna(deposit)
    rh = test_df['return_high'].fillna(0)
    rl = test_df['return_low'].fillna(0)

    conditions = [shifted_signal == 1, shifted_signal == -1]
    test_df['equity_high'] = np.select(conditions,
        [prev_equity * (1 + rh),      # long: high is best
         prev_equity * (1 - rl)],     # short: low is best
        default=prev_equity)

    test_df['equity_low'] = np.select(conditions,
        [prev_equity * (1 + rl),      # long: low is worst
         prev_equity * (1 - rh)],     # short: high is worst
        default=prev_equity)

    # 6. BEZPIECZNE wypełnianie braków (tylko dla kolumn kapitałowych!)
    test_df['equity'] = test_df['equity'].fillna(deposit)
    test_df['equity_high'] = test_df['equity_high'].fillna(deposit)
    test_df['equity_low'] = test_df['equity_low'].fillna(deposit)

    strat_metrics = im.metrics(df=test_df, investment_type=instrument,
                               risk_free_rate=safe_investment,
                               returns_column='strat_return',
                               starting_equity=deposit,
                               high='equity_high', low='equity_low',
                               close='equity', verbose=False)

    # --- Pessimistic intra-trade Max DD (same method as calmar_ratio) ---
    rolling_peaks = test_df['equity_high'].cummax()
    drawdowns = (test_df['equity_low'] - rolling_peaks) / rolling_peaks
    max_dd = abs(drawdowns.min())          # positive fraction, 0.25 == 25%

    # --- Trade statistics from position segments ---
    pos = shifted_signal
    seg_id = (pos != pos.shift()).cumsum()
    trade_returns = []
    for _, seg in test_df.groupby(seg_id):
        if pos.loc[seg.index[0]] == 0:
            continue                        # flat period, not a trade
        trade_returns.append((1 + seg['strat_return']).prod() - 1)

    num_trades = count_trades(pos)          # == len(trade_returns)
    if num_trades > 0:
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r < 0]
        pct_profitable = len(wins) / num_trades
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    else:
        pct_profitable = 0.0
        profit_factor = 0.0

    return {'max_dd':         max_dd,
            'sortino':        strat_metrics.sortino_ratio(),
            'sharpe':         strat_metrics.sharpe_ratio(),
            'profit_factor':  profit_factor,
            'pct_profitable': pct_profitable,
            'num_trades':     num_trades,
            'omega':          strat_metrics.omega_ratio(),
            # extra metrics for the downstream pipeline (not in the table)
            'calmar':         strat_metrics.calmar_ratio(),
            'alpha':          strat_metrics.alpha(benchmark_returns),
            'close_max_dd':   close_max_dd}


def build_step_values(base, lo, hi, step, is_int, n_side=N_SIDE):
    total = 2 * n_side + 1
    start = base - n_side * step
    if start < lo:
        start = lo
    if start + (total - 1) * step > hi:
        start = hi - (total - 1) * step
    if start < lo:
        vals = np.arange(lo, hi + step / 2, step)   # float-safe fallback
    else:
        vals = start + step * np.arange(total)
    vals = np.round(vals, 10)
    return [int(v) for v in vals] if is_int else [float(v) for v in vals]


def coefficient_of_variation(values):
    """CoV = sample std / |mean|, matching the sheet's STDEV/AVERAGE.
    Non-finite values (e.g. inf profit factor) are excluded."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan
    mean = arr.mean()
    if np.isclose(mean, 0):
        return np.inf
    return arr.std(ddof=1) / abs(mean)


def candidate_parameter_robustness(price, deposit, instrument, safe_investment,
                                   mode, benchmark_returns, base_params,
                                   backtest_cache=None, verbose=False,
                                   table='main', components=None,
                                   start='2018-01-01', end=None):
    """Robustness Factory parameter test for one candidate.

    For every parameter: perturb it over +-N_SIDE step deviations, run the
    backtest for each column and apply the color rule (>=5/7 green, no red).
    Also computes the CoV of every table metric across the columns, averaged
    into a per-parameter CoV and an overall CoV (the sheet's evaluation).

    Returns (passed_colors, overall_cov, per_param_report) or
    (False, None, reason) if disqualified (liquidation on any column)."""
    if backtest_cache is None:
        backtest_cache = {}
    window = dict(components=components, start=start, end=end)

    per_param_report = {}
    all_columns_pass = True

    # Base run is needed by the improvement loop (guide: adopt settings that
    # improve drawdown/Sharpe/Sortino). Cached, so this is free on re-scans.
    base_key = tuple(sorted(base_params.items()))
    if base_key not in backtest_cache:
        backtest_cache[base_key] = run_backtest(
            price, deposit, instrument, safe_investment, mode,
            benchmark_returns, base_params, **window)
    base_metrics = backtest_cache[base_key]
    if base_metrics is None:
        return False, None, "liquidated at base parameters"

    for param_name, (ptype, lo, hi, step) in tpi.param_space(components).items():
        step_values = build_step_values(
            base=base_params[param_name],
            lo=lo, hi=hi,
            step=step,
            is_int=(ptype == 'int'),
            )        # respect the fast_ma < slow_ma constraint while perturbing
        step_values = [v for v in step_values
                       if tpi.params_valid({**base_params, param_name: v}, components)]

        

        metric_series = {m: [] for m in TABLE_METRICS}
        columns = []
        for value in step_values:
            test_params = dict(base_params)
            test_params[param_name] = value

            cache_key = tuple(sorted(test_params.items()))
            if cache_key not in backtest_cache:
                backtest_cache[cache_key] = run_backtest(
                    price, deposit, instrument, safe_investment, mode,
                    benchmark_returns, test_params, **window)
            metrics = backtest_cache[cache_key]

            if metrics is None:
                return False, None, f"liquidated at {param_name}={value}"

            ok, greens, reds, colors = column_verdict(metrics, table)
            columns.append({'value': value, 'pass': ok,
                            'greens': greens, 'reds': reds, 'colors': colors,
                            'metrics': metrics})
            if not ok:
                all_columns_pass = False
                if verbose:
                    bad = [f"{m}={metrics[m]:.3f}({c})"
                           for m, c in colors.items() if c != 'green']
                    print(f"      {param_name}={value}: {greens} green / "
                          f"{reds} red -> FAIL [{', '.join(bad)}]")
            for m in TABLE_METRICS:
                metric_series[m].append(metrics[m])

        metric_covs = {m: coefficient_of_variation(v)
                       for m, v in metric_series.items()}
        finite_covs = [c for c in metric_covs.values() if np.isfinite(c)]
        per_param_report[param_name] = {
            'step_values': step_values,
            'columns': columns,
            'metric_covs': metric_covs,
            'param_cov': float(np.mean(finite_covs)) if finite_covs else np.nan,
        }

    overall_cov = float(np.nanmean([r['param_cov']
                                    for r in per_param_report.values()]))
    per_param_report['_base_metrics'] = base_metrics
    return all_columns_pass, overall_cov, per_param_report


def improved(col_metrics, base_metrics):
    """Guide's improvement rule: a perturbed setting is 'new and improved'
    if it is no worse on drawdown, Sharpe and Sortino, and strictly better
    on at least one of them."""
    no_worse = (col_metrics['max_dd'] <= base_metrics['max_dd'] and
                col_metrics['sharpe'] >= base_metrics['sharpe'] and
                col_metrics['sortino'] >= base_metrics['sortino'])
    strictly_better = (col_metrics['max_dd'] < base_metrics['max_dd'] or
                       col_metrics['sharpe'] > base_metrics['sharpe'] or
                       col_metrics['sortino'] > base_metrics['sortino'])
    return no_worse and strictly_better


def candidate_robustness_with_improvement(price, deposit, instrument,
                                          safe_investment, mode,
                                          benchmark_returns, base_params,
                                          backtest_cache=None, verbose=False,
                                          max_improvement_iters=3,
                                          table='main', components=None,
                                          start='2018-01-01', end=None):
    """Robustness scan + the guide's iterative improvement loop.

    'If you stumble across new settings that improve the strategy's
    performance overall (particularly the drawdown, Sharpe and Sortino)
    ... do the same thing again with the new and improved setting.'

    After each full scan, look for a PASSING column that `improved()` the
    base. If found, promote it to the new base and re-scan around it.
    Capped at max_improvement_iters to avoid endless hill-climbing (which
    would itself be a form of overfitting to the test).

    Returns (colors_ok, overall_cov, report, final_params)."""
    if backtest_cache is None:
        backtest_cache = {}

    params = dict(base_params)
    colors_ok, overall_cov, report = candidate_parameter_robustness(
        price, deposit, instrument, safe_investment, mode,
        benchmark_returns, params, backtest_cache, verbose, table=table,
        components=components, start=start, end=end)
    if overall_cov is None:
        return colors_ok, overall_cov, report, params

    for _ in range(max_improvement_iters):
        base_metrics = report['_base_metrics']

        # best passing & improving column across all parameters
        best = None
        for param_name in tpi.param_space(components):
            for col in report[param_name]['columns']:
                if col['value'] == params[param_name]:
                    continue                      # that's the base itself
                if not col['pass']:
                    continue                      # never adopt a failing column
                if not improved(col['metrics'], base_metrics):
                    continue
                key = (col['metrics']['sortino'] - base_metrics['sortino']) \
                    + (col['metrics']['sharpe'] - base_metrics['sharpe']) \
                    + (base_metrics['max_dd'] - col['metrics']['max_dd'])
                if best is None or key > best[0]:
                    best = (key, param_name, col['value'])

        if best is None:
            break                                 # nothing better -> converged

        _, param_name, value = best
        if verbose:
            print(f"    improvement: {param_name} "
                  f"{params[param_name]} -> {value}, re-scanning...")
        params[param_name] = value

        # 'do the same thing again with the new and improved setting':
        # full re-scan around the promoted base (cache makes this cheap)
        new_ok, new_cov, new_report = candidate_parameter_robustness(
            price, deposit, instrument, safe_investment, mode,
            benchmark_returns, params, backtest_cache, verbose, table=table,
        components=components, start=start, end=end)
        if new_cov is None:
            # promoted base liquidates somewhere in its neighbourhood;
            # 'sacrificing parameter robustness for performance' -> revert
            params[param_name] = base_params[param_name]
            break
        if not new_ok and colors_ok:
            # guide: keep the setting only if robustness is NOT sacrificed
            params[param_name] = base_params[param_name]
            break
        colors_ok, overall_cov, report = new_ok, new_cov, new_report
        base_params = dict(params)

    return colors_ok, overall_cov, report, params


def cov_class(overall_cov):
    if overall_cov <= 0.10:
        return '1st Class'
    if overall_cov <= 0.20:
        return '2nd Class'
    if overall_cov <= 0.30:
        return '3rd Class'
    return 'Economy Class'


def parameter_robustness_test(deposit: int, instrument, pareto_fronts,
                              safe_investment: float = 0.00,
                              mode: str = 'long_short',
                              cov_threshold: float = None,
                              max_fronts: int = 10,
                              verbose: bool = False,
                              table: str = 'main', *,
                              ticker: str, data_dir: str = None,
                              file_path: str = None, components=None,
                              start: str = '2018-01-01',
                              in_sample_end: str = None):
    """Walk the Pareto fronts in order and run the Robustness Factory
    parameter test on every candidate of each front.

    PASS = every step-deviation column of every parameter has >=5/7 green
    metrics and NO red (Cobra color table), and - if cov_threshold is given -
    the overall CoV is also <= cov_threshold (0.10 restricts to 1st Class).

    Every front up to max_fronts is evaluated (front rank is based on the
    optimisation objectives, which say nothing about robustness, so later
    fronts can hold MORE robust candidates). All passing candidates are
    returned sorted by overall CoV (most robust first), then front rank.
    Each candidate is first run through the guide's improvement loop, so
    the returned params may differ from the trial's originals. Returns []
    if nothing passes. `table` selects the Cobra table ('main' or 'alt').

    `instrument` is the instrument TYPE ('crypto'/'stocks'); the price data
    comes from `ticker` (+ data_dir / file_path). `mode` must match the mode
    the study was optimised with. `components` defaults to the components
    recorded on each trial (user attr), else all registered components.
    `in_sample_end` must match the study's: the test only uses bars up to it."""
    check_table(table)
    optuna_testing.check_mode(mode)
    optuna_testing.check_instrument_type(instrument)
    dataframe = data_import.data_importer(ticker, data_dir=data_dir, file_path=file_path)
    dataframe.import_csv_file()
    price = dataframe.df

    # benchmark is identical for every candidate -> compute it once
    benchmark_metrics = buyhold_data.buyhold_benchmark(
        price.loc[start:in_sample_end], deposit, instrument, safe_investment)
    benchmark_returns = benchmark_metrics['return']

    backtest_cache = {}   # avoids re-running duplicate parameter sets

    passing_candidates = []
    seen_params = set()   # improvement loop can converge to duplicates

    for layer_idx, front in enumerate(pareto_fronts, start=1):
        if layer_idx > max_fronts:
            break
        print(f"Evaluating Front {layer_idx} containing {len(front)} candidates...")
        
        for i, trial in enumerate(front):
            if not optuna_testing.is_feasible(trial):
                # filtered-out trials ([-1000, 0]) are never robustness candidates
                print(f"  Candidate {i+1}: SKIPPED (infeasible trial #{trial.number})")
                continue
            trial_components = tpi.resolve_components(
                components if components is not None
                else trial.user_attrs.get('components'))
            base_params = tpi.params_from_trial(trial, trial_components)

            colors_ok, overall_cov, report, final_params = \
                candidate_robustness_with_improvement(
                    price, deposit, instrument, safe_investment, mode,
                    benchmark_returns, base_params, backtest_cache, verbose,
                    table=table, components=trial_components, start=start,
                    end=in_sample_end)

            if overall_cov is None:
                print(f"  Candidate {i+1}: DISQUALIFIED ({report})")
                continue

            passed = colors_ok and (cov_threshold is None
                                    or overall_cov <= cov_threshold)
            per_param_str = ", ".join(
                f"{p}={r['param_cov']:.2%}" for p, r in report.items()
                if not p.startswith('_'))
            improved_str = ("" if final_params == base_params
                            else f" (improved from {base_params})")
            print(f"  Candidate {i+1}: colors {'OK' if colors_ok else 'FAIL'}, "
                  f"overall CoV = {overall_cov:.2%} ({cov_class(overall_cov)}) "
                  f"-> {'PASS' if passed else 'fail'} "
                  f"({per_param_str}) params={final_params}{improved_str}")

            if passed:
                params_key = tuple(sorted(final_params.items()))
                if params_key in seen_params:
                    continue
                seen_params.add(params_key)
                passing_candidates.append({
                    'front': layer_idx,
                    'candidate_idx': i + 1,
                    'components': trial_components,
                    'trial': trial,
                    'params': final_params,
                    'original_params': base_params,
                    'overall_cov': overall_cov,
                    'cov_class': cov_class(overall_cov),
                    'per_param': report,
                    'base_metrics': report['_base_metrics'],
                })

    if not passing_candidates:
        print("\nNo candidate on any evaluated front passed the robustness test.")
        return []

    # Robustness first (the guide's priority), front rank as tie-breaker
    passing_candidates.sort(key=lambda c: (c['overall_cov'], c['front']))
    print(f"\n{len(passing_candidates)} candidate(s) passed across all "
          f"evaluated fronts, sorted by overall CoV.")
    return passing_candidates


def holdout_report(deposit: int, instrument, params: dict, in_sample_end: str,
                   safe_investment: float = 0.0, mode: str = 'long_short',
                   table: str = 'main', components=None, *, ticker: str,
                   data_dir: str = None, file_path: str = None, end: str = None,
                   verbose: bool = True):
    """Table metrics and Cobra colours on the HELD-BACK period, i.e. the bars
    after `in_sample_end` (up to `end`, default: last bar).

    Indicators are computed on the full history up to `end` (the in-sample
    bars are the warm-up); nothing after `end` is used. Like every window in
    run_backtest, the hold-out starts flat on its first bar. Note: the trade
    count bands are meant for multi-year histories, so on a short hold-out
    the num_trades colour is mostly informative.
    Returns a dict (metrics None if the strategy was liquidated)."""
    check_table(table)
    optuna_testing.check_mode(mode)
    optuna_testing.check_instrument_type(instrument)
    components = tpi.resolve_components(components)
    dataframe = data_import.data_importer(ticker, data_dir=data_dir, file_path=file_path)
    dataframe.import_csv_file()
    price = dataframe.df

    in_sample = price.loc[:in_sample_end]
    after = price.index[price.index > in_sample.index[-1]] if len(in_sample) else price.index
    if end is not None:
        after = after[after <= price.loc[:end].index[-1]]
    if len(after) == 0:
        raise ValueError(f"No bars after in_sample_end={in_sample_end} for {ticker}")
    holdout_start, holdout_end = after[0], after[-1]

    benchmark = buyhold_data.buyhold_benchmark(
        price.loc[holdout_start:holdout_end], deposit, instrument, safe_investment)
    metrics = run_backtest(price, deposit, instrument, safe_investment, mode,
                           benchmark['return'], params, components=components,
                           start=holdout_start, end=holdout_end)
    report = {'ticker': ticker, 'mode': mode, 'table': table,
              'components': components, 'params': dict(params),
              'holdout_start': holdout_start, 'holdout_end': holdout_end,
              'bars': len(after), 'benchmark': benchmark, 'metrics': metrics,
              'colors': None, 'greens': None, 'reds': None, 'passed': False}
    if metrics is not None:
        ok, greens, reds, colors = column_verdict(metrics, table)
        report.update(colors=colors, greens=greens, reds=reds, passed=ok)
    if verbose:
        print(f"Hold-out {ticker} {holdout_start.date()} -> {holdout_end.date()} "
              f"({len(after)} bars, {mode}, table={table}):")
        if metrics is None:
            print("  LIQUIDATED")
        else:
            for m in TABLE_METRICS:
                print(f"  {m:15s} {metrics[m]:10.4f}  {report['colors'][m]}")
            print(f"  calmar {metrics['calmar']:.4f}, alpha vs B&H {metrics['alpha']:.2f} pp "
                  f"(B&H return {benchmark['return']:.2f}%) -> "
                  f"{report['greens']} green / {report['reds']} red, "
                  f"{'PASS' if report['passed'] else 'FAIL'}")
    return report


if __name__ == "__main__":
    # Ustawienia uruchomienia (CSV: <repo>/dane_cenowe/<TICKER>.csv lub $TPI_DATA_DIR)
    TICKER = 'BTC-USD'
    INSTRUMENT_TYPE = 'crypto'     # 'crypto' (365) albo 'stocks' (252)
    DEPOSIT = 12000
    RISK_FREE = 0
    MODE = 'long_short'
    TABLE = 'main'

    pareto = eval(DEPOSIT, INSTRUMENT_TYPE, RISK_FREE, ticker=TICKER, mode=MODE, table=TABLE)
    pareto = dedupe_fronts(pareto)
    robust_candidates = parameter_robustness_test(DEPOSIT, INSTRUMENT_TYPE, pareto, RISK_FREE,
                                                  mode=MODE, table=TABLE, ticker=TICKER)
    for c in robust_candidates:
        print(f"Front {c['front']} candidate {c['candidate_idx']}: "
              f"CoV={c['overall_cov']:.2%} ({c['cov_class']}), "
              f"params={c['params']}, metrics={c['base_metrics']}")