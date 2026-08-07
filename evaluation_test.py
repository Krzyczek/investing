import optuna_testing
import pandas as pd
from optuna_testing import instrument_strategy as istr
import buyhold_data
import data_import
import tpi
import numpy as np
import investment_metrics as im


# --- Parameter search space (must mirror optuna_testing.objective) ---
PARAM_BOUNDS = {
    'fast_ma':    (5, 100),
    'slow_ma':    (5, 100),
    'adx_period': (5, 100),
    'threshold':  (5, 50),
}
STEP_SIZE = 1        # optuna uses step=1 for every parameter
N_SIDE = 3           # 3 step-deviations on each side -> 7 columns total

# The 7 metrics of the Cobra color table (used for green/red and CoV)
TABLE_METRICS = ('max_dd', 'sortino', 'sharpe', 'profit_factor',
                 'pct_profitable', 'num_trades', 'omega')

MIN_GREEN = 5        # "5/7 green metrics at least and NO RED" per column


def eval(deposit: int, instrument, safe_investment: float = 0.03):

    strat_1 = optuna_testing.instrument_strategy('CDR.WA', instrument, deposit, safe_investment)
    strat_1.strategy_evaluation()

    study = strat_1.study

    pareto_fronts = strat_1.get_all_pareto_fronts(study)

    return pareto_fronts


def classify_metric(name, value):
    """Return 'green', 'yellow' or 'red' according to the Cobra table.
    max_dd is a POSITIVE fraction (0.25 == 25%), pct_profitable a fraction."""
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
        if value < 40 or value > 105:
            return 'red'
        return 'green' if value >= 45 else 'yellow'
    if name == 'omega':
        return 'red' if value < 1.1 else ('green' if value > 1.31 else 'yellow')
    raise ValueError(f"Unknown metric {name}")


def column_verdict(metrics):
    """Apply the guide's per-column rule: >=5/7 green and NO red."""
    colors = {m: classify_metric(m, metrics[m]) for m in TABLE_METRICS}
    greens = sum(1 for c in colors.values() if c == 'green')
    reds = sum(1 for c in colors.values() if c == 'red')
    return (greens >= MIN_GREEN and reds == 0), greens, reds, colors


def run_backtest(price, deposit, instrument, safe_investment, mode,
                 benchmark_returns, params):
    """Run one backtest for a given parameter set.
    Returns a metrics dict (7 table metrics + calmar/alpha), or None if the
    strategy got liquidated (automatic robustness failure)."""
    fast_ma = params['fast_ma']
    slow_ma = params['slow_ma']
    adx_period = params['adx_period']
    threshold = params['threshold']

    test_df = price.copy()
    # --- OBLICZANIE SYGNAŁU (Z poprawkami znoszącymi wehikuł czasu) ---
    tpi_signal = tpi.tpi(test_df)
    tpi_signal.calculate_tpi(slow_ma, fast_ma, adx_period, threshold, mode)
    # 1. Sygnał na koniec dzisiejszego dnia
    test_df['signal'] = tpi_signal.signal
    test_df = test_df.loc[test_df.index >= '2018-01-01']

    # 2. PRZESUNIĘCIE SYGNAŁU (Likwidacja wehikułu czasu)
    shifted_signal = test_df['signal'].shift(1).fillna(0)

    # 3. Zyski i Kapitał
    test_df['strat_return'] = test_df['return'] * shifted_signal
    test_df['equity'] = deposit * (1 + test_df['strat_return']).cumprod()

    # Liquidation check (same rule as in the optuna objective)
    if (1 + test_df['strat_return'] <= 0).any():
        return None

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

    num_trades = len(trade_returns)
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
            'alpha':          strat_metrics.alpha(benchmark_returns)}


def build_step_values(base, lo, hi, n_side=N_SIDE, step=STEP_SIZE):
    """Build the step-deviation column values around `base`.
    Follows the guide's rule: if you can't get n_side steps on one side,
    shift the extra steps to the other side so no columns are empty."""
    total = 2 * n_side + 1
    start = base - n_side * step
    if start < lo:
        start = lo
    if start + (total - 1) * step > hi:
        start = hi - (total - 1) * step
    if start < lo:
        return list(range(lo, hi + 1, step))
    return [start + k * step for k in range(total)]


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
                                   backtest_cache=None, verbose=False):
    """Robustness Factory parameter test for one candidate.

    For every parameter: perturb it over +-N_SIDE step deviations, run the
    backtest for each column and apply the color rule (>=5/7 green, no red).
    Also computes the CoV of every table metric across the columns, averaged
    into a per-parameter CoV and an overall CoV (the sheet's evaluation).

    Returns (passed_colors, overall_cov, per_param_report) or
    (False, None, reason) if disqualified (liquidation on any column)."""
    if backtest_cache is None:
        backtest_cache = {}

    per_param_report = {}
    all_columns_pass = True

    for param_name in PARAM_BOUNDS:
        lo, hi = PARAM_BOUNDS[param_name]
        # respect the fast_ma < slow_ma constraint while perturbing
        if param_name == 'fast_ma':
            hi = min(hi, base_params['slow_ma'] - 1)
        elif param_name == 'slow_ma':
            lo = max(lo, base_params['fast_ma'] + 1)

        step_values = build_step_values(base_params[param_name], lo, hi)

        metric_series = {m: [] for m in TABLE_METRICS}
        columns = []
        for value in step_values:
            test_params = dict(base_params)
            test_params[param_name] = value

            cache_key = tuple(sorted(test_params.items()))
            if cache_key not in backtest_cache:
                backtest_cache[cache_key] = run_backtest(
                    price, deposit, instrument, safe_investment, mode,
                    benchmark_returns, test_params)
            metrics = backtest_cache[cache_key]

            if metrics is None:
                return False, None, f"liquidated at {param_name}={value}"

            ok, greens, reds, colors = column_verdict(metrics)
            columns.append({'value': value, 'pass': ok,
                            'greens': greens, 'reds': reds, 'colors': colors})
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
    return all_columns_pass, overall_cov, per_param_report


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
                              verbose: bool = False):
    """Walk the Pareto fronts in order and run the Robustness Factory
    parameter test on every candidate of each front.

    PASS = every step-deviation column of every parameter has >=5/7 green
    metrics and NO red (Cobra color table), and - if cov_threshold is given -
    the overall CoV is also <= cov_threshold (0.10 restricts to 1st Class).

    As soon as one front contains at least one passing candidate, stop and
    return ALL passing candidates of that front. Returns [] otherwise."""
    dataframe = data_import.data_importer(instrument)
    dataframe.import_csv_file()
    price = dataframe.df

    # benchmark is identical for every candidate -> compute it once
    benchmark_metrics = buyhold_data.buyhold_benchmark(
        price.loc['2018-01-01':], deposit, instrument, safe_investment)
    benchmark_returns = benchmark_metrics['return']

    backtest_cache = {}   # avoids re-running duplicate parameter sets

    for layer_idx, front in enumerate(pareto_fronts, start=1):
        if layer_idx > max_fronts:
            break
        print(f"Evaluating Front {layer_idx} containing {len(front)} candidates...")

        passing_candidates = []

        for i, trial in enumerate(front):
            base_params = {k: trial.params[k] for k in PARAM_BOUNDS}

            colors_ok, overall_cov, report = candidate_parameter_robustness(
                price, deposit, instrument, safe_investment, mode,
                benchmark_returns, base_params, backtest_cache, verbose)

            if overall_cov is None:
                print(f"  Candidate {i+1}: DISQUALIFIED ({report})")
                continue

            passed = colors_ok and (cov_threshold is None
                                    or overall_cov <= cov_threshold)
            per_param_str = ", ".join(
                f"{p}={r['param_cov']:.2%}" for p, r in report.items())
            print(f"  Candidate {i+1}: colors {'OK' if colors_ok else 'FAIL'}, "
                  f"overall CoV = {overall_cov:.2%} ({cov_class(overall_cov)}) "
                  f"-> {'PASS' if passed else 'fail'} "
                  f"({per_param_str}) params={base_params}")

            if passed:
                base_run = run_backtest(price, deposit, instrument,
                                        safe_investment, mode,
                                        benchmark_returns, base_params)
                passing_candidates.append({
                    'front': layer_idx,
                    'candidate_idx': i + 1,
                    'trial': trial,
                    'params': base_params,
                    'overall_cov': overall_cov,
                    'cov_class': cov_class(overall_cov),
                    'per_param': report,
                    'base_metrics': base_run,
                })

        if passing_candidates:
            print(f"\nFront {layer_idx}: {len(passing_candidates)} candidate(s) "
                  f"passed the Robustness Factory test. Stopping here.")
            return passing_candidates

    print("\nNo candidate on any evaluated front passed the robustness test.")
    return []


if __name__ == "__main__":
    pareto = eval(12000, 'crypto', 0)
    robust_candidates = parameter_robustness_test(12000, 'crypto', pareto, 0)
    for c in robust_candidates:
        print(f"Front {c['front']} candidate {c['candidate_idx']}: "
              f"CoV={c['overall_cov']:.2%} ({c['cov_class']}), "
              f"params={c['params']}, metrics={c['base_metrics']}")