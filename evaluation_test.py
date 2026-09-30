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


# ---------- automatic colour-table choice by history length ----------
# Decided with Krzyczek:
#   MAIN  first valid bar (first non-NaN OHLC row) on or before the backtest
#         start (default 2018-01-01); a first bar exactly ON the start is MAIN
#   ALT   first valid bar after the start AND at least MIN_HISTORY_YEARS
#         calendar years from the first valid bar to the last complete bar
#   SKIP  less than that: no TPI for the asset (logged, never raised)
# The last row of the loaded data is taken as the last complete bar: the
# incomplete current bar must already be removed by the data source (the TA
# data store drops it).
# ALT assets have no history before the start, so their backtest window
# starts ALT_WARMUP_BARS valid bars after the first bar (those bars are the
# indicator warm-up). 200 bars covers the longest lookback in the search
# space (EMA / ADX / Aroon up to 100, ADX needs ~2x its period to settle) and
# is the same for every trial, so all trials of a study share one window.
MIN_HISTORY_YEARS = 3
ALT_WARMUP_BARS = 200


class AssetSkipped(list):
    """Result for an asset that gets no TPI (too little history).
    It is an EMPTY list, so code iterating over Pareto fronts / candidates
    simply does nothing, and multi-asset loops continue. Check with
    isinstance(result, AssetSkipped); .reason says why."""
    skipped = True

    def __init__(self, ticker, reason, choice=None):
        super().__init__()
        self.ticker = ticker
        self.reason = reason
        self.choice = choice or {}

    def __repr__(self):
        return f"AssetSkipped({self.ticker!r}: {self.reason})"


def select_table(price_df, start='2018-01-01', min_years=MIN_HISTORY_YEARS,
                 warmup_bars=ALT_WARMUP_BARS) -> dict:
    """Choose the Cobra table from the asset's history length (rules above).
    Returns a dict: table ('main'|'alt'|None), skip (bool), reason, start
    (effective backtest start), first_bar, last_bar, history_years."""
    ohlc = [c for c in ('open', 'high', 'low', 'close') if c in price_df.columns]
    valid = price_df.dropna(subset=ohlc)
    start_ts = pd.Timestamp(start)
    if valid.empty:
        return {'table': None, 'skip': True, 'reason': 'no valid OHLC bars',
                'start': None, 'first_bar': None, 'last_bar': None, 'history_years': 0.0}
    first, last = valid.index[0], valid.index[-1]
    years = (last - first).days / 365.25
    out = {'first_bar': first, 'last_bar': last, 'history_years': round(years, 2)}
    if first <= start_ts:
        return {**out, 'table': 'main', 'skip': False, 'start': start,
                'reason': (f"first valid bar {first.date()} is on or before the backtest "
                           f"start {start_ts.date()} -> MAIN table")}
    if last >= first + pd.DateOffset(years=min_years):
        eff = valid.index[min(warmup_bars, len(valid) - 1)]
        return {**out, 'table': 'alt', 'skip': False, 'start': eff,
                'reason': (f"first valid bar {first.date()} is after {start_ts.date()} and "
                           f"{years:.2f} years to the last bar {last.date()} (>= {min_years}) "
                           f"-> ALT table; backtest starts {eff.date()} after "
                           f"{warmup_bars} warm-up bars")}
    return {**out, 'table': None, 'skip': True, 'start': None,
            'reason': (f"first valid bar {first.date()} is after {start_ts.date()} and only "
                       f"{years:.2f} years to the last bar {last.date()} (< {min_years}) "
                       f"-> no TPI, asset skipped")}


def resolve_table(price_df, table=None, start='2018-01-01', ticker='', verbose=True) -> dict:
    """table None/'auto' -> select_table(); explicit 'main'/'alt' overrides it
    (the requested start is then used unchanged). Always logs the choice."""
    if table in (None, 'auto'):
        choice = {**select_table(price_df, start), 'source': 'auto'}
    else:
        check_table(table)
        choice = {'table': table, 'skip': False, 'start': start, 'source': 'explicit',
                  'reason': f"explicit table='{table}' (automatic rule not applied)",
                  'first_bar': None, 'last_bar': None, 'history_years': None}
    if verbose:
        s = choice['start']
        s = s.date() if isinstance(s, pd.Timestamp) else s
        print(f"[colour table] {ticker}: {choice['table'] or 'SKIP'} ({choice['source']}) - "
              f"{choice['reason']}; backtest start: {s}")
    return choice


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
         mode: str = 'long_short', table: str = None, components=None,
         n_trials: int = 5000, n_jobs: int = -1, seed: int = None,
         start: str = '2018-01-01', in_sample_end: str = None):
    """Optuna search -> all Pareto fronts (feasible trials only).
    `instrument` is the instrument TYPE ('crypto' or 'stocks'); the data is
    chosen by `ticker` (+ optional data_dir / file_path). Before, this ran
    'CDR.WA' labelled with whatever type was passed.
    table=None/'auto' picks the table from the history length (select_table);
    an asset with too little history returns an (empty) AssetSkipped."""

    strat_1 = optuna_testing.instrument_strategy(ticker, instrument, deposit, safe_investment,
                                                 data_dir=data_dir, file_path=file_path)
    strat_1.strategy_evaluation(table=table, mode=mode, components=components,
                                n_trials=n_trials, n_jobs=n_jobs, seed=seed,
                                start=start, in_sample_end=in_sample_end)
    if strat_1.skipped is not None:
        return strat_1.skipped

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


# Granice czerwonego zakresu (poza num_trades): metric -> (kierunek, próg)
#   'above': red when value > threshold, 'below': red when value < threshold
RED_THRESHOLDS = {'max_dd':         ('above', 0.40),
                  'sortino':        ('below', 2.0),
                  'sharpe':         ('below', 1.0),
                  'profit_factor':  ('below', 2.0),
                  'pct_profitable': ('below', 0.35),
                  'omega':          ('below', 1.1)}


def red_violations(metrics, table='main'):
    """Graded 'how red' of each of the 7 table metrics (0.0 = not red).

    > 0 exactly when classify_metric(...) == 'red'. The size is the relative
    distance past the red boundary (e.g. Sortino 1.5 -> (2 - 1.5) / 2 = 0.25),
    so an optimizer can tell 'almost yellow' from 'far off'. Non-finite values
    that classify as red get 1.0. Used as Optuna constraints."""
    check_table(table)
    out = {}
    for name in TABLE_METRICS:
        value = metrics[name]
        if classify_metric(name, value, table) != 'red':
            out[name] = 0.0
            continue
        if not np.isfinite(value):
            out[name] = 1.0
            continue
        if name == 'num_trades':
            red_below, _, green_to = TRADE_COUNT_BANDS[table]
            out[name] = ((red_below - value) / red_below if value < red_below
                         else (value - green_to) / green_to)
        else:
            direction, threshold = RED_THRESHOLDS[name]
            out[name] = ((value - threshold) if direction == 'above'
                         else (threshold - value)) / threshold
        out[name] = float(out[name])
    return out


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
            'close_max_dd':   close_max_dd,
            'total_return':   strat_metrics.total_return()}   # % (hold-out gate)


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
                              table: str = None, *,
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
    if nothing passes. `table` selects the Cobra table ('main' or 'alt');
    None/'auto' = select_table() by history length, which also sets the
    effective start for ALT assets. Too little history -> AssetSkipped.

    `instrument` is the instrument TYPE ('crypto'/'stocks'); the price data
    comes from `ticker` (+ data_dir / file_path). `mode` must match the mode
    the study was optimised with. `components` defaults to the components
    recorded on each trial (user attr), else all registered components.
    `in_sample_end` must match the study's: the test only uses bars up to it."""
    optuna_testing.check_mode(mode)
    optuna_testing.check_instrument_type(instrument)
    dataframe = data_import.data_importer(ticker, data_dir=data_dir, file_path=file_path)
    dataframe.import_csv_file()
    price = dataframe.df
    choice = resolve_table(price, table, start, ticker)
    if choice['skip']:
        return AssetSkipped(ticker, choice['reason'], choice)
    table, start = choice['table'], choice['start']

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
                   table: str = None, components=None, *, ticker: str,
                   data_dir: str = None, file_path: str = None, end: str = None,
                   verbose: bool = True, start: str = '2018-01-01'):
    """Table metrics and Cobra colours on the HELD-BACK period, i.e. the bars
    after `in_sample_end` (up to `end`, default: last bar).

    Indicators are computed on the full history up to `end` (the in-sample
    bars are the warm-up); nothing after `end` is used. Like every window in
    run_backtest, the hold-out starts flat on its first bar. Note: the trade
    count bands are meant for multi-year histories, so on a short hold-out
    the num_trades colour is mostly informative.
    table=None/'auto' uses the same history-length rule as the study
    (`start` is the study's backtest start, used only for that rule).
    Returns a dict (metrics None if the strategy was liquidated; skipped=True
    with a reason if the asset has too little history)."""
    optuna_testing.check_mode(mode)
    optuna_testing.check_instrument_type(instrument)
    components = tpi.resolve_components(components)
    dataframe = data_import.data_importer(ticker, data_dir=data_dir, file_path=file_path)
    dataframe.import_csv_file()
    price = dataframe.df
    choice = resolve_table(price, table, start, ticker, verbose)
    if choice['skip']:
        return {'ticker': ticker, 'skipped': True, 'reason': choice['reason'],
                'table': None, 'metrics': None, 'passed': False}
    table = choice['table']

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
    report = {'ticker': ticker, 'skipped': False, 'mode': mode, 'table': table,
              'table_reason': choice['reason'],
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
            print(f"  total return {metrics['total_return']:.3f}%, sortino {metrics['sortino']:.4f}")
            print(f"  calmar {metrics['calmar']:.4f}, alpha vs B&H {metrics['alpha']:.2f} pp "
                  f"(B&H return {benchmark['return']:.2f}%) -> "
                  f"{report['greens']} green / {report['reds']} red, "
                  f"{'PASS' if report['passed'] else 'FAIL'}")
    return report


# ---------- final acceptance gate (hold-out) ----------
# Decided with Krzyczek: a tuned TPI that has ALREADY passed the in-sample
# colour-table robustness test is ACCEPTED only if, on the hold-out,
#     total return > 0  AND  Sortino > 0
# (investment_metrics.sortino_ratio exactly as implemented, same risk-free
# rate). No comparison with buy-and-hold, no ratio to in-sample. If nothing
# is accepted the result is 'no accepted TPI': no substitute is ever chosen
# (no textbook params, no best-of-failing).

def accept_tpi(robustness_passed: bool, holdout: dict) -> dict:
    """Verdict for ONE candidate. `holdout` is a holdout_report() dict (or
    None when there is no hold-out). Returns verdict 'ACCEPTED'/'REJECTED',
    the reason, and the hold-out total return (%) and Sortino."""
    tr = so = None
    if holdout is not None and holdout.get('metrics') is not None:
        tr = holdout['metrics']['total_return']
        so = holdout['metrics']['sortino']
    reasons = []
    if not robustness_passed:
        reasons.append('failed the in-sample colour-table robustness test')
    if holdout is None:
        reasons.append('no hold-out period (in_sample_end not set)')
    elif holdout.get('skipped'):
        reasons.append(f"asset skipped: {holdout.get('reason')}")
    elif holdout.get('metrics') is None:
        reasons.append('liquidated on the hold-out')
    else:
        if not tr > 0:
            reasons.append(f'hold-out total return {tr:.3f}% <= 0')
        if not so > 0:          # NaN (no activity) also fails
            reasons.append(f'hold-out Sortino {so:.4f} <= 0')
    accepted = not reasons
    return {'verdict': 'ACCEPTED' if accepted else 'REJECTED', 'accepted': accepted,
            'reason': 'hold-out total return > 0 and Sortino > 0' if accepted
                      else '; '.join(reasons),
            'holdout_total_return': tr, 'holdout_sortino': so}


def final_acceptance(deposit: int, instrument, robust_candidates, in_sample_end: str,
                     safe_investment: float = 0.0, mode: str = 'long_short',
                     table: str = None, *, ticker: str, data_dir: str = None,
                     file_path: str = None, start: str = '2018-01-01') -> dict:
    """Final step after parameter_robustness_test: run the hold-out gate on
    every robust candidate. Returns {'accepted': [...], 'results': [...],
    'message': ...}. With nothing accepted it reports 'no accepted TPI' and
    returns an empty 'accepted' list - no fallback is chosen."""
    results, accepted = [], []
    for c in robust_candidates:
        ho = None
        if in_sample_end is not None:
            ho = holdout_report(deposit, instrument, c['params'], in_sample_end,
                                safe_investment, mode, table, c.get('components'),
                                ticker=ticker, data_dir=data_dir, file_path=file_path,
                                start=start)
        v = accept_tpi(True, ho)     # candidates here already passed robustness
        fmt = lambda x, f: 'n/a' if x is None else format(x, f)
        print(f"[acceptance] front {c.get('front')} candidate {c.get('candidate_idx')}: "
              f"hold-out total return {fmt(v['holdout_total_return'], '.3f')}%, "
              f"Sortino {fmt(v['holdout_sortino'], '.4f')} -> {v['verdict']} ({v['reason']})")
        results.append({**v, 'candidate': c, 'holdout': ho})
        if v['accepted']:
            accepted.append({**c, 'holdout': ho, 'verdict': v})
    if accepted:
        msg = f"{len(accepted)} accepted TPI candidate(s) for {ticker}"
    elif not robust_candidates:
        msg = (f"no accepted TPI for {ticker}: no candidate passed the in-sample "
               f"robustness test (no substitute chosen)")
    else:
        msg = (f"no accepted TPI for {ticker}: no robust candidate passed the hold-out "
               f"gate (no substitute chosen)")
    print(f"[acceptance] {msg}")
    return {'accepted': accepted, 'results': results, 'message': msg}


if __name__ == "__main__":
    # Ustawienia uruchomienia (CSV: <repo>/dane_cenowe/<TICKER>.csv lub $TPI_DATA_DIR)
    TICKER = 'BTC-USD'
    INSTRUMENT_TYPE = 'crypto'     # 'crypto' (365) albo 'stocks' (252)
    DEPOSIT = 12000
    RISK_FREE = 0
    MODE = 'long_short'
    TABLE = None                   # None = automatyczny wybór main/alt wg długości historii
    IN_SAMPLE_END = None           # np. '2025-03-31'; bez hold-outu bramka akceptacji odrzuca

    pareto = eval(DEPOSIT, INSTRUMENT_TYPE, RISK_FREE, ticker=TICKER, mode=MODE, table=TABLE,
                  in_sample_end=IN_SAMPLE_END)
    if isinstance(pareto, AssetSkipped):
        raise SystemExit(f"{TICKER}: skipped ({pareto.reason})")
    pareto = dedupe_fronts(pareto)
    robust_candidates = parameter_robustness_test(DEPOSIT, INSTRUMENT_TYPE, pareto, RISK_FREE,
                                                  mode=MODE, table=TABLE, ticker=TICKER,
                                                  in_sample_end=IN_SAMPLE_END)
    for c in robust_candidates:
        print(f"Front {c['front']} candidate {c['candidate_idx']}: "
              f"CoV={c['overall_cov']:.2%} ({c['cov_class']}), "
              f"params={c['params']}, metrics={c['base_metrics']}")
    # końcowa bramka: hold-out total return > 0 i Sortino > 0
    final = final_acceptance(DEPOSIT, INSTRUMENT_TYPE, robust_candidates, IN_SAMPLE_END,
                             RISK_FREE, MODE, TABLE, ticker=TICKER)