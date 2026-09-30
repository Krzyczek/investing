import pandas as pd
import evaluation_test   # evaluation.py jest przestarzały (deprecated)


def candidates_to_frame(candidates) -> pd.DataFrame:
    """Robust candidates from evaluation_test.parameter_robustness_test ->
    one row per candidate with its base metrics (the table selection() ranks)."""
    rows = []
    for c in candidates:
        m = c['base_metrics']
        rows.append({'front': c['front'], 'candidate_idx': c['candidate_idx'],
                     'overall_cov': c['overall_cov'], 'cov_class': c['cov_class'],
                     'sharpe': m['sharpe'], 'sortino': m['sortino'],
                     'omega': m['omega'], 'calmar': m['calmar'], 'alpha': m['alpha'],
                     'max_dd': m['max_dd'], 'profit_factor': m['profit_factor'],
                     'pct_profitable': m['pct_profitable'], 'num_trades': m['num_trades'],
                     'params': c['params']})
    return pd.DataFrame(rows)


def selected_test(pareto_fronts, instrument, safe_investment, deposit: int = 12000,
                  **robustness_kwargs) -> pd.DataFrame:
    """Replacement for the never-implemented evaluation.selected_test:
    robustness-test the Pareto fronts (evaluation_test pipeline) and return
    the passing candidates as a DataFrame. robustness_kwargs go to
    evaluation_test.parameter_robustness_test (ticker, mode, table, ...)."""
    candidates = evaluation_test.parameter_robustness_test(
        deposit, instrument, evaluation_test.dedupe_fronts(pareto_fronts),
        safe_investment, **robustness_kwargs)
    return candidates_to_frame(candidates)


def selection(ticker: str = None, instrument: str = None, safe_investment: float = None,
              deposit: int = 12000, data_dir: str = None, file_path: str = None,
              mode: str = 'long_short', table: str = 'main', components=None,
              start: str = '2018-01-01', in_sample_end: str = None, **optuna_kwargs):
    """Optuna -> robustness -> z-score ranking. Anything not passed is asked
    for interactively (as before). optuna_kwargs: n_trials, n_jobs, seed."""
    if ticker is None:
        ticker = input("Ticker (CSV name in the data folder, e.g. BTC-USD):").strip()
    if instrument is None:
        instrument = int(input("""Select the class of instrument:)
                           1. stocks
                           2. crypto"""))
        if instrument == 1:
            instrument = 'stocks'
        elif instrument == 2:
            instrument = 'crypto'

    if safe_investment is None:
        safe_investment = float(input("What is your yearly return for safe investment (e.g. bonds/savings):"))

    common = dict(ticker=ticker, data_dir=data_dir, file_path=file_path, mode=mode,
                  table=table, components=components, start=start,
                  in_sample_end=in_sample_end)
    eval = evaluation_test.eval(deposit, instrument, safe_investment, **common, **optuna_kwargs)
    selection = selected_test(eval, instrument, safe_investment, deposit, **common)
    if selection is None or selection.empty:
        print("No valid data found in the selection.")
        return None
    else:
        print(selection)

        metric_columns = ['sharpe','sortino','omega','calmar','alpha']
        selection = selection.drop_duplicates(subset=metric_columns).copy()
        df_zscores = (selection[metric_columns] - selection[metric_columns].mean()) / selection[metric_columns].std()
        selection['avg_zscore'] = df_zscores.mean(axis=1)
        selection_best = selection.sort_values(by='avg_zscore', ascending=False)

        print(selection_best)
        print(selection_best.iloc[0:6])
        return selection_best


if __name__ == "__main__":
    selection()
