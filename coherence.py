"""Koherencja czasowa komponentów TPI (time-coherence report).

Used to pick the 5-7 indicators for the final TPI: components that agree on
the trend direction and flip at about the same time as the group.

Given one price series, a set of components and their fitted params, the
report contains
    agreement   pairwise % of bars on which two components give the same signal
    correlation pairwise Pearson correlation of the +1/0/-1 signals
    flips       every flip (signal change) of every component, with its date
    consensus_flips  the MAJOR flips of the group consensus
    lags        for each major consensus flip, how many bars each component
                flipped into the new direction after (+ = lags) or before
                (- = leads) it; NaN = no such flip within +-max_lag bars
    summary     one row per component, ranked by coherence_score

Consensus = sign of the TPI score of the selected components (the TPI's own
group averaging; weighting='equal' uses the plain mean of the signals). A
score of exactly 0 keeps the previous consensus. A consensus flip is MAJOR
when the new direction then holds for at least `min_hold` bars.

coherence_score (0..1, higher = more coherent) is the mean of
    agree_loo        share of bars where the component agrees with the
                     consensus of the OTHER components (leave-one-out, so a
                     component is not rewarded for agreeing with itself)
    flip_hit_rate    share of major consensus flips matched within +-max_lag
    flip_precision   share of the component's own flips into +1/-1 that fall
                     within +-max_lag of a major consensus flip in the same
                     direction (penalises noisy components, which would
                     otherwise "hit" every consensus flip by flipping often)
    timing           1 - median(|lag|) / max_lag  (0 if nothing matched)
Note: the major flips and hit rate use the full-group consensus, which
includes the component itself; with few components this favours all of them.

Example:
    import data_import, coherence
    d = data_import.data_importer('BTC-USD', data_dir='/path/to/csvs'); d.import_csv_file()
    rep = coherence.coherence_report(d.df, coherence.TEXTBOOK_PARAMS, csv_path='coherence.csv')
    print(rep.summary)
"""
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

import tpi

# Parametry "podręcznikowe" 5 komponentów (punkt odniesienia, nie wynik optymalizacji)
TEXTBOOK_PARAMS = {
    'fast_ma': 12, 'slow_ma': 26,
    'parabolic_sar_start': 0.02, 'parabolic_sar_acceleration': 0.02,
    'parabolic_sar_maximum': 0.2,
    'adx_period': 14, 'threshold': 25,
    'aroon_length': 14,
    'supertrend_atr_period': 10, 'supertrend_factor': 3.0,
}


@dataclass
class CoherenceReport:
    summary: pd.DataFrame
    agreement: pd.DataFrame
    correlation: pd.DataFrame
    flips: pd.DataFrame
    consensus_flips: pd.DataFrame
    lags: pd.DataFrame
    signals: pd.DataFrame

    def to_csv(self, path: str):
        """Write summary to `path` and the other tables next to it
        (<stem>_agreement.csv, _correlation.csv, _flips.csv,
        _consensus_flips.csv, _lags.csv). Returns the list of files."""
        stem, ext = os.path.splitext(path)
        ext = ext or '.csv'
        files = {path if path.endswith(ext) else stem + ext: self.summary,
                 f'{stem}_agreement{ext}': self.agreement,
                 f'{stem}_correlation{ext}': self.correlation,
                 f'{stem}_flips{ext}': self.flips,
                 f'{stem}_consensus_flips{ext}': self.consensus_flips,
                 f'{stem}_lags{ext}': self.lags}
        for f, df in files.items():
            df.to_csv(f)
        return list(files)


def _score(price, params, components, weighting):
    if weighting == 'tpi':
        t = tpi.tpi(price)
        t.calculate_tpi(params, 'long_short', components)
        return pd.Series(t.tpi_score, index=price.index)
    if weighting == 'equal':
        sigs = [np.asarray(tpi.COMPONENTS[c]['signal'](price, params), dtype=float)
                for c in components]
        return pd.Series(np.mean(np.vstack(sigs), axis=0), index=price.index)
    raise ValueError("weighting must be 'tpi' or 'equal'")


def consensus_state(score: pd.Series) -> pd.Series:
    """+1 / -1 by the sign of the score; 0 keeps the previous state."""
    state = pd.Series(np.sign(score.to_numpy(dtype=float)), index=score.index)
    return state.replace(0, np.nan).ffill().fillna(0)


def major_flips(state: pd.Series, min_hold: int = 20) -> pd.DataFrame:
    """Major consensus flips: the consensus switches to the other direction
    and holds it for at least min_hold bars. Runs shorter than min_hold are
    treated as noise (the previous confirmed direction stays), so a short
    dip and the return from it are NOT flips. The flip is dated at the first
    bar of the confirming run."""
    s = state.to_numpy()
    rows = []
    confirmed = None
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        length, val = j - i + 1, s[i]
        if val != 0 and length >= min_hold:
            if confirmed is not None and val != confirmed:
                rows.append({'date': state.index[i], 'bar': i, 'direction': int(val)})
            confirmed = val
        i = j + 1
    return pd.DataFrame(rows, columns=['date', 'bar', 'direction'])


def flip_table(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in signals.columns:
        s = signals[c]
        changed = s.ne(s.shift()) & s.shift().notna()
        for d in s.index[changed]:
            i = s.index.get_loc(d)
            rows.append({'component': c, 'date': d,
                         'from': int(s.iloc[i - 1]), 'to': int(s.iloc[i])})
    return pd.DataFrame(rows, columns=['component', 'date', 'from', 'to'])


def _lag(values: np.ndarray, bar: int, direction: int, max_lag: int) -> float:
    """Signed distance (bars) from `bar` to the component's nearest flip INTO
    `direction` within +-max_lag; ties go to the earlier (leading) flip."""
    best = np.nan
    lo, hi = max(1, bar - max_lag), min(len(values) - 1, bar + max_lag)
    for k in range(lo, hi + 1):
        if values[k] == direction and values[k - 1] != direction:
            d = k - bar
            if np.isnan(best) or abs(d) < abs(best):
                best = d
    return best


def coherence_report(price: pd.DataFrame, params: dict, components=None,
                     start: str = '2018-01-01', end: str = None,
                     min_hold: int = 20, max_lag: int = 10,
                     weighting: str = 'tpi', csv_path: str = None) -> CoherenceReport:
    """Coherence of `components` (default: all registered) with fitted
    `params` on one price series. Signals are computed on the history up to
    `end` (earlier bars = warm-up) and analysed on [start, end]."""
    components = tpi.resolve_components(components)
    if len(components) < 2:
        raise ValueError("Coherence needs at least 2 components.")
    df = price.loc[:end] if end is not None else price
    window = df.index >= pd.Timestamp(start) if start is not None else np.ones(len(df), bool)

    signals = pd.DataFrame(
        {c: np.asarray(tpi.COMPONENTS[c]['signal'](df, params), dtype=float) for c in components},
        index=df.index).loc[window]
    state = consensus_state(_score(df, params, components, weighting).loc[window])

    # pairwise agreement % and signal correlation
    agreement = pd.DataFrame(index=components, columns=components, dtype=float)
    for a in components:
        for b in components:
            agreement.loc[a, b] = 100.0 * (signals[a] == signals[b]).mean()
    correlation = signals.corr()

    flips = flip_table(signals)
    cflips = major_flips(state, min_hold)

    lag_rows = []
    values = {c: signals[c].to_numpy() for c in components}
    for _, f in cflips.iterrows():
        row = {'date': f['date'], 'direction': f['direction']}
        for c in components:
            row[c] = _lag(values[c], int(f['bar']), int(f['direction']), max_lag)
        lag_rows.append(row)
    lags = pd.DataFrame(lag_rows, columns=['date', 'direction'] + components)

    in_market = state != 0
    summary_rows = []
    for c in components:
        others = [o for o in components if o != c]
        loo_state = consensus_state(_score(df, params, others, weighting).loc[window])
        valid_loo = loo_state != 0
        lc = lags[c].dropna() if len(lags) else pd.Series(dtype=float)
        hit = len(lc) / len(lags) if len(lags) else np.nan
        med_abs = float(lc.abs().median()) if len(lc) else np.nan
        timing = 0.0 if np.isnan(med_abs) else max(0.0, 1 - med_abs / max_lag)
        agree_loo = float((signals[c][valid_loo] == loo_state[valid_loo]).mean())
        # own flips into +1/-1 that sit near a major consensus flip of the same direction
        v = values[c]
        own = [(k, int(v[k])) for k in range(1, len(v)) if v[k] != v[k - 1] and v[k] != 0]
        cf = list(zip(cflips['bar'].astype(int), cflips['direction'].astype(int)))
        matched = sum(1 for k, d in own
                      if any(dd == d and abs(k - b) <= max_lag for b, dd in cf))
        precision = matched / len(own) if own else np.nan
        summary_rows.append({
            'component': c,
            'n_flips': int((flips['component'] == c).sum()),
            'mean_pairwise_agreement_pct': float(agreement.loc[c, others].mean()),
            'mean_pairwise_corr': float(correlation.loc[c, others].mean()),
            'agree_consensus_pct': 100.0 * float((signals[c][in_market] == state[in_market]).mean()),
            'agree_loo_pct': 100.0 * agree_loo,
            'flip_hit_rate': hit,
            'flip_precision': precision,
            'median_lag': float(lc.median()) if len(lc) else np.nan,
            'median_abs_lag': med_abs,
            'mean_lag': float(lc.mean()) if len(lc) else np.nan,
            'coherence_score': float(np.mean([agree_loo,
                                              0.0 if np.isnan(hit) else hit,
                                              0.0 if np.isnan(precision) else precision,
                                              timing])),
        })
    summary = (pd.DataFrame(summary_rows).sort_values('coherence_score', ascending=False)
               .reset_index(drop=True))
    summary.index = summary.index + 1
    summary.index.name = 'rank'

    report = CoherenceReport(summary=summary, agreement=agreement.round(2),
                             correlation=correlation.round(4), flips=flips,
                             consensus_flips=cflips, lags=lags, signals=signals)
    if csv_path:
        report.to_csv(csv_path)
    return report


if __name__ == "__main__":
    import argparse
    import data_import
    ap = argparse.ArgumentParser(description="TPI component coherence report (textbook params by default)")
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--data-dir')
    ap.add_argument('--file-path')
    ap.add_argument('--start', default='2018-01-01')
    ap.add_argument('--end')
    ap.add_argument('--components', nargs='*')
    ap.add_argument('--min-hold', type=int, default=20)
    ap.add_argument('--max-lag', type=int, default=10)
    ap.add_argument('--csv')
    a = ap.parse_args()
    d = data_import.data_importer(a.ticker, data_dir=a.data_dir, file_path=a.file_path)
    d.import_csv_file()
    rep = coherence_report(d.df, TEXTBOOK_PARAMS, a.components, a.start, a.end,
                           a.min_hold, a.max_lag, csv_path=a.csv)
    pd.set_option('display.width', 200)
    print(rep.summary.to_string())
    print("\nPairwise agreement %:\n", rep.agreement.to_string())
    print("\nSignal correlation:\n", rep.correlation.to_string())
    print(f"\n{len(rep.consensus_flips)} major consensus flips; lags (bars, + = after):")
    print(rep.lags.to_string())
