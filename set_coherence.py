"""Set-level time coherence (v2.1: v2 approved by Krzyczek 30 Sep 2026, plus
the v2.1 fixes of 30 Sep: median-agreement clusters, leave-one-out ties hold).

Scores whole candidate TPI sets (5-7 components) on how coherently their
members flip with the set's own consensus, and ranks all sets of a pool over
a grid of trade-count targets and hold lengths. Works on whichever
components are registered in tpi.COMPONENTS (it does not import any
indicator itself); the pool is just a list of registered names.

Horizon matching (per target, pool-wide): every member's primary speed knob
(KNOB, or the `knobs` argument) is scanned over its registered search space
(or a `space_override`) with all other params at textbook (TEXTBOOK_PARAMS,
or the middle of the range when a component has no textbook value). The
value is picked from the longest plateau of values whose in-sample trade
count is within +-tol of the target (nearest-to-target, then nearest to
textbook); if no value gets within +-tol, the closest reachable value is used
and the member is flagged 'NOT horizon-matched' (it still competes). A knob
can be a pair of params (MACD: fast and slow, native values, no scaling);
then the whole valid 2-D grid is scanned and the valid cell closest to the
target is picked (tie -> nearest to textbook in grid steps); its 'plateau' is
the number of its 4 grid neighbours that are also within +-tol.
Trades are counted as run_backtest counts them (v2.1): evaluation_test.
count_trades on the hold-rule position lagged by one bar inside the window.

Consensus (as the TPI builds it, equal weights): the mean of the members' RAW
signals (+1/0/-1), turned into a position with the framework hold rule
(tpi.position_from_score: a score of 0 holds the previous position, flat
until the first non-zero score). A MAJOR FLIP is a sign change of that
position confirmed by a run of >= `hold` bars (shorter runs are noise).
Note: the TPI itself averages inside groups first (tpi.calculate_tpi); the
coherence consensus uses the plain equal-weight mean.

Leave-one-out: member i is scored against the consensus of the OTHER
members (mean of their raw signals). A leave-one-out score of exactly 0
HOLDS the previous leave-one-out position (the same zero-score hold rule as
the TPI; if the first bars are 0 it stays flat until the first non-zero
score). v2.1: ties are no longer broken with the full-set score (that leaked
member i's own vote into its own consensus).

Member flips for recall/precision: the DEBOUNCED raw signal (raw -> hold
rule -> the same `hold`-bar confirmation as the consensus).

Window w (pool-wide, from the target, never from the set):
    w = max(5, round(0.25 * n_bars / target)).
recall    = share of major flips matched by a same-direction member flip within +-w
precision = share of the member's debounced flips within +-w of a same-direction major flip
F1        = harmonic mean;  timing = clip(1 - median|lag|/w, 0, 1), 0 with no matches
A         = mean pairwise sign agreement of the members' hold-rule positions
C         = 0.4 F1 + 0.3 timing + 0.3 A                       (no CV term)
Rules     : precision veto (any member precision < precision_veto, 0.3);
            cluster rule (at most one member per connected cluster of
            pairwise agreement > cluster_threshold, 0.9). v2.1: the agreement
            used is each pair's MEDIAN agreement across the targets (each at
            that target's matched params), computed once per pool, so the same
            clusters apply to all 9 configs. Pairs with median agreement in
            (0.85, 0.9] are reported, not enforced.
Ranking   : every set is ranked by C in each (target, hold) config (default
            60/70/80 x 15/20/30); a set excluded in a config gets rank
            n_ranked + 1 there. Final order (v2.1): median rank, then C at
            70/20 (higher first), then spread (max - min), then IQR.

In-sample only: pass the price series cut at the in-sample end; bars before
`start` are warm-up.

Example:
    import data_import, set_coherence as sc
    d = data_import.data_importer('BTC-USD', data_dir='/path'); d.import_csv_file()
    price = d.df.loc[:'2025-03-31']
    pool = ['ema_cross', 'parabolic_sar', 'supertrend', 'adx', 'aroon']
    agg, scores, tables, params = sc.rank_pool(price, pool, sizes=(5,))
"""
import itertools

import numpy as np
import pandas as pd

import evaluation_test as et
import tpi
from coherence import TEXTBOOK_PARAMS

TARGETS = (60, 70, 80)
HOLDS = (15, 20, 30)
REF = (70, 20)
WEIGHTS = (0.4, 0.3, 0.3)          # F1, timing, A
PRECISION_VETO = 0.3
CLUSTER_THRESHOLD = 0.9
WATCH_BAND = (0.85, 0.9)
PLATEAU_TOL = 10
ATOL = tpi.ZERO_ATOL
SET1_BASELINE = ('ema_cross', 'parabolic_sar', 'supertrend', 'adx', 'aroon')

# primary speed knob per component (names only; components that are not
# registered are simply never used). No secondary-knob / scaling fallback.
KNOB = {
    'ema_cross': 'slow_ma', 'parabolic_sar': 'parabolic_sar_acceleration',
    'supertrend': 'supertrend_factor', 'adx': 'adx_period', 'aroon': 'aroon_length',
    'donchian': 'donchian_entry_length', 'keltner': 'kc_multiplier', 'rsi50': 'rsi_length',
    'roc': 'roc_length', 'hull': 'hull_length', 'tema': 'tema_length',
    'vortex': 'vortex_length', 'bollinger': 'bollinger_mult', 'macd': ('macd_fast', 'macd_slow'),
    'cci': 'cci_length', 'kama': 'kama_fast',
    # 1 Oct 2026: the remaining POOL-READY members (previously passed ad hoc as `knobs`
    # in the T1/T2 runs). regime_gate is matched natively on its EMA pair like MACD; its
    # Hurst/ADF params stay at textbook (tpi caches those series per price content, so the
    # 2-D scan costs one Hurst + one ADF pass, then ~1 ms per cell).
    'obv': 'obv_ema_length', 'linreg': 'linreg_length', 'ehlers_itrend': 'itrend_alpha',
    'ichimoku': 'ichimoku_base', 'regime_gate': ('regime_fast', 'regime_slow'),
}


# ---------------- horizon matching ----------------
def grid(spec):
    kind, lo, hi, step = spec
    n = int(round((hi - lo) / step))
    vals = [lo + i * step for i in range(n + 1)]
    return [int(v) for v in vals] if kind == 'int' else [round(v, 6) for v in vals]


def textbook(comp, base=None):
    """Textbook params of a registered component (middle of the range when
    there is no textbook value)."""
    base = TEXTBOOK_PARAMS if base is None else base
    out = {}
    for k, spec in tpi.COMPONENTS[comp]['params'].items():
        if k in base:
            out[k] = base[k]
        else:
            g = grid(spec)
            out[k] = g[len(g) // 2]
    return out


def member_arrays(price, comp, params, start='2018-01-01'):
    """Raw signal and hold-rule position on bars >= start (the signal is
    computed on the whole series; earlier bars are warm-up)."""
    raw = np.nan_to_num(np.asarray(tpi.COMPONENTS[comp]['signal'](price, params), dtype=float), nan=0.0)
    pos = tpi.position_from_score(raw, 'long_short')
    keep = price.index >= pd.Timestamp(start)
    return raw[keep], pos[keep]


def trades(pos):
    """Trades of a window position as run_backtest counts them (one-bar lag:
    the first bar of the window is flat, the last bar's new position is not
    traded)."""
    pos = np.asarray(pos)
    return int(et.count_trades(np.r_[0, pos[:-1]])) if len(pos) else 0


def pick(values, counts, textbook_value, target, tol=PLATEAU_TOL):
    """Index into `values`: longest plateau of |count - target| <= tol
    (tie -> plateau nearest to textbook), inside it the value closest to the
    target. Returns (index, matched, plateau index range or None)."""
    counts = np.asarray(counts, dtype=float)
    ok = np.abs(counts - target) <= tol
    tb = int(np.argmin([abs(float(v) - float(textbook_value)) for v in values]))
    if ok.any():
        runs, i = [], 0
        while i < len(values):
            if ok[i]:
                j = i
                while j + 1 < len(values) and ok[j + 1]:
                    j += 1
                runs.append((i, j))
                i = j + 1
            else:
                i += 1
        best = max(j - i for i, j in runs)
        i, j = min([r for r in runs if r[1] - r[0] == best],
                   key=lambda r: min(abs(k - tb) for k in range(r[0], r[1] + 1)))
        k = min({(i + j) // 2, (i + j + 1) // 2}, key=lambda m: (abs(counts[m] - target), abs(m - tb)))
        return k, True, (i, j)
    d = np.abs(counts - target)
    k = int(min(np.flatnonzero(d == d.min()), key=lambda m: abs(m - tb)))
    return k, False, None


def _counts(price, c, tb, keys, specs, start, cache):
    """Valid grid cells and their trade counts (cached across targets)."""
    grids = [grid(sp) for sp in specs]
    cells, counts = [], []
    for vals in itertools.product(*grids):
        p = {**tb, **dict(zip(keys, vals))}
        if not tpi.params_valid(p, [c]):
            continue
        ck = (c, start, str(price.index[-1]), len(price), tuple(sorted(p.items())))
        if ck not in cache:
            cache[ck] = trades(member_arrays(price, c, p, start)[1])
        cells.append(vals)
        counts.append(cache[ck])
    return grids, cells, counts


def pick2d(grids, cells, counts, tb_vals, target, tol=PLATEAU_TOL):
    """Closest-to-target valid cell of a 2-D grid (tie -> nearest to textbook
    in grid steps). Returns (index, matched, n of 4 neighbours within tol)."""
    counts = np.asarray(counts, dtype=float)
    pos = [{v: i for i, v in enumerate(g)} for g in grids]
    tbi = [int(np.argmin([abs(float(v) - float(t)) for v in g])) for g, t in zip(grids, tb_vals)]
    ij = np.array([[pos[0][a], pos[1][b]] for a, b in cells])
    d = np.abs(counts - target)
    best = np.flatnonzero(d == d.min())
    k = int(min(best, key=lambda m: abs(ij[m, 0] - tbi[0]) + abs(ij[m, 1] - tbi[1])))
    ok = d <= tol
    lookup = {tuple(x): o for x, o in zip(ij, ok)}
    nb = sum(lookup.get((ij[k, 0] + a, ij[k, 1] + b), False) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    return k, bool(ok[k]), int(nb)


def horizon_match(price, comps, target, start='2018-01-01', tol=PLATEAU_TOL, knobs=None,
                  space_override=None, base=None, cache=None):
    """Match each component's primary knob to `target` trades. Returns
    (table, params) with params[comp] = full param dict. Pass the same
    `cache` dict for several targets to count each grid cell once."""
    knobs = {**KNOB, **(knobs or {})}
    space_override = space_override or {}
    cache = {} if cache is None else cache
    rows, params = [], {}
    no_knob = [c for c in comps if c not in knobs]
    if no_knob:
        raise ValueError(f'no speed knob for {no_knob}: add them to KNOB (or pass knobs=...); '
                         f'a pool member must never be dropped silently')
    for c in comps:
        tb = textbook(c, base)
        key = knobs[c]
        keys = list(key) if isinstance(key, (tuple, list)) else [key]
        specs = [space_override.get(k, tpi.COMPONENTS[c]['params'][k]) for k in keys]
        grids, cells, counts = _counts(price, c, tb, keys, specs, start, cache)
        if len(keys) == 1:
            vals = [x[0] for x in cells]
            k, ok, plateau = pick(vals, counts, tb[keys[0]], target, tol)
            plateau = None if plateau is None else (vals[plateau[0]], vals[plateau[1]])
        else:
            k, ok, plateau = pick2d(grids, cells, counts, [tb[x] for x in keys], target, tol)
            plateau = f'{plateau}/4 neighbours within tol'
        chosen = dict(zip(keys, cells[k]))
        params[c] = {**tb, **chosen}
        rows.append(dict(target=target, component=c, param='/'.join(keys),
                         value='/'.join(str(v) for v in cells[k]), trades=counts[k],
                         matched=ok, flag='' if ok else 'NOT horizon-matched', plateau=plateau,
                         scan_min=min(counts), scan_max=max(counts),
                         search_space=specs if len(specs) > 1 else specs[0],
                         space_overridden=any(x in space_override for x in keys),
                         held_fixed={kk: vv for kk, vv in tb.items() if kk not in keys}))
    return pd.DataFrame(rows), params


# ---------------- events and matching ----------------
def window_w(target, n_bars):
    return max(5, int(round(0.25 * n_bars / target)))


def hold_positions(score):
    """Row-wise tpi.position_from_score for a (k, T) score matrix."""
    s = np.where(np.abs(score) <= ATOL, 0.0, np.sign(score))
    T = s.shape[-1]
    idx = np.maximum.accumulate(np.where(s != 0, np.arange(T), -1), axis=-1)
    out = np.take_along_axis(s, np.clip(idx, 0, None), axis=-1)
    return np.where(idx >= 0, out, 0.0)


def major_flip_events(state, hold):
    """Confirmed runs (non-zero, >= hold bars) whose direction differs from
    the previous confirmed run; dated at the run's first bar."""
    s = np.asarray(state)
    if len(s) == 0:
        return np.array([], int), np.array([], int)
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    lengths = np.diff(np.r_[starts, len(s)])
    vals = s[starts]
    keep = (vals != 0) & (lengths >= hold)
    cs, cv = starts[keep], vals[keep]
    if len(cv) < 2:
        return np.array([], int), np.array([], int)
    flip = np.r_[False, cv[1:] != cv[:-1]]
    return cs[flip], cv[flip]


def match(member_flips, major, w):
    """recall, precision, F1, timing, median signed lag, n matched."""
    fb, fd = member_flips
    mb, md = major
    if len(mb) == 0 or len(fb) == 0:
        return 0.0, 0.0, 0.0, 0.0, np.nan, 0
    lags = np.full(len(mb), np.nan)
    for d in (1, -1):
        mf, sel = fb[fd == d], md == d
        if len(mf) == 0 or not sel.any():
            continue
        b = mb[sel]
        j = np.searchsorted(mf, b)
        dl = mf[np.clip(j - 1, 0, len(mf) - 1)] - b
        dr = mf[np.clip(j, 0, len(mf) - 1)] - b
        lags[sel] = np.where(np.abs(dl) <= np.abs(dr), dl, dr)     # tie -> leading flip
    matched = np.abs(lags) <= w
    hit = np.zeros(len(fb), bool)
    for d in (1, -1):
        b, sel = mb[md == d], fd == d
        if len(b) == 0 or not sel.any():
            continue
        f = fb[sel]
        j = np.searchsorted(b, f)
        near = np.minimum(np.abs(f - b[np.clip(j - 1, 0, len(b) - 1)]),
                          np.abs(f - b[np.clip(j, 0, len(b) - 1)]))
        hit[sel] = near <= w
    recall, precision = matched.mean(), hit.mean()
    f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
    if matched.any():
        timing = float(np.clip(1 - np.median(np.abs(lags[matched])) / w, 0, 1))
        signed = float(np.median(lags[matched]))
    else:
        timing, signed = 0.0, np.nan
    return float(recall), float(precision), float(f1), timing, signed, int(matched.sum())


def clusters(agree, names, thr=CLUSTER_THRESHOLD):
    """Connected components of the graph 'pairwise agreement > thr'."""
    lab = list(range(len(names)))

    def find(i):
        while lab[i] != i:
            lab[i] = lab[lab[i]]
            i = lab[i]
        return i
    for a, b in itertools.combinations(range(len(names)), 2):
        if agree[a, b] > thr:
            lab[find(a)] = find(b)
    comp = np.array([find(i) for i in range(len(names))])
    groups = [tuple(names[i] for i in np.flatnonzero(comp == r)) for r in sorted(set(comp))]
    return comp, [g for g in groups if len(g) > 1]


class Pool:
    """Per-target precompute: raw signals, positions, agreement, clusters,
    per-hold debounced member flips."""

    def __init__(self, names, raw, target, holds=HOLDS, cluster_threshold=CLUSTER_THRESHOLD):
        self.names, self.target = list(names), target
        self.raw = np.asarray(raw, dtype=float)
        self.pos = np.vstack([tpi.position_from_score(r, 'long_short') for r in self.raw]).astype(float)
        self.T = self.raw.shape[1]
        self.w = window_w(target, self.T)
        both = (self.pos[:, None] != 0) & (self.pos[None] != 0)
        self.agree = ((self.pos[:, None] == self.pos[None]) & both).sum(-1) / np.maximum(both.sum(-1), 1)
        # per-target clusters until apply_clusters() sets the pool-wide ones
        self.apply_clusters(self.agree, cluster_threshold)
        self.flips = {h: [major_flip_events(p, h) for p in self.pos] for h in holds}
        self.n_trades = np.array([trades(p) for p in self.pos.astype(int)])

    def apply_clusters(self, agree, cluster_threshold=CLUSTER_THRESHOLD):
        """Set the cluster rule from an agreement matrix (v2.1: the
        pool-wide median across targets, see median_agreement)."""
        self.cluster_agree = np.asarray(agree, dtype=float)
        self.comp, self.cluster_groups = clusters(self.cluster_agree, self.names, cluster_threshold)
        self.watch = [(a, b, round(float(self.cluster_agree[i, j]), 3))
                      for (i, a), (j, b) in itertools.combinations(enumerate(self.names), 2)
                      if WATCH_BAND[0] < self.cluster_agree[i, j] <= WATCH_BAND[1]]

    @classmethod
    def from_price(cls, price, params, names, target, start='2018-01-01', **kw):
        raw = [member_arrays(price, c, params[c], start)[0] for c in names]
        return cls(names, np.vstack(raw), target, **kw)


def median_agreement(pools):
    """Element-wise median of the pairwise agreement matrices of the
    per-target pools (same member order)."""
    names = pools[0].names
    assert all(p.names == names for p in pools), 'pools must have the same members in the same order'
    return np.median(np.stack([p.agree for p in pools]), axis=0)


def apply_pool_clusters(pools, cluster_threshold=CLUSTER_THRESHOLD):
    """v2.1 cluster rule: one set of clusters for the whole pool, from the
    median agreement across targets, applied to every per-target pool."""
    med = median_agreement(pools)
    for p in pools:
        p.apply_clusters(med, cluster_threshold)
    return med, pools[0].cluster_groups


def loo_positions(R):
    """Leave-one-out consensus positions for a (k, T) raw-signal matrix:
    mean of the other k-1 members, then the hold rule (0 holds the previous
    position; flat until the first non-zero score)."""
    R = np.asarray(R, dtype=float)
    k = len(R)
    return hold_positions((R.sum(0)[None] - R) / (k - 1))


def set_coherence(P, members, holds=HOLDS, weights=WEIGHTS, precision_veto=PRECISION_VETO,
                  detail=False):
    """Score one set (names or indices into P.names) for every hold.
    Returns {hold: dict(C, mean_f1, mean_timing, A, min_precision, weakest,
    precision_veto, cluster_rule, excluded[, per_member])}."""
    idx = np.array(sorted(P.names.index(m) if isinstance(m, str) else int(m) for m in members))
    k = len(idx)
    pos = loo_positions(P.raw[idx])
    A = float(np.mean([P.agree[a, b] for a, b in itertools.combinations(idx, 2)]))
    cluster_hit = len(set(P.comp[idx])) < k
    out = {}
    for h in holds:
        res = [match(P.flips[h][i], major_flip_events(pos[r], h), P.w) for r, i in enumerate(idx)]
        rec, prec, f1, tim = np.array([x[:4] for x in res]).T
        o = dict(C=float(weights[0] * f1.mean() + weights[1] * tim.mean() + weights[2] * A),
                 mean_f1=float(f1.mean()), mean_timing=float(tim.mean()), A=A,
                 min_precision=float(prec.min()), weakest=P.names[idx[int(np.argmin(prec))]],
                 precision_veto=bool(prec.min() < precision_veto), cluster_rule=bool(cluster_hit))
        o['excluded'] = o['precision_veto'] or o['cluster_rule']
        if detail:
            o['per_member'] = pd.DataFrame(
                [dict(member=P.names[i], recall=x[0], precision=x[1], f1=x[2], timing=x[3],
                      median_signed_lag=x[4], matched=x[5], debounced_flips=len(P.flips[h][i][0]),
                      trades=int(P.n_trades[i])) for i, x in zip(idx, res)])
        out[h] = o
    return out


def _score_combos(P, k, lo, hi, holds, kw):
    rows = []
    for combo in itertools.islice(itertools.combinations(range(len(P.names)), k), lo, hi):
        key = '+'.join(P.names[i] for i in combo)
        for h, o in set_coherence(P, combo, holds=holds, **kw).items():
            rows.append(dict(members=key, k=k, target=P.target, hold=h, **o))
    return rows


_WORK = None        # (pools, holds, kw) inherited by forked score workers


def _score_task(task):
    pi, k, lo, hi = task
    pools, holds, kw = _WORK
    return _score_combos(pools[pi], k, lo, hi, holds, kw)


def _score_pools(pools, sizes, holds, n_jobs=1, chunk=2000, **kw):
    """Rows of every (pool, k, combo, hold), in exactly the serial order. n_jobs > 1
    scores contiguous chunks of combinations in forked worker processes and
    concatenates them in task order, so the result is identical to n_jobs=1
    (the scoring is CPU-bound pure Python; this is the slow part of rank_pool)."""
    import math
    import multiprocessing as mp
    tasks = [(pi, k, lo, min(lo + chunk, math.comb(len(P.names), k)))
             for pi, P in enumerate(pools) for k in sizes
             for lo in range(0, math.comb(len(P.names), k), chunk)]
    if n_jobs is None or n_jobs < 1:
        import os
        n_jobs = os.cpu_count() or 1
    if n_jobs == 1 or len(tasks) <= 1 or 'fork' not in mp.get_all_start_methods():
        parts = [_score_combos(pools[pi], k, lo, hi, holds, kw) for pi, k, lo, hi in tasks]
    else:
        global _WORK
        _WORK = (pools, holds, kw)
        try:
            with mp.get_context('fork').Pool(min(n_jobs, len(tasks))) as ex:
                parts = ex.map(_score_task, tasks, chunksize=1)      # map keeps task order
        finally:
            _WORK = None
    per_pool = [[] for _ in pools]
    for (pi, *_), part in zip(tasks, parts):
        per_pool[pi].extend(part)
    return [pd.DataFrame(r) for r in per_pool]


def score_all(P, sizes=(5, 6, 7), holds=HOLDS, n_jobs=1, **kw):
    """Score every set of the given sizes in one per-target pool (n_jobs: worker
    processes, 1 = serial, None/0 = all CPUs; the result does not depend on it)."""
    return _score_pools([P], sizes, holds, n_jobs=n_jobs, **kw)[0]


def rank_configs(scores):
    """Rank by C inside each (target, hold); excluded sets get n_ranked + 1."""
    scores = scores.copy()
    scores['rank'] = np.nan
    for _, g in scores.groupby(['target', 'hold']):
        ok = g[~g.excluded].sort_values('C', ascending=False)
        scores.loc[ok.index, 'rank'] = np.arange(1, len(ok) + 1)
        scores.loc[g.index[g.excluded.values], 'rank'] = len(ok) + 1
    return scores


def differs_enough(candidate, earlier_sets=(SET1_BASELINE,), min_diff=3):
    """True if `candidate` has >= min_diff members outside each earlier set."""
    cand = set(candidate.split('+') if isinstance(candidate, str) else candidate)
    return all(len(cand - set(e.split('+') if isinstance(e, str) else e)) >= min_diff
               for e in earlier_sets)


def aggregate(ranked, ref=REF, earlier_sets=(SET1_BASELINE,)):
    """One row per set, ordered by median rank, then C at ref (higher
    first), then spread, then IQR.
    spread_passed / n_configs_passed describe the configs the set passed."""
    g = ranked.groupby('members')['rank']
    agg = pd.DataFrame({'median_rank': g.median(), 'min_rank': g.min(), 'max_rank': g.max(),
                        'iqr': g.quantile(0.75) - g.quantile(0.25)})
    passed = ranked[~ranked.excluded].groupby('members')['rank']
    agg['n_configs_passed'] = passed.size().reindex(agg.index).fillna(0).astype(int)
    agg['spread_passed'] = (passed.max() - passed.min()).reindex(agg.index)
    agg['spread'] = agg.max_rank - agg.min_rank
    r = ranked[(ranked.target == ref[0]) & (ranked.hold == ref[1])].set_index('members')
    tag = f'@{ref[0]}/{ref[1]}'
    for c in ('C', 'mean_f1', 'mean_timing', 'A', 'min_precision', 'weakest', 'precision_veto',
              'cluster_rule', 'rank'):
        agg[c + tag] = r[c]
    agg['k'] = r['k']
    agg = agg.reset_index().sort_values(['median_rank', 'C' + tag, 'spread', 'iqr'],
                                        ascending=[True, False, True, True],
                                        na_position='last').reset_index(drop=True)
    agg['final_position'] = np.arange(1, len(agg) + 1)
    agg['differs_from_set1'] = agg.members.apply(lambda m: differs_enough(m, earlier_sets))
    return agg


def rank_pool(price, pool, targets=TARGETS, holds=HOLDS, sizes=(5, 6, 7), start='2018-01-01',
              space_override=None, knobs=None, cluster_threshold=CLUSTER_THRESHOLD, n_jobs=1, **kw):
    """Horizon-match per target, apply the pool-wide (median) clusters,
    score every set in every config, rank and aggregate.
    n_jobs: scoring worker processes (1 = serial, None/0 = all CPUs); results are
    identical for any n_jobs. BTC in-sample, 21 members (190,893 sets x 9 configs):
    5.4 min wall on 8 CPUs (1 Oct 2026); serial about 33 min (T2's 20-member serial
    run took 23 min, of which horizon matching was about 16 s).
    Returns (agg, ranked scores, horizon tables, params per target, pools)."""
    tables, params, pools, cache = [], {}, [], {}
    for t in targets:
        tab, params[t] = horizon_match(price, pool, t, start, knobs=knobs, space_override=space_override,
                                       cache=cache)
        tables.append(tab)
        pools.append(Pool.from_price(price, params[t], pool, t, start, holds=holds))
    apply_pool_clusters(pools, cluster_threshold)
    scores = _score_pools(pools, sizes, holds, n_jobs=n_jobs, **kw)
    ranked = rank_configs(pd.concat(scores, ignore_index=True))
    return aggregate(ranked), ranked, pd.concat(tables, ignore_index=True), params, pools
