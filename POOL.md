# Indicator pool registry

**Rule: grep this file (and `tpi.COMPONENTS` / `indicators.py`) before building anything. Nothing is re-implemented.** If an indicator or helper is already listed here, extend or reuse it; if it is queued for someone else, coordinate instead of starting a second copy.

## Status tiers

`built` → `solo run documented` → **REVIEWED** → **POOLED**

| tier | meaning |
|---|---|
| built | Implemented in `indicators.py` and registered in `tpi.COMPONENTS` (signal +1/0/-1, params, constraints), with tests. |
| solo run documented | A solo Optuna search + robustness test has been run and its log is listed here. |
| **REVIEWED** | Built **and** reference-checked (values match an independent reference, e.g. the TradingView/Pine definition) **and** solo run documented. **A REVIEWED indicator may be used in set tests**, even if it failed solo. |
| **POOLED** | Passed the solo test (feasible, robust on the colour table) **plus** the reviewer's check. |

This two-tier rule (REVIEWED may enter set tests; POOLED = solo pass + review) is **decided**; Krzyczek may still overrule it.

**Every set (joint) result must list which of its members failed solo.**

## Signal rule that all results depend on

TPI score exactly 0 = **hold the previous position** (no flip, no exit), in both modes. If the first scored bars are 0, the TPI stays flat until the first non-zero score. In `long_short` (the primary mode), nothing goes flat after that first non-zero bar: the position is always long or short, unless Krzyczek asks for a separate flat state. (`long_only`, diagnostic only: a negative score still exits to flat.) The backtest's one-bar signal lag is unchanged. Rule introduced in commit `9ee57cd`; results from earlier commits are **pre-hold-rule**.

## Registry

"solo result" = feasible trials / robust candidates, colour table, framework SHA the run used. BTC-USD daily, 300 trials, seed 42, in-sample 2018-01-01 to 2025-03-31, `long_short` unless noted. **All solo results so far are pre-hold-rule, have 0 feasible trials, and are information only.**

### In `main`

| indicator | component key | status tier | owner/branch | solo result | review | log paths |
|---|---|---|---|---|---|---|
| SMA + EMA (EMA cross) | `ema_cross` (uses `indicators.sma`, `indicators.ema`) | solo run documented | main (original) | 0 feasible / 0 robust, main, `a5fbf73` (pre-hold-rule) | reference check not done | `/workspace/ta/work/runs/solo_ema_cross_long_short.log` (+ `.json`, `_trials.csv`; `long_only` diagnostic alongside) |
| ADX (directional: +DI vs -DI above threshold) | `adx` | built | main (directional since PR #1) | no solo run yet (joint runs only) | reference check not done | - |
| Aroon oscillator | `aroon` | built | main (original) | no solo run yet (joint runs only) | reference check not done | - |
| Parabolic SAR | `parabolic_sar` | built | main (original) | no solo run yet (joint runs only) | reference check not done | - |
| Supertrend | `supertrend` | solo run documented | main (original) | 0 feasible / 0 robust, main, `a5fbf73` (pre-hold-rule) | reference check not done | `/workspace/ta/work/runs/solo_supertrend_long_short.log` (+ `.json`, `_trials.csv`) |

Helpers in `indicators.py` with **no signal / not a component** (reuse them, don't rebuild):

| helper | purpose |
|---|---|
| `standard_deviation_bands` | mean ± k·std bands (Bollinger rework builds on this) |
| `get_hurst_series` | rolling Hurst exponent |
| `calculate_rolling_adf_pvalues` | rolling ADF p-values (lazy `statsmodels` import) |

### On branches

| indicator | component key | status tier | owner/branch | solo result | review | log paths |
|---|---|---|---|---|---|---|
| Donchian channel | `donchian` | solo run documented | A / `ind-donchian` | 0 feasible (206 complete, 94 pruned) / 0 robust, main, `a5fbf73` (also `619af99`) (pre-hold-rule) | pending | `/workspace/ta/runs/builder-a/solo_donchian_long_short.*` |
| Keltner channel | `keltner` | solo run documented | A / `ind-keltner` | 0 feasible (300 complete) / 0 robust, main, `a5fbf73` (also `619af99`) (pre-hold-rule) | reference verification on branch (`4bd4558`); reviewer sign-off pending | `/workspace/ta/runs/builder-a/solo_keltner_long_short.*` |
| Vortex (VI+/VI-) | `vortex` | solo run documented | B / `ind-vortex` | 0 feasible (300 complete) / 0 robust, main, `619af99` (pre-hold-rule) | reference check log present; reviewer sign-off pending | `/workspace/ta/runs/builder-b/vortex/` |
| Bollinger bands | - | in rework | B | - | - | Being reworked on top of `standard_deviation_bands`; the old `ind-bollinger` branch is **superseded**, do not use it. |

### Queued (not built yet)

| indicator | owner |
|---|---|
| RSI | A |
| ROC | A |
| Hull MA | A |
| TEMA | A |
| OBV volume-trend | A |
| Linear-regression slope t-stat | A |
| MACD | B |
| CCI | B |
| KAMA | B |
| Ichimoku | B |
| Hurst / ADF regime gate (reuse `get_hurst_series`, `calculate_rolling_adf_pvalues`) | B |
| Ehlers ITrend | B |
