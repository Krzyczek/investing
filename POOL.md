# Indicator pool registry

**Rule: grep this file (and `tpi.COMPONENTS` / `indicators.py`) before building anything. Nothing is re-implemented.** If an indicator or helper is already listed here, extend or reuse it. If it's queued for someone else, coordinate with them instead of starting a second copy.

## Method

**Testing is done only on sets of 5-7 indicators.** There is no solo validation: an indicator is never Optuna-searched or robustness-tested on its own to decide whether it enters the pool. A set is tuned jointly and has to pass the colour-table robustness test and the hold-out gate as a set.

## Pool entry rule

An indicator enters the pool when it is all three of:

1. **built**: implemented in `indicators.py` and registered in `tpi.COMPONENTS` (signal +1/0/-1, params, constraints), with tests;
2. **reference-checked**: its values match an independent reference (e.g. the TradingView/Pine definition, TA-Lib, `ta`, `pandas_ta`, or a plain loop);
3. **reviewed**: a reviewer has signed off on the code, the reference check and the signal definition.

Only indicators that meet all three may be used in set tests.

## Signal rule that all results depend on

A TPI score of exactly 0 = **hold the previous position** (no flip, no exit), in both modes. If the first scored bars are 0, the TPI stays flat until the first non-zero score. In `long_short` (the primary mode), nothing goes flat after that first non-zero bar: the position is always long or short, unless Krzyczek asks for a separate flat state. (In `long_only`, which is diagnostic only, a negative score still exits to flat.) The backtest's one-bar signal lag is unchanged. The rule was introduced in commit `9ee57cd`.

## Registry

### In `main` / `tpi-framework-fixes`

| indicator | key | owner/branch | built | reference check | review | log paths |
|---|---|---|---|---|---|---|
| SMA + EMA (EMA cross) | `ema_cross` (uses `indicators.sma`, `indicators.ema`) | main (original) | yes | not done | not done | - |
| ADX (directional: +DI vs -DI above threshold) | `adx` | main (directional since PR #1) | yes | not done | not done | - |
| Aroon oscillator | `aroon` | main (original) | yes | not done | not done | - |
| Parabolic SAR | `parabolic_sar` | main (original) | yes | not done | not done | - |
| Supertrend | `supertrend` | main (original) | yes | not done | not done | - |

Helpers in `indicators.py` with **no signal and not a component** (reuse them, don't rebuild):

| helper | purpose |
|---|---|
| `standard_deviation_bands` | mean ± k·std bands (the Bollinger rework builds on this) |
| `get_hurst_series` | rolling Hurst exponent |
| `calculate_rolling_adf_pvalues` | rolling ADF p-values (lazy `statsmodels` import) |

### On branches (not merged yet)

All of these are built and reference-checked. None is reviewed yet, so none may be used in official set tests until review.

| indicator | key | owner/branch | built | reference check | review | log paths |
|---|---|---|---|---|---|---|
| Donchian channel | `donchian` | A / `ind-donchian` @ `72e22b3` | yes | yes (`03baffd`: loop, TA-Lib, ta) | pending | `/workspace/ta/runs/builder-a/` |
| Keltner channel | `keltner` | A / `ind-keltner` @ `f2c7936` | yes | yes (`4bd4558`: loop, ta, TA-Lib) | pending | `/workspace/ta/runs/builder-a/` |
| RSI vs 50 | `rsi50` | A / `ind-rsi50` @ `6d6e046` | yes | yes (`acd720c`: loop, TA-Lib, pandas_ta, ta) | pending | - |
| ROC sign | `roc` | A / `ind-roc` @ `75a3705` | yes | yes (`3b3ef93`: loop, TA-Lib, ta, pandas_ta) | pending | - |
| Hull MA slope | `hull` | A / `ind-hull` @ `88fe6ac` | yes | yes (`125efc4`: loop, TA-Lib WMA, pandas_ta) | pending | - |
| TEMA vs price | `tema` | A / `ind-tema` @ `d041e6c` | yes | yes (`a986c61`: loop, TA-Lib, pandas_ta) | pending | - |
| Vortex (VI+/VI-) | `vortex` | B / `ind-vortex` @ `2b6ae3c` | yes | yes (`reference_check.log`) | pending | `/workspace/ta/runs/builder-b/vortex/` |
| CCI | `cci` | B / `ind-cci` @ `af0cba5` | yes | yes (`reference_check.log`) | pending | `/workspace/ta/runs/builder-b/cci/` |
| KAMA | `kama` | B / `ind-kama` @ `992f79c` | yes | yes (`reference_check.log`) | pending | `/workspace/ta/runs/builder-b/kama/` |
| MACD (histogram sign) | `macd` | B / `ind-macd` @ `e0e6f6e` | yes | yes (`reference_check.log`) | pending | `/workspace/ta/runs/builder-b/macd/` |
| Bollinger bands | `bollinger` | B / `ind-bollinger` @ `5f928bc` | yes, reworked on an extended `standard_deviation_bands` (sma basis + multiplier) | yes (`reference_check.log`) | pending | `/workspace/ta/runs/builder-b/bollinger/`. The earlier version is `ind-bollinger-dropped` (`/workspace/ta/runs/builder-b/bollinger-dropped/`); it is **dropped**, don't use it. |

### Queued (not built yet)

| indicator | owner |
|---|---|
| OBV volume-trend | A |
| Linear-regression slope t-stat | A |
| Hurst / ADF regime gate (reuse `get_hurst_series`, `calculate_rolling_adf_pvalues`) | B |
| Ehlers ITrend | B |
| Ichimoku (will reuse the Donchian helper) | B |

## Note on earlier solo results

Solo Optuna/robustness runs were made before the method changed (`ema_cross`, `supertrend`, `donchian`, `keltner`, `vortex`, all BTC-USD, 0 feasible trials). They are **information only, pre-method-change**. They don't count for or against pool entry, and they aren't used to pick sets. Their logs are still in `/workspace/ta/work/runs/`, `/workspace/ta/runs/builder-a/` and `/workspace/ta/runs/builder-b/vortex/`.
