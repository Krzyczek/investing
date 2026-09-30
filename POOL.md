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
| Parabolic SAR | `parabolic_sar` | main (original); **tuning search space changed 30 Sep** (see below) | yes | not done | range POOL-READY at `0ef1708` | - |
| Supertrend | `supertrend` | main (original) | yes | not done | not done | - |

Helpers in `indicators.py` with **no signal and not a component** (reuse them, don't rebuild):

| helper | purpose |
|---|---|
| `standard_deviation_bands` | mean ± k·std bands (the Bollinger rework builds on this) |
| `get_hurst_series` | rolling Hurst exponent |
| `calculate_rolling_adf_pvalues` | rolling ADF p-values (lazy `statsmodels` import) |

### On branches (not merged yet)

Review status as of 30 Sep 2026, 12:10 BST, from the TA Validation Reviewer (`/workspace/ta/runs/reviewer/review_summary.md`). POOL-READY means built, reference-checked and reviewed, so the indicator may be used in set tests. Builder A's merge heads are tree-identical to the reviewed SHA merged onto `0ef1708`, so the verdict carries over to both.

| indicator | key | owner/branch | reviewed SHA | current head | review | log paths |
|---|---|---|---|---|---|---|
| Donchian channel | `donchian` | A / `ind-donchian` | `5ec61e9` | `5b3ced7` | POOL-READY | `/workspace/ta/runs/builder-a/` |
| Keltner channel | `keltner` | A / `ind-keltner` | `bce3c81` | `e27195f` | POOL-READY | `/workspace/ta/runs/builder-a/` |
| RSI vs 50 | `rsi50` | A / `ind-rsi50` | `1ddab3e` | `7f42257` | POOL-READY | `/workspace/ta/runs/builder-a/` |
| ROC sign | `roc` | A / `ind-roc` | `eda4ccb` | `935f582` | POOL-READY | `/workspace/ta/runs/builder-a/` |
| Hull MA slope | `hull` | A / `ind-hull` | `4152163` | `acf1bfe` | POOL-READY | `/workspace/ta/runs/builder-a/` |
| TEMA vs price | `tema` | A / `ind-tema` | `f85895c` | `7e05eb2` | POOL-READY (range 5-452) | `/workspace/ta/runs/builder-a/` |
| Vortex (VI+/VI-) | `vortex` | B / `ind-vortex` | `75398c2` | `75398c2` | POOL-READY | `/workspace/ta/runs/builder-b/vortex/` |
| CCI | `cci` | B / `ind-cci` | `1f46c0e` | `1f46c0e` | POOL-READY | `/workspace/ta/runs/builder-b/cci/` |
| KAMA | `kama` | B / `ind-kama` | `60b38a4` | `60b38a4` | POOL-READY | `/workspace/ta/runs/builder-b/kama/` |
| MACD (histogram sign) | `macd` | B / `ind-macd` | `7ab2696` | `7ab2696` | POOL-READY (fast 2-74, slow 5-100) | `/workspace/ta/runs/builder-b/macd/` |
| Bollinger bands | `bollinger` | B / `ind-bollinger` | `6534ea8` | `6534ea8` | POOL-READY | `/workspace/ta/runs/builder-b/bollinger/`. `ind-bollinger-dropped` is **dropped**; don't use it. |
| OBV volume-trend | `obv` | A / `ind-obv` | - | `21159cf` | **FAILS check (4)**: trades above the band across obv_ema_length 5-150; all-zero volume votes 0 silently. Fix pending, not in pool | `/workspace/ta/runs/builder-a/` |
| Linear-regression slope t-stat | `linreg` | A / `ind-linreg` | `0992d7d` | `0992d7d` (decoder fix) | POOL-READY | `/workspace/ta/runs/builder-a/` |
| Hurst / ADF regime gate | - | B / `ind-regime-gate` | `12a69fe` | `12a69fe` | POOL-READY (about ema_cross at loose gates) | `/workspace/ta/runs/builder-b/` |
| Ehlers ITrend | - | B / `ind-ehlers-itrend` | `6f12cc3` | `6f12cc3` | POOL-READY (94-96% overlap with ema_cross; cluster rule applies) | `/workspace/ta/runs/builder-b/` |
| Ichimoku | - | B / `ind-ichimoku` | `7fbea9c` | `7fbea9c` (rebased onto Donchian `5b3ced7`; `0841b35` dropped) | POOL-READY (`5b3ced7` is an ancestor, so merging `7fbea9c` brings the reviewed Donchian) | `/workspace/ta/runs/builder-b/` |

Parabolic SAR's widened range (on this branch at `0ef1708`) is also POOL-READY.

## Tuning search space changed 30 Sep (Krzyczek)

Widened minimally so that every trade-count target of the set-coherence horizon grid is REACHED. A target T (60, 70 or 80) counts as reached if some valid value gives T±1 trades; the ±1 allows for flip parity. Trades are counted the way run_backtest counts them, on BTC-USD in-sample 2018-01-01 to 2025-03-31, with the other params at textbook. There is no scaling fallback. Details and evidence: `/workspace/ta/runs/proto_coherence/BTC-USD/v2/widened_ranges.{json,md}`.

| indicator | param | old | new | reached at 60 / 70 / 80 | where |
|---|---|---|---|---|---|
| Parabolic SAR | `parabolic_sar_acceleration` | 0.01-0.1 step 0.01 | **0.0004-0.1 step 0.0001** | 0.0004→61 / 0.001→71 / 0.0013→81 | this branch (`tpi.py`). Start 0.0-1.0 and max 0.1-1.0 are unchanged, and `start <= maximum` is kept (`tests/test_psar_search_space.py`) |
| TEMA | `tema_length` | 5-150 | **5-452** | 452→60 / 399→70 / 252→79-81 | `ind-tema` (builder A, reviewed `f85895c`, head `7e05eb2`; agrees) |
| MACD | `macd_fast` | 2-50 | **2-74** (slow 5-100, signal 2-50 unchanged, `fast < slow` kept) | 74/100→61 / 50/100→71 / 50/51→79 | `ind-macd` (builder B, `7ab2696`, reviewed; aligned. The earlier 2-61 / 5-200 proposal is superseded) |

Any set run that contains `parabolic_sar`, `tema` or `macd` must log "tuning search space changed 30 Sep" together with the old and new ranges above.

## Set coherence v2 (`set_coherence.py`, approved by Krzyczek 30 Sep)

Sets of 5-7 are ranked with `set_coherence.rank_pool`; see the module docstring. It uses pool-wide horizon matching per target and 9 configs (targets 60/70/80 x holds 15/20/30), with C = 0.4 F1 + 0.3 timing + 0.3 A. Sets are excluded by a precision veto (< 0.3) and by a cluster rule (> 0.9 agreement). It works with whatever components are registered.

**v2.1 fixes (30 Sep):**
1. The cluster rule uses each pair's MEDIAN agreement across targets 60/70/80. It is computed once per pool, and the same clusters apply to all 9 configs.
2. A leave-one-out vote of exactly 0 HOLDS the previous leave-one-out position, like the zero-score hold rule (flat until the first non-zero score). It is no longer broken with the full-set score.
3. The final order is median rank, then C at 70/20, then spread, then IQR.
4. Trades are counted as run_backtest counts them (one-bar lag).
5. MACD is matched natively on (fast, slow).

## Note on earlier solo results

Solo Optuna/robustness runs were made before the method changed (`ema_cross`, `supertrend`, `donchian`, `keltner`, `vortex`, all BTC-USD, 0 feasible trials). They are **information only, pre-method-change**. They don't count for or against pool entry, and they aren't used to pick sets. Their logs are still in `/workspace/ta/work/runs/`, `/workspace/ta/runs/builder-a/` and `/workspace/ta/runs/builder-b/vortex/`.
