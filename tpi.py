import numpy as np
import pandas as pd
from decimal import Decimal
import indicators

# =====================================================================
#  TPI COMPONENT REGISTRY
#
#  To add a new indicator to the TPI you edit ONLY this file:
#    1. write a small signal function:  (df, params) -> array of +1/0/-1
#    2. register it in COMPONENTS with its parameter search space
#    3. (optional) add cross-parameter rules to CONSTRAINTS, tagged with
#       the component they belong to: ('component_name', rule)
#
#  Nothing in optuna_testing.py / evaluation.py / evaluation_test.py
#  has to change: they discover parameters and constraints from here.
#
#  Param spec format:  name: (type, low, high, step)
#     type is 'int' or 'float' (matches optuna suggest_int/suggest_float)
#
#  Every helper below takes an optional `components` list (default: all
#  registered components), so a single indicator can be optimised and
#  robustness-tested SOLO before it is combined with others.
# =====================================================================


# ---------- signal functions (one per indicator) ----------

def _ema_cross_signal(df, p):
    fast = indicators.ema(df, p['fast_ma'])
    slow = indicators.ema(df, p['slow_ma'])
    fast.calculate()
    slow.calculate()
    return np.where(fast.ema > slow.ema, 1, -1)


def _parabolic_sar_signal(df, p):
    psar = indicators.parabolic_sar(
        df,
        start=p['parabolic_sar_start'],
        increment=p['parabolic_sar_acceleration'],
        maximum=p['parabolic_sar_maximum'],
    )
    psar.calculate()
    return np.where(df['close'] > psar.sar_values, 1, -1)


def _adx_signal(df, p):
    # ADX mierzy SIŁĘ trendu, a kierunek daje +DI vs -DI:
    #   +1  gdy ADX > threshold i +DI > -DI  (silny trend wzrostowy)
    #   -1  gdy ADX > threshold i -DI > +DI  (silny trend spadkowy)
    #    0  w pozostałych przypadkach (słaby trend, remis DI, rozgrzewka/NaN)
    a = indicators.adx(df, p['adx_period'])
    a.calculate()
    t = p['threshold']
    strong = (a.adx_values > t).to_numpy()
    plus_di = a.plus_di.to_numpy()
    minus_di = a.minus_di.to_numpy()
    return np.where(strong & (plus_di > minus_di), 1,
                    np.where(strong & (minus_di > plus_di), -1, 0))


def _aroon_signal(df, p):
    ar = indicators.aroon_oscillator(df, p['aroon_length'])
    ar.calculate()
    return np.where(ar.osc > 0, 1, np.where(ar.osc < 0, -1, 0))

def _supertrend_signal(df, p):
    st = indicators.supertrend(
        df,
        atr_period=p['supertrend_atr_period'],
        factor=p['supertrend_factor'],
    )
    st.calculate()
    # direction < 0 == uptrend (konwencja Pine), więc long przy -1
    return np.where(st.direction < 0, 1, -1)


# ---------- the registry: THE single source of truth ----------

COMPONENTS = {
    'ema_cross': {
        'group': 'perpetual',
        'signal': _ema_cross_signal,
        'params': {
            'fast_ma': ('int', 5, 100, 1),
            'slow_ma': ('int', 5, 100, 1),
        },
    },
    'parabolic_sar': {
        'group': 'perpetual',
        'signal': _parabolic_sar_signal,
        'params': {
            'parabolic_sar_start':        ('float', 0.0, 1.0, 0.01),
            # tuning search space changed 30 Sep 2026 (Krzyczek): was 0.01-0.1 step 0.01;
            # widened down to 0.0005 (step 0.0005) so the TPI trade-count horizon grid
            # 60/70/80 is reachable with start/maximum at textbook (BTC-USD 2018-2025/03:
            # 0.0005 -> 63 trades, 0.001 -> 71, 0.0015 -> 83; 0.01 gave 156)
            'parabolic_sar_acceleration': ('float', 0.0005, 0.1, 0.0005),
            'parabolic_sar_maximum':      ('float', 0.1, 1.0, 0.1),
        },
    },
    'adx': {
        'group': 'oscillator',
        'signal': _adx_signal,
        'params': {
            'adx_period': ('int', 5, 100, 1),
            'threshold':  ('int', 5, 50, 1),
        },
    },
    'aroon': {
        'group': 'oscillator',
        'signal': _aroon_signal,
        'params': {
            'aroon_length': ('int', 5, 100, 1),
        },
    },
    'supertrend': {
        'group': 'perpetual',
        'signal': _supertrend_signal,
        'params': {
            'supertrend_atr_period': ('int', 5, 50, 1),
            'supertrend_factor':     ('float', 1.0, 10.0, 0.1),
        },
    },
}

# Cross-parameter validity rules. A trial whose params fail any rule
# gets pruned by the optimizer (replaces the hardcoded
# "if fast_ma_period >= slow_ma_period: raise TrialPruned()").
# Each rule is tagged with its component, so it is only applied when that
# component is part of the selected subset.
CONSTRAINTS = [
    ('ema_cross', lambda p: p['fast_ma'] < p['slow_ma']),
    # AF startowy nie może przekraczać maksymalnego (inaczej acceleration nic nie zmienia)
    ('parabolic_sar', lambda p: p['parabolic_sar_start'] <= p['parabolic_sar_maximum']),
]


# ---------- generic machinery (never needs editing) ----------

def resolve_components(components=None) -> list:
    """Validate a component selection. None -> all registered components.
    A single name may be passed as a string. Returned in registry order."""
    if components is None:
        return list(COMPONENTS)
    if isinstance(components, str):
        components = [components]
    unknown = [c for c in components if c not in COMPONENTS]
    if unknown:
        raise KeyError(f"Unknown TPI component(s) {unknown}; registered: {list(COMPONENTS)}")
    if not components:
        raise ValueError("Component selection is empty.")
    return [c for c in COMPONENTS if c in components]


def params_from_trial(trial, components=None) -> dict:
    """Inverse of suggest_params: rebuild the runtime param dict from a
    (Frozen)Trial. suggest_params stores float params as int '<name>_scaled',
    so trial.params cannot be indexed with param_space() names directly."""
    out = {}
    for name, (ptype, low, high, step) in param_space(components).items():
        if name in trial.params:
            out[name] = trial.params[name]
        elif f'{name}_scaled' in trial.params:
            out[name] = round(trial.params[f'{name}_scaled'] * step, 10)
        else:
            raise KeyError(
                f"trial {getattr(trial, 'number', '?')} has neither "
                f"'{name}' nor '{name}_scaled' - search space changed since the study ran"
            )
    return out

def param_space(components=None) -> dict:
    """Merged search space of the selected components (default: all),
    keyed by param name."""
    space = {}
    for comp_name in resolve_components(components):
        comp = COMPONENTS[comp_name]
        for name, spec in comp['params'].items():
            if name in space and space[name] != spec:
                raise ValueError(
                    f"Parameter '{name}' declared twice with different specs "
                    f"(second time in component '{comp_name}')."
                )
            space[name] = spec
    return space


def suggest_params(trial, components=None) -> dict:
    """Build the full parameter dict from an optuna trial.
    The objective function calls this instead of listing suggest_* lines.

    Float parameters are sampled on an integer scale so Optuna uses exact
    decimal increments (0.01, 0.1, etc.) instead of float precision drift.
    """
    params = {}
    for name, (ptype, low, high, step) in param_space(components).items():
        if ptype == 'int':
            params[name] = trial.suggest_int(name, low, high, step=step)
        elif ptype == 'float':
            if step <= 0:
                raise ValueError(f"Float step for '{name}' must be > 0")

            decimal_step = Decimal(str(step))
            scale = 10 ** (-decimal_step.as_tuple().exponent) if decimal_step.as_tuple().exponent < 0 else 1
            low_i = int(round(low * scale))
            high_i = int(round(high * scale))
            step_i = int(round(step * scale))

            params[name] = trial.suggest_int(f'{name}_scaled', low_i, high_i, step=step_i) / scale
        else:
            raise ValueError(f"Unknown param type '{ptype}' for '{name}'")
    return params


def params_valid(params: dict, components=None) -> bool:
    """True when the parameter combination satisfies all CONSTRAINTS of the
    selected components (default: all)."""
    selected = set(resolve_components(components))
    return all(rule(params) for comp, rule in CONSTRAINTS if comp in selected)


# Wynik TPI == 0 -> utrzymaj poprzednią pozycję (decyzja Krzyczka).
ZERO_ATOL = 1e-12   # group averaging can leave float residue like 1e-17


def position_from_score(score, mode: str = 'long_short') -> np.ndarray:
    """TPI score -> position (signal at the close of each bar, before the
    one-bar lag applied in the backtest).

    score > 0 -> long (+1); score < 0 -> short (-1) in long_short, flat (0)
    in long_only; score == 0 -> HOLD the previous position (no flip, no
    exit). Same hold rule in both modes. If the first bars score 0 there is
    no previous position, so it stays flat until the first non-zero score.
    Uses only past bars (forward fill), so there is no lookahead."""
    if mode not in ('long_only', 'long_short'):
        raise ValueError(f"mode must be 'long_only' or 'long_short', got {mode!r}")
    score = np.asarray(score, dtype=float)
    zero = np.isnan(score) | np.isclose(score, 0.0, rtol=0.0, atol=ZERO_ATOL)
    below = -1.0 if mode == 'long_short' else 0.0
    raw = np.where(zero, np.nan, np.where(score > 0, 1.0, below))
    return pd.Series(raw).ffill().fillna(0.0).to_numpy().astype(int)


class tpi():
    def __init__(self, df):
        self.df = df

    def calculate_tpi(self, params: dict, mode: str = 'long_only', components=None):
        """Compute the TPI from a flat params dict (e.g. optuna trial.params).

        Aggregation preserves the original behaviour: signals are averaged
        inside each group, then the group means are averaged together.
        `components` restricts the TPI to a subset (default: all); with a
        single component or a single group the score is just that mean.
        """
        if mode not in ('long_only', 'long_short'):
            raise ValueError(f"mode must be 'long_only' or 'long_short', got {mode!r}")
        self.components = resolve_components(components)
        groups = {}
        for name in self.components:
            comp = COMPONENTS[name]
            sig = np.asarray(comp['signal'](self.df, params), dtype=float)
            setattr(self, f'{name}_signal', sig)   # kept for inspection
            groups.setdefault(comp.get('group', 'default'), []).append(sig)

        group_means = [np.mean(np.vstack(sigs), axis=0) for sigs in groups.values()]
        score = np.mean(np.vstack(group_means), axis=0)
        self.tpi_score = score

        # Wynik 0 = trzymaj poprzednią pozycję (przedtem: long_short 0 -> long,
        # long_only 0 -> flat); patrz position_from_score
        self.signal = position_from_score(score, mode)