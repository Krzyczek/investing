import numpy as np
from decimal import Decimal
import indicators

# =====================================================================
#  TPI COMPONENT REGISTRY
#
#  To add a new indicator to the TPI you edit ONLY this file:
#    1. write a small signal function:  (df, params) -> array of +1/0/-1
#    2. register it in COMPONENTS with its parameter search space
#    3. (optional) add cross-parameter rules to CONSTRAINTS
#
#  Nothing in optuna_testing.py / evaluation.py / evaluation_test.py
#  has to change: they discover parameters and constraints from here.
#
#  Param spec format:  name: (type, low, high, step)
#     type is 'int' or 'float' (matches optuna suggest_int/suggest_float)
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
            'parabolic_sar_acceleration': ('float', 0.01, 0.1, 0.01),
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
CONSTRAINTS = [
    lambda p: p['fast_ma'] < p['slow_ma'],
]


# ---------- generic machinery (never needs editing) ----------

def params_from_trial(trial) -> dict:
    """Inverse of suggest_params: rebuild the runtime param dict from a
    (Frozen)Trial. suggest_params stores float params as int '<name>_scaled',
    so trial.params cannot be indexed with param_space() names directly."""
    out = {}
    for name, (ptype, low, high, step) in param_space().items():
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

def param_space() -> dict:
    """Merged search space of every component, keyed by param name."""
    space = {}
    for comp_name, comp in COMPONENTS.items():
        for name, spec in comp['params'].items():
            if name in space and space[name] != spec:
                raise ValueError(
                    f"Parameter '{name}' declared twice with different specs "
                    f"(second time in component '{comp_name}')."
                )
            space[name] = spec
    return space


def suggest_params(trial) -> dict:
    """Build the full parameter dict from an optuna trial.
    The objective function calls this instead of listing suggest_* lines.

    Float parameters are sampled on an integer scale so Optuna uses exact
    decimal increments (0.01, 0.1, etc.) instead of float precision drift.
    """
    params = {}
    for name, (ptype, low, high, step) in param_space().items():
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


def params_valid(params: dict) -> bool:
    """True when the parameter combination satisfies all CONSTRAINTS."""
    return all(rule(params) for rule in CONSTRAINTS)


class tpi():
    def __init__(self, df):
        self.df = df

    def calculate_tpi(self, params: dict, mode: str = 'long_only'):
        """Compute the TPI from a flat params dict (e.g. optuna trial.params).

        Aggregation preserves the original behaviour: signals are averaged
        inside each group, then the group means are averaged together.
        """
        groups = {}
        for name, comp in COMPONENTS.items():
            sig = np.asarray(comp['signal'](self.df, params), dtype=float)
            setattr(self, f'{name}_signal', sig)   # kept for inspection
            groups.setdefault(comp.get('group', 'default'), []).append(sig)

        group_means = [np.mean(np.vstack(sigs), axis=0) for sigs in groups.values()]
        score = np.mean(np.vstack(group_means), axis=0)
        self.tpi_score = score

        if mode == 'long_short':
            self.signal = np.where(score >= 0, 1, -1)
        else:
            self.signal = np.where(score > 0, 1, 0)