"""set_coherence v2 on synthetic signals (plus one smoke run on the repo
components, which are the only ones it needs)."""
import numpy as np
import pandas as pd

import set_coherence as sc
import tpi


def _regime(n=400, period=50, shift=0):
    t = np.arange(n)
    return np.where(((t - shift) // period) % 2 == 0, 1.0, -1.0)


def _pool(rows, holds=(20,), target=8):
    return sc.Pool([f'm{i}' for i in range(len(rows))], np.vstack(rows), target, holds=holds)


def test_hold_positions_matches_framework_rule():
    S = np.random.default_rng(3).choice([-1, -0.5, 0, 0.5, 1], size=(4, 300))
    for r in range(4):
        assert np.array_equal(sc.hold_positions(S[r:r + 1])[0], tpi.position_from_score(S[r]))


def test_coin_flip_member_is_vetoed_on_precision():
    rng = np.random.default_rng(0)
    noise = np.where(rng.random(400) < 0.5, 1.0, -1.0)
    P = _pool([_regime(shift=s) for s in (0, 1, 2, 3)] + [noise])
    o = sc.set_coherence(P, range(5), holds=(20,), detail=True)[20]
    pm = o['per_member'].set_index('member')
    assert pm.loc['m4', 'precision'] < sc.PRECISION_VETO
    assert o['precision_veto'] and o['excluded'] and o['weakest'] == 'm4'


def test_clean_set_passes_and_cluster_rule_fires_on_duplicates():
    P = _pool([_regime(shift=s) for s in (0, 5, 10, 15, 20)])
    o = sc.set_coherence(P, range(5), holds=(20,))[20]
    assert not o['precision_veto'] and not o['cluster_rule']      # 0.9 exactly is not > 0.9
    P2 = _pool([_regime()] * 2 + [_regime(shift=s) for s in (10, 20, 30)])
    assert sc.set_coherence(P2, range(5), holds=(20,))[20]['cluster_rule']
    assert ('m0', 'm1') in P2.cluster_groups


def test_loo_zero_broken_by_full_set_score():
    a = np.r_[np.ones(100), -np.ones(100)]
    b, c = np.ones(200), -np.ones(200)
    R = np.vstack([a, b, c])
    total = R.sum(0)
    loo = (total[None] - R) / 2
    assert np.all(loo[0] == 0)
    broken = np.where(np.abs(loo) <= sc.ATOL, total[None] / 3, loo)
    assert np.array_equal(np.sign(broken[0]), np.sign(a))        # = member's own vote


def test_timing_clipped_and_zero_without_matches():
    P = _pool([_regime()] * 4 + [-_regime()])
    pm = sc.set_coherence(P, range(5), holds=(20,), detail=True)[20]['per_member'].set_index('member')
    assert pm.loc['m4', 'timing'] == 0 and 0 <= pm.timing.min() <= pm.timing.max() <= 1


def test_window_is_pool_wide():
    assert (sc.window_w(60, 2647), sc.window_w(70, 2647), sc.window_w(80, 2647)) == (11, 9, 8)
    assert sc.window_w(1000, 2647) == 5


def test_no_cv_term_weights():
    P = _pool([_regime(shift=s) for s in (0, 5, 10, 15, 20)])
    o = sc.set_coherence(P, range(5), holds=(20,))[20]
    assert np.isclose(o['C'], 0.4 * o['mean_f1'] + 0.3 * o['mean_timing'] + 0.3 * o['A'])


def test_rank_aggregate_orders_by_median_rank():
    rows = []
    base = dict(k=5, mean_f1=.6, mean_timing=.9, A=.8, min_precision=.5, precision_veto=False,
                cluster_rule=False, excluded=False)
    for t in (60, 70, 80):
        for h in (15, 20, 30):
            rows += [dict(members='a+b+c+d+e', target=t, hold=h, C=0.8, weakest='a', **base),
                     dict(members='f+g+h+i+j', target=t, hold=h, C=0.9 if h == 20 else 0.1,
                          weakest='f', **base)]
    agg = sc.aggregate(sc.rank_configs(pd.DataFrame(rows)))
    assert agg.members.iloc[0] == 'a+b+c+d+e'
    assert agg.set_index('members').loc['f+g+h+i+j', 'spread'] == 1


def test_excluded_get_bottom_rank():
    rows = [dict(members=m, k=5, target=70, hold=20, C=c, mean_f1=.5, mean_timing=.5, A=.5,
                 min_precision=.5, weakest='x', precision_veto=ex, cluster_rule=False, excluded=ex)
            for m, c, ex in (('a', .9, True), ('b', .5, False), ('c', .4, False))]
    r = sc.rank_configs(pd.DataFrame(rows)).set_index('members')['rank']
    assert (r['b'], r['c'], r['a']) == (1, 2, 3)


def test_three_new_rule():
    assert not sc.differs_enough('ema_cross+parabolic_sar+supertrend+adx+donchian')
    assert sc.differs_enough('ema_cross+adx+donchian+keltner+vortex')


def test_textbook_falls_back_to_mid_range_for_unknown_components(monkeypatch):
    monkeypatch.setitem(tpi.COMPONENTS, 'dummy', {'group': 'perpetual', 'signal': None,
                                                  'params': {'dummy_len': ('int', 10, 30, 1)}})
    assert sc.textbook('dummy') == {'dummy_len': 20}


def test_smoke_rank_pool_on_repo_components(price):
    pool = list(sc.SET1_BASELINE)
    agg, ranked, tables, params = sc.rank_pool(price, pool, targets=(20,), holds=(5,), sizes=(5,),
                                               start=str(price.index[0].date()),
                                               space_override={'slow_ma': ('int', 20, 40, 5),
                                                               'parabolic_sar_acceleration': ('float', 0.01, 0.03, 0.01),
                                                               'supertrend_factor': ('float', 2.0, 4.0, 1.0),
                                                               'adx_period': ('int', 10, 20, 5),
                                                               'aroon_length': ('int', 10, 20, 5)})
    assert len(agg) == 1 and set(tables.component) == set(pool)
    assert {'median_rank', 'spread', 'iqr', 'C@70/20'} <= set(agg.columns) or len(ranked) == 1
