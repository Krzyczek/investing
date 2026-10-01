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
    # target 4 -> w = 25: with ties now holding, the earliest member's
    # leave-one-out consensus flips when 3 of its 4 others have (lag 15)
    P = _pool([_regime(shift=s) for s in (0, 5, 10, 15, 20)], target=4)
    o = sc.set_coherence(P, range(5), holds=(20,))[20]
    assert not o['precision_veto'] and not o['cluster_rule']      # 0.9 exactly is not > 0.9
    P2 = _pool([_regime()] * 2 + [_regime(shift=s) for s in (10, 20, 30)])
    assert sc.set_coherence(P2, range(5), holds=(20,))[20]['cluster_rule']
    assert ('m0', 'm1') in P2.cluster_groups


def test_loo_tie_holds_previous_loo_position():
    # member a's others are b (+1 always) and c (+1, then -1 from bar 100):
    # their mean is 0 from bar 100 on -> a's leave-one-out position HOLDS +1
    a = -np.ones(200)
    b = np.ones(200)
    c = np.r_[np.ones(100), -np.ones(100)]
    pos = sc.loo_positions(np.vstack([a, b, c]))
    assert np.all(pos[0] == 1)
    # the full-set score would have followed a's own vote (-1): not used any more
    assert np.all(np.sign((a + b + c)[100:]) == -1)


def test_loo_tie_on_first_bars_stays_flat():
    a = np.ones(100)
    b = np.ones(100)
    c = np.r_[-np.ones(30), np.ones(70)]        # b + c = 0 on bars 0-29
    pos = sc.loo_positions(np.vstack([a, b, c]))
    assert np.all(pos[0, :30] == 0) and np.all(pos[0, 30:] == 1)
    assert np.array_equal(pos[0], tpi.position_from_score((b + c) / 2))


def _agree_pool(agree, names=('a', 'b', 'c', 'd', 'e')):
    P = _pool([_regime(shift=s) for s in (0, 7, 14, 21, 28)])
    P.names = list(names)
    P.agree = np.asarray(agree, dtype=float)
    return P


def _mat(pairs, n=5, base=0.5):
    m = np.full((n, n), base)
    np.fill_diagonal(m, 1.0)
    for (i, j), v in pairs.items():
        m[i, j] = m[j, i] = v
    return m


def test_cluster_rule_uses_median_agreement_across_targets():
    # pair (a, b): > 0.9 at two of three targets -> median > 0.9 -> clustered
    # pair (c, d): > 0.9 at ONE target only       -> median <= 0.9 -> not clustered
    pools = [_agree_pool(_mat({(0, 1): 0.95, (2, 3): 0.93})),
             _agree_pool(_mat({(0, 1): 0.92, (2, 3): 0.88})),
             _agree_pool(_mat({(0, 1): 0.89, (2, 3): 0.86}))]
    assert pools[0].agree[2, 3] > sc.CLUSTER_THRESHOLD
    med, groups = sc.apply_pool_clusters(pools)
    assert np.isclose(med[0, 1], 0.92) and np.isclose(med[2, 3], 0.88)
    assert groups == [('a', 'b')]
    for P in pools:                                    # same clusters in every config
        assert P.cluster_groups == [('a', 'b')]
        assert ('c', 'd', 0.88) in P.watch             # 0.85-0.9 reported, not enforced
        assert sc.set_coherence(P, range(5), holds=(20,))[20]['cluster_rule']
        assert not sc.set_coherence(P, [1, 2, 3, 4], holds=(20,))[20]['cluster_rule']
    # the threshold stays a parameter
    _, g2 = sc.apply_pool_clusters(pools, cluster_threshold=0.87)
    assert g2 == [('a', 'b'), ('c', 'd')]


def test_trades_counted_like_run_backtest():
    # one-bar lag: a flip on the last bar is not traded, the first bar is flat
    assert sc.trades(np.array([1, 1, -1, -1, 1])) == 2
    assert sc.trades(np.array([1, 1, -1, -1, -1])) == 2
    assert sc.trades(np.array([0, 1, 1])) == 1


def test_pick2d_closest_cell_then_textbook():
    grids = [[10, 20, 30], [40, 50, 60]]
    cells = [(10, 40), (10, 50), (20, 50), (30, 60), (20, 60)]
    counts = [90, 71, 69, 71, 50]
    k, ok, nb = sc.pick2d(grids, cells, counts, (20, 50), 70)
    assert ok and cells[k] in [(10, 50), (20, 50), (30, 60)]
    assert cells[k] == (20, 50)                        # |69-70| = |71-70|: nearest to textbook


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


def test_tie_break_median_then_C_at_70_20_then_spread():
    base = dict(k=5, mean_f1=.6, mean_timing=.9, A=.8, min_precision=.5, precision_veto=False,
                cluster_rule=False, weakest='x', excluded=False)
    # C at the 3 configs (hold 20): every set gets ranks {1, 2, 3} -> median 2
    C = {'x': {60: .70, 70: .80, 80: .70}, 'y': {60: .10, 70: .90, 80: .80},
         'z': {60: .69, 70: .79, 80: .90}}
    rows = [dict(members=m, target=t, hold=20, C=cs[t], **base) for m, cs in C.items() for t in cs]
    agg = sc.aggregate(sc.rank_configs(pd.DataFrame(rows)))
    assert list(agg.median_rank) == [2, 2, 2]
    assert list(agg.members) == ['y', 'x', 'z']            # C at 70/20 decides
    # equal median and equal C at 70/20 -> smaller spread first
    ranked = pd.DataFrame([dict(members=m, target=t, hold=20, C=.5, rank=r, **base)
                           for m, rr in (('p', (1, 5, 9)), ('q', (4, 5, 6)))
                           for t, r in zip((60, 70, 80), rr)])
    assert list(sc.aggregate(ranked).members) == ['q', 'p']


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
    agg, ranked, tables, params, pools = sc.rank_pool(price, pool, targets=(20,), holds=(5,), sizes=(5,),
                                               start=str(price.index[0].date()),
                                               space_override={'slow_ma': ('int', 20, 40, 5),
                                                               'parabolic_sar_acceleration': ('float', 0.01, 0.03, 0.01),
                                                               'supertrend_factor': ('float', 2.0, 4.0, 1.0),
                                                               'adx_period': ('int', 10, 20, 5),
                                                               'aroon_length': ('int', 10, 20, 5)})
    assert len(agg) == 1 and set(tables.component) == set(pool)
    assert {'median_rank', 'spread', 'iqr', 'C@70/20'} <= set(agg.columns) or len(ranked) == 1
