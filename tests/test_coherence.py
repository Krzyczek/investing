import numpy as np
import pandas as pd

import coherence


def test_report_tables_and_csv(price, tmp_path):
    rep = coherence.coherence_report(price, coherence.TEXTBOOK_PARAMS,
                                     csv_path=str(tmp_path / 'coh.csv'))
    comps = list(rep.agreement.index)
    assert len(comps) == 5 and len(rep.summary) == 5
    assert np.allclose(np.diag(rep.agreement.to_numpy(float)), 100)
    assert np.allclose(rep.agreement.to_numpy(float), rep.agreement.to_numpy(float).T)
    assert rep.summary['coherence_score'].is_monotonic_decreasing
    assert (tmp_path / 'coh.csv').exists() and (tmp_path / 'coh_lags.csv').exists()
    assert set(rep.lags.columns) == {'date', 'direction', *comps}


def test_lag_sign_and_major_flip_rule():
    idx = pd.date_range('2020-01-01', periods=60, freq='D')
    state = pd.Series([-1] * 20 + [1] * 25 + [-1] * 3 + [1] * 12, index=idx, dtype=float)
    flips = coherence.major_flips(state, min_hold=10)
    # the 3-bar dip and the return from it are not major flips
    assert list(flips['bar']) == [20]
    state2 = pd.Series([-1] * 20 + [1] * 2 + [-1] * 3 + [1] * 35, index=idx, dtype=float)
    assert list(coherence.major_flips(state2, min_hold=10)['bar']) == [25]
    late = np.array([-1] * 23 + [1] * 37, float)   # flips 3 bars after the consensus
    early = np.array([-1] * 18 + [1] * 42, float)  # flips 2 bars before
    assert coherence._lag(late, 20, 1, 10) == 3
    assert coherence._lag(early, 20, 1, 10) == -2
    assert np.isnan(coherence._lag(late, 20, 1, 2))
