import os

import numpy as np
import pandas as pd
import pytest

import data_import
from conftest import make_price


def _write(path, df):
    df.drop(columns='return').to_csv(path)


def test_loads_the_requested_ticker_not_the_last_file(tmp_path):
    a, b = make_price(seed=1), make_price(seed=2)
    _write(tmp_path / 'AAA-USD.csv', a)
    _write(tmp_path / 'ZZZ-USD.csv', b)
    d = data_import.data_importer('AAA-USD', data_dir=str(tmp_path))
    d.import_csv_file()
    assert np.allclose(d.df['close'].to_numpy(), a['close'].to_numpy())
    assert d.path == str(tmp_path / 'AAA-USD.csv')


def test_tradingview_style_name_and_ambiguity(tmp_path):
    _write(tmp_path / 'BTCUSD_1D.csv', make_price(seed=1))
    d = data_import.data_importer('BTC-USD', data_dir=str(tmp_path))
    d.import_csv_file()
    assert d.path.endswith('BTCUSD_1D.csv')
    _write(tmp_path / 'BTCUSD_1W.csv', make_price(seed=2))
    with pytest.raises(ValueError):
        data_import.data_importer('BTC-USD', data_dir=str(tmp_path)).import_csv_file()
    with pytest.raises(FileNotFoundError):
        data_import.data_importer('ETH-USD', data_dir=str(tmp_path)).import_csv_file()


def test_explicit_file_path_and_cwd_independence(tmp_path, monkeypatch):
    _write(tmp_path / 'x.csv', make_price(seed=1))
    monkeypatch.chdir('/')
    d = data_import.data_importer('whatever', file_path=str(tmp_path / 'x.csv'))
    d.import_csv_file()
    assert len(d.df) == 1600
    assert os.path.isabs(data_import.DEFAULT_DATA_DIR)
    monkeypatch.setenv('TPI_DATA_DIR', str(tmp_path))
    d = data_import.data_importer('x')
    d.import_csv_file()
    assert d.path == str(tmp_path / 'x.csv')


@pytest.mark.parametrize('scale', [1, 1000])   # seconds and milliseconds
def test_unix_time_column(tmp_path, scale):
    df = make_price(n=900, start='2017-06-01').drop(columns='return')
    raw = df.reset_index()
    raw['time'] = (raw['time'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s') * scale
    raw.to_csv(tmp_path / 'UNIX.csv', index=False)
    d = data_import.data_importer('UNIX', data_dir=str(tmp_path))
    d.import_csv_file(start='2018-01-01')
    assert isinstance(d.df.index, pd.DatetimeIndex)
    assert d.df.index[0] == pd.Timestamp('2018-01-01')
    assert len(d.df.loc['2018-01-01':]) == len(d.df) > 0


def test_nan_ohlc_rows_dropped_and_reported(tmp_path, capsys):
    df = make_price(n=100).drop(columns='return')
    df.iloc[10, df.columns.get_loc('close')] = np.nan
    df.iloc[50, df.columns.get_loc('high')] = np.nan
    df.to_csv(tmp_path / 'NAN.csv')
    d = data_import.data_importer('NAN', data_dir=str(tmp_path))
    d.import_csv_file()
    assert len(d.df) == 98
    assert d.df[['open', 'high', 'low', 'close']].notna().all().all()
    assert list(d.dropped_rows.index) == [df.index[10], df.index[50]]
    assert 'dropped 2 row(s) with NaN OHLC' in capsys.readouterr().out
