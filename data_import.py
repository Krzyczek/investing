import yfinance as yf
import pandas as pd
import os
import re

# Domyślny katalog z danymi: dane_cenowe/ obok TEGO pliku (a nie względem cwd),
# można nadpisać zmienną środowiskową TPI_DATA_DIR albo argumentem data_dir.
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dane_cenowe')

OHLC_COLUMNS = ['open', 'high', 'low', 'close']


def _normalise(name: str) -> str:
    """'BTC-USD' -> 'BTCUSD', 'btc_usd' -> 'BTCUSD' (only letters/digits, upper case)."""
    return re.sub(r'[^0-9A-Za-z]', '', str(name)).upper()


class data_importer():
    def __init__(self, ticker: str, data_dir: str = None, file_path: str = None):
        """ticker     - symbol whose CSV should be loaded (e.g. 'BTC-USD', 'CDR.WA')
        data_dir   - folder with one CSV per ticker; default: $TPI_DATA_DIR or
                     <repo>/dane_cenowe (resolved next to this file, not the cwd)
        file_path  - explicit CSV path; overrides the ticker -> file lookup"""
        self.ticker = ticker
        self.data_dir = data_dir
        self.file_path = file_path
        self.path = None            # the CSV actually loaded
        self.dropped_rows = None    # rows removed because of NaN OHLC
    def __str__(self):
        return f'what is the result? {self.ticker}'

    def _drop_nan_ohlc(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Drop (never fill) bars with any NaN open/high/low/close and report them."""
        cols = [c for c in OHLC_COLUMNS if c in df.columns]
        bad = df[cols].isna().any(axis=1)
        self.dropped_rows = df.loc[bad].copy()
        if bad.any():
            dates = ', '.join(str(i) for i in df.index[bad][:20])
            more = '' if bad.sum() <= 20 else f' ... (+{int(bad.sum()) - 20} more)'
            print(f"[data_import] {self.ticker}: dropped {int(bad.sum())} row(s) with NaN OHLC "
                  f"from {source}: {dates}{more}")
        return df.loc[~bad]

    def import_yfinance_ticker(self):
        df = yf.download(self.ticker, start='2018-01-01', interval='1d')
        # Zabezpieczenie przed błędem w yfinance przy pojedynczym tickerze
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1) # Bezpieczniejsze niż szukanie stringa "Ticker"
         
        df.columns = df.columns.str.lower()
        df = self._drop_nan_ohlc(df, 'yfinance')
        df['return'] = df['close'].pct_change(fill_method=None)
        self.df = df

    def resolve_csv_path(self) -> str:
        """Find the CSV for self.ticker.

        Order: explicit file_path; else in data_dir (arg / $TPI_DATA_DIR /
        <repo>/dane_cenowe) the file '<ticker>.csv' (case-insensitive), else the
        single file whose name (or the part before the first '_', e.g. a
        TradingView export 'BTCUSD_1D.csv') equals the ticker ignoring
        punctuation. Zero or several matches raise instead of guessing."""
        if self.file_path is not None:
            path = os.path.abspath(os.path.expanduser(self.file_path))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"CSV file not found: {path}")
            return path

        folder = self.data_dir or os.environ.get('TPI_DATA_DIR') or DEFAULT_DATA_DIR
        folder = os.path.abspath(os.path.expanduser(folder))
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Data directory not found: {folder}")
        if not self.ticker:
            raise ValueError("A ticker (or file_path) is required to pick the CSV file.")

        csvs = sorted(f for f in os.listdir(folder) if f.lower().endswith('.csv'))
        exact = [f for f in csvs if f.lower() == f'{self.ticker}.csv'.lower()]
        if len(exact) == 1:
            return os.path.join(folder, exact[0])

        want = _normalise(self.ticker)
        matches = [f for f in csvs
                   if _normalise(os.path.splitext(f)[0]) == want
                   or _normalise(os.path.splitext(f)[0].split('_')[0]) == want]
        if len(matches) == 1:
            return os.path.join(folder, matches[0])
        if not matches:
            raise FileNotFoundError(
                f"No CSV for ticker '{self.ticker}' in {folder}. Available: {csvs}")
        raise ValueError(
            f"Ticker '{self.ticker}' matches several CSVs in {folder}: {matches}. "
            f"Pass file_path explicitly.")

    @staticmethod
    def _parse_time(col: pd.Series) -> pd.DatetimeIndex:
        """Date strings or UNIX timestamps (seconds, or milliseconds if > 1e11).
        Timezone-aware values are converted to UTC and made naive."""
        if pd.api.types.is_numeric_dtype(col):
            unit = 'ms' if col.abs().max() > 1e11 else 's'
            idx = pd.to_datetime(col, unit=unit, utc=True)
        else:
            idx = pd.to_datetime(col, utc=True)
        return pd.DatetimeIndex(idx).tz_localize(None).rename('time')

    def import_csv_file(self, start: str = None, end: str = None):
        path = self.resolve_csv_path()
        self.path = path
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        missing = [c for c in ['time'] + OHLC_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing column(s) {missing}")
        df.index = self._parse_time(df.pop('time'))
        df = df.sort_index()
        df = self._drop_nan_ohlc(df, os.path.basename(path))
        if start is not None:
            df = df.loc[start:]
        if end is not None:
            df = df.loc[:end]
        df['return'] = df['close'].pct_change(fill_method=None)
        self.df = df
