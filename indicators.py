import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


class sma():
    def __init__(self,df: pd.DataFrame,sma_length:int,source:str='close') -> pd.Series:
        self.df = df
        self.sma_length = sma_length
        self.source = source
    def calculate(self):
        return self.df[self.source].rolling(self.sma_length).mean()


# def sma(df: pd.DataFrame,length: int,source: str = 'close') -> pd.Series:
#     return df[source].rolling(length).mean()
   
class ema():
    def __init__(self,df: pd.DataFrame,ema_length:int, source:str = 'close'):
        self.df = df
        self.ema_length = ema_length
        self.source = source
    
    def calculate(self) -> pd.Series:
        self.ema = self.df[self.source].ewm(span=self.ema_length,adjust=False).mean()




def standard_deviation_bands(series: pd.Series, length: int = 20) -> list:
    # Używamy okna kroczącego! Wstęga w dniu X wie tylko to, co działo się przez ostatnie 'length' dni.
    deviation = series.rolling(window=length).std()
    
    first_band = series + deviation
    second_band = series + 2 * deviation
    first_negative = series - deviation
    second_negative = series - 2 * deviation
    
    return [first_band, second_band, first_negative, second_negative]

def _calculate_single_hurst(price_array, max_lags=20):
    # 1. Zawsze pracujemy na logarytmach cen! (Kluczowe w finansach)
    log_prices = np.log(price_array)
    lags = range(2, max_lags)
    
    # 2. Zamiast np.std() używamy Średniej Wartości Bezwzględnej.
    # To pozwala zachować "kierunek i moc" trendu w wyliczeniach.
    tau = [np.mean(np.abs(log_prices[lag:] - log_prices[:-lag])) for lag in lags]
    
    tau = np.maximum(tau, 1e-8)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    return poly[0]

def get_hurst_series(price_series: pd.Series, window: int = 100, max_lags: int = 20) -> pd.Series:
    """
    Przesuwa okno po serii cenowej i zwraca nową serię z wartościami Hursta.
    
    Parametry:
    - price_series: kolumna z cenami (np. df['close'])
    - window: szerokość okna (ile wierszy wstecz bierzemy do kalkulacji)
    - max_lags: maksymalne opóźnienie dla algorytmu Hursta
    """
    # Aplikujemy funkcję matematyczną do każdego przesuniętego okna
    hurst_series = price_series.rolling(window=window).apply(
        _calculate_single_hurst, 
        raw=True, 
        kwargs={'max_lags': max_lags}
    )
    
    return hurst_series


    

def calculate_rolling_adf_pvalues(price_series: pd.Series, window: int = 200) -> pd.Series:
    """
    Oblicza kroczący test ADF dla serii cen i zwraca pd.Series 
    zawierający wyłącznie wartości p-value.
    
    Parametry:
    - price_series: Dane wejściowe (np. ceny zamknięcia BTC, ETH lub SOL)
    - window: Rozmiar okna (dla interwału 1D rekomendowane 180-250 dni)
    """
    
    # 1. Funkcja pomocnicza wyciągająca tylko p-value z pojedynczego okna
    def get_pvalue(window_data):
        try:
            # adfuller zwraca tuple, interesuje nas indeks [1]
            result = adfuller(window_data, autolag='AIC')
            return result[1]
        except Exception:
            # Zabezpieczenie na wypadek błędów matematycznych (np. brak zmienności)
            return np.nan

    # 2. Zastosowanie okna kroczącego na serii danych
    # raw=True drastycznie przyspiesza obliczenia, przekazując tablice numpy zamiast obiektów pandas
    p_values = price_series.rolling(window=window).apply(get_pvalue, raw=True)
    
    # 3. Nadanie odpowiedniej nazwy dla serii
    p_values.name = f'adf_p_value_{window}d'
    
    return p_values


import numpy as np
import pandas as pd


class adx():
    def __init__(self, df: pd.DataFrame, adx_length: int = 14, high: str = 'high', low: str = 'low', close: str = 'close') -> pd.Series:
        self.df = df
        self.adx_length = adx_length
        self.high = high
        self.low = low
        self.close = close

    def _wilder_smoothing(self, series: pd.Series) -> pd.Series:
        """
        Wygładzanie metodą Wildera (krok 2/5 z opisu):
        - pierwsza wartość = suma pierwszych `adx_length` elementów
        - kolejne = poprzednia - poprzednia/length + bieżąca
        Działa też gdy seria zaczyna się od NaN (np. DX powstałe z wcześniej
        wygładzonych TR/+DM/-DM).
        """
        length = self.adx_length
        values = series.to_numpy(dtype='float64')
        smoothed = np.full(len(values), np.nan)

        first_valid = series.first_valid_index()
        if first_valid is None:
            return pd.Series(smoothed, index=series.index)

        start_pos = series.index.get_loc(first_valid)
        if start_pos + length > len(values):
            return pd.Series(smoothed, index=series.index)

        smoothed[start_pos + length - 1] = np.nansum(values[start_pos:start_pos + length])

        for i in range(start_pos + length, len(values)):
            smoothed[i] = smoothed[i - 1] - (smoothed[i - 1] / length) + values[i]

        return pd.Series(smoothed, index=series.index)

    def calculate(self) -> pd.Series:
        df = self.df
        length = self.adx_length

        high = df[self.high]
        low = df[self.low]
        close = df[self.close]
        prev_close = close.shift(1)

        # krok 1: +DM, -DM, True Range
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=df.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=df.index
        )

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        # krok 2: wygładzanie Wildera dla TR, +DM, -DM
        smoothed_tr = self._wilder_smoothing(tr)
        smoothed_plus_dm = self._wilder_smoothing(plus_dm)
        smoothed_minus_dm = self._wilder_smoothing(minus_dm)

        # krok 3: +DI, -DI
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)

        # krok 4: DX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

        # krok 5-6: ADX = wygładzone DX, przeskalowane z "sumy" na "średnią"
        self.adx_values = self._wilder_smoothing(dx) / length

        return self.adx_values