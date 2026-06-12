import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def sma(df: pd.DataFrame,length: int,source: str = 'close') -> pd.Series:
    return df[source].rolling(length).mean()
   

def ema(df: pd.DataFrame, length: int, source: str = 'close') -> pd.Series:
    return df[source].ewm(span=length,adjust=False).mean()

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