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
        dx = dx.fillna(0)

        # krok 5-6: ADX = wygładzone DX, przeskalowane z "sumy" na "średnią"
        self.adx_values = self._wilder_smoothing(dx) / length

        return self.adx_values


 
class aroon_oscillator():
    """
    Aroon Oscillator - odwzorowanie wskaźnika z TradingView (Pine Script v6):
 
        [aroonUp, aroonDn] = TVta.aroon(length)
        osc = aroonUp - aroonDn
 
    Formuły (zgodne z ta.aroon z biblioteki TradingView/ta):
        aroonUp = 100 * (length - bary_od_najwyższego_high) / length
        aroonDn = 100 * (length - bary_od_najniższego_low)  / length
    gdzie ekstrema szukane są w oknie `length + 1` świec (tak jak
    ta.highestbars(high, length + 1) w Pine).
 
    Wynik oscylatora mieści się w zakresie od -100 do 100.
    """
 
    def __init__(self, df: pd.DataFrame, aroon_length: int = 14, high: str = 'high', low: str = 'low') -> pd.Series:
        self.df = df
        self.aroon_length = aroon_length
        self.high = high
        self.low = low
 
    @staticmethod
    def _bars_since_max(values: np.ndarray) -> float:
        # Odwracamy okno, żeby przy remisie wybrać NAJNOWSZĄ świecę
        # (tak samo zachowuje się ta.highestbars w Pine Script).
        return float(values[::-1].argmax())
 
    @staticmethod
    def _bars_since_min(values: np.ndarray) -> float:
        return float(values[::-1].argmin())
 
    def calculate(self) -> pd.Series:
        length = self.aroon_length
        window = length + 1  # Pine: ta.highestbars(high, length + 1)
 
        bars_since_high = self.df[self.high].rolling(window).apply(self._bars_since_max, raw=True)
        bars_since_low = self.df[self.low].rolling(window).apply(self._bars_since_min, raw=True)
 
        self.aroon_up = 100 * (length - bars_since_high) / length
        self.aroon_down = 100 * (length - bars_since_low) / length
 
        # Oscylator = AroonUp - AroonDown (zakres -100..100)
        self.osc = self.aroon_up - self.aroon_down
        self.osc.name = f'aroon_osc_{length}'
 
        return self.osc


class parabolic_sar():
    """
    Parabolic SAR - odwzorowanie wbudowanego wskaźnika z TradingView (Pine Script v6):
 
        out = ta.sar(start, increment, maximum)
 
    Implementacja jest portem oficjalnego pseudokodu `ta.sar` z dokumentacji Pine,
    więc wartości (łącznie z momentami odwrócenia trendu i klampowaniem do low/high
    z 1-2 poprzednich świec) odpowiadają 1:1 temu, co rysuje TradingView.
 
    Zasada działania:
        - trend rosnący:  SAR = SAR_prev + AF * (EP - SAR_prev), rysowany POD ceną
        - trend malejący: SAR analogicznie, rysowany NAD ceną
        - EP (extreme point) = najwyższy high / najniższy low bieżącego trendu
        - AF (acceleration factor) startuje od `start`, rośnie o `increment`
          przy każdym nowym ekstremum, maksymalnie do `maximum`
        - przebicie SAR przez cenę odwraca trend i resetuje AF
 
    Zwraca pd.Series; pierwsza świeca ma wartość NaN (SAR startuje od drugiej,
    tak samo jak w Pine).
    """
 
    def __init__(self, df: pd.DataFrame, start: float = 0.02, increment: float = 0.02,
                 maximum: float = 0.2, high: str = 'high', low: str = 'low',
                 close: str = 'close') -> pd.Series:
        self.df = df
        self.start = start
        self.increment = increment
        self.maximum = maximum
        self.high = high
        self.low = low
        self.close = close
 
    def calculate(self) -> pd.Series:
        high = self.df[self.high].to_numpy(dtype='float64')
        low = self.df[self.low].to_numpy(dtype='float64')
        close = self.df[self.close].to_numpy(dtype='float64')
 
        n = len(close)
        sar = np.full(n, np.nan)
 
        if n < 2:
            self.sar_values = pd.Series(sar, index=self.df.index, name='parabolic_sar')
            return self.sar_values
 
        # --- inicjalizacja na drugiej świecy (bar_index == 1 w Pine) ---
        result = np.nan        # bieżąca wartość SAR
        max_min = np.nan       # EP - extreme point bieżącego trendu
        acceleration = np.nan  # AF - acceleration factor
        is_below = None        # True = SAR pod ceną (trend rosnący)
 
        if close[1] > close[0]:
            is_below = True
            max_min = high[1]
            result = low[0]
        else:
            is_below = False
            max_min = low[1]
            result = high[0]
        acceleration = self.start
        sar[1] = result
 
        # --- główna pętla, od trzeciej świecy ---
        for i in range(2, n):
            is_first_trend_bar = False
 
            # krok paraboliczny
            result = result + acceleration * (max_min - result)
 
            # sprawdzenie odwrócenia trendu (przebicie SAR przez cenę)
            if is_below:
                if result > low[i]:
                    is_first_trend_bar = True
                    is_below = False
                    result = max(high[i], max_min)
                    max_min = low[i]
                    acceleration = self.start
            else:
                if result < high[i]:
                    is_first_trend_bar = True
                    is_below = True
                    result = min(low[i], max_min)
                    max_min = high[i]
                    acceleration = self.start
 
            # aktualizacja EP i przyspieszenie AF (tylko gdy trend trwa)
            if not is_first_trend_bar:
                if is_below:
                    if high[i] > max_min:
                        max_min = high[i]
                        acceleration = min(acceleration + self.increment, self.maximum)
                else:
                    if low[i] < max_min:
                        max_min = low[i]
                        acceleration = min(acceleration + self.increment, self.maximum)
 
            # klampowanie: SAR nie może wejść w zakres 1-2 poprzednich świec
            if is_below:
                result = min(result, low[i - 1])
                result = min(result, low[i - 2])
            else:
                result = max(result, high[i - 1])
                result = max(result, high[i - 2])
 
            sar[i] = result
 
        self.sar_values = pd.Series(sar, index=self.df.index, name='parabolic_sar')
        return self.sar_values


# ---------------------------------------------------------------
# Fragment do wklejenia w tpi.py, jeśli chcesz podpiąć Supertrend
# pod TPI (sekcja "signal functions" + wpis w COMPONENTS).
# ---------------------------------------------------------------

class supertrend():
    """
    Supertrend - odwzorowanie wbudowanego wskaźnika z TradingView (Pine Script v6):

        [supertrend, direction] = ta.supertrend(factor, atrPeriod)

    Implementacja jest portem oficjalnego pseudokodu `ta.supertrend` z dokumentacji
    Pine, więc wartości linii oraz momenty odwrócenia trendu odpowiadają 1:1 temu,
    co rysuje TradingView.

    Zasada działania:
        - src = hl2 = (high + low) / 2
        - atr = ta.atr(atrPeriod)  -> wygładzanie Wildera (RMA) z True Range
        - upperBand = src + factor * atr,  lowerBand = src - factor * atr
        - pasma są "zapadkowe" (ratchet): lowerBand może tylko rosnąć,
          a upperBand tylko maleć, dopóki cena zamknięcia ich nie przebije
        - direction == -1  -> trend rosnący, linia = lowerBand (w Pine zielona)
          direction ==  1  -> trend malejący, linia = upperBand (w Pine czerwona)

    UWAGA co do konwencji znaku: w Pine `direction < 0` oznacza UPTREND
    (dlatego oryginał rysuje `direction < 0 ? supertrend : na` na zielono).
    Zachowuję tę samą konwencję, żeby wyniki dały się porównać z TradingView.

    Zwraca pd.Series z linią Supertrend; pierwsza świeca ma wartość NaN
    (odpowiednik `supertrend := barstate.isfirst ? na : supertrend`), a świece
    z okresu rozgrzewki ATR również są NaN (w Pine mają tam wartości śmieciowe
    wynikające z nz(...) == 0).

    Dodatkowe atrybuty po wywołaniu calculate():
        - self.supertrend_values : pd.Series - linia Supertrend
        - self.direction         : pd.Series - -1 (uptrend) / 1 (downtrend)
        - self.signal            : pd.Series -  1 (long)    / -1 (short)
        - self.up_trend          : pd.Series - linia tylko w trendzie rosnącym (reszta NaN)
        - self.down_trend        : pd.Series - linia tylko w trendzie malejącym (reszta NaN)
        - self.to_uptrend        : pd.Series[bool] - alert 'Downtrend to Uptrend'
        - self.to_downtrend      : pd.Series[bool] - alert 'Uptrend to Downtrend'
        - self.trend_change      : pd.Series[bool] - alert 'Trend Change'

    Elementy czysto wizualne z oryginału (plot.style_linebr, bodyMiddle,
    fill(...) dla tła trendu) nie mają odpowiednika w tej warstwie - do wykresu
    wystarczą serie self.up_trend / self.down_trend.
    """

    def __init__(self, df: pd.DataFrame, atr_period: int = 10, factor: float = 3.0,
                 high: str = 'high', low: str = 'low', close: str = 'close') -> pd.Series:
        self.df = df
        self.atr_period = max(int(atr_period), 1)      # Pine: minval = 1
        self.factor = max(float(factor), 0.01)         # Pine: minval = 0.01
        self.high = high
        self.low = low
        self.close = close

    def _true_range(self) -> np.ndarray:
        """
        True Range zgodny z ta.tr(true):
        na pierwszej świecy (brak close[1]) TR = high - low.
        """
        high = self.df[self.high].to_numpy(dtype='float64')
        low = self.df[self.low].to_numpy(dtype='float64')
        close = self.df[self.close].to_numpy(dtype='float64')

        prev_close = np.roll(close, 1)
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
        )
        if len(tr) > 0:
            tr[0] = high[0] - low[0]
        return tr

    def _atr(self, tr: np.ndarray) -> np.ndarray:
        """
        ta.atr = ta.rma(ta.tr(true), length):
        - pierwsza wartość (na indeksie length-1) = SMA z pierwszych `length` TR
        - kolejne = (TR + (length-1) * poprzednia) / length
        Świece przed rozgrzewką pozostają NaN (tak jak atr w Pine).
        """
        length = self.atr_period
        n = len(tr)
        atr = np.full(n, np.nan)

        if n < length:
            return atr

        atr[length - 1] = tr[:length].mean()
        for i in range(length, n):
            atr[i] = (tr[i] + (length - 1) * atr[i - 1]) / length
        return atr

    def calculate(self) -> pd.Series:
        df = self.df
        close = df[self.close].to_numpy(dtype='float64')
        src = ((df[self.high].to_numpy(dtype='float64')
                + df[self.low].to_numpy(dtype='float64')) / 2.0)     # hl2

        atr = self._atr(self._true_range())
        n = len(close)

        supertrend_line = np.full(n, np.nan)
        direction = np.full(n, np.nan)

        # nz(lowerBand[1]) / nz(upperBand[1]) w Pine dają 0 na pierwszej świecy
        prev_lower = 0.0
        prev_upper = 0.0
        prev_supertrend = np.nan

        for i in range(n):
            # surowe pasma; przy NaN w ATR porównania w Pine są fałszywe,
            # więc pasmo "dziedziczy" poprzednią wartość
            raw_upper = src[i] + self.factor * atr[i]
            raw_lower = src[i] - self.factor * atr[i]

            # zapadka: lowerBand rośnie, chyba że poprzednie zamknięcie ją przebiło
            if raw_lower > prev_lower or (i > 0 and close[i - 1] < prev_lower):
                lower = raw_lower
            else:
                lower = prev_lower

            # zapadka: upperBand maleje, chyba że poprzednie zamknięcie ją przebiło
            if raw_upper < prev_upper or (i > 0 and close[i - 1] > prev_upper):
                upper = raw_upper
            else:
                upper = prev_upper

            # NaN po prawej stronie porównania = warunek fałszywy (semantyka Pine)
            if np.isnan(lower):
                lower = prev_lower
            if np.isnan(upper):
                upper = prev_upper

            # kierunek trendu
            if i == 0 or np.isnan(atr[i - 1]):
                # w Pine: na(atr[1]) ? 1 : ...  -> start zawsze jako downtrend
                current_direction = 1.0
            elif prev_supertrend == prev_upper:
                current_direction = -1.0 if close[i] > upper else 1.0
            else:
                current_direction = 1.0 if close[i] < lower else -1.0

            current_supertrend = lower if current_direction == -1.0 else upper

            direction[i] = current_direction
            supertrend_line[i] = current_supertrend

            prev_lower = lower
            prev_upper = upper
            prev_supertrend = current_supertrend

        # supertrend := barstate.isfirst ? na : supertrend
        # + świece z rozgrzewki ATR (w Pine wychodzą z nz(...) == 0)
        supertrend_line[np.isnan(atr)] = np.nan
        if n > 0:
            supertrend_line[0] = np.nan

        index = df.index
        self.supertrend_values = pd.Series(supertrend_line, index=index, name='supertrend')
        self.direction = pd.Series(direction, index=index, name='supertrend_direction')

        # plot(direction < 0 ? supertrend : na) / plot(direction < 0 ? na : supertrend)
        self.up_trend = self.supertrend_values.where(self.direction < 0)
        self.down_trend = self.supertrend_values.where(self.direction >= 0)

        # sygnał w konwencji TPI: 1 = long, -1 = short
        self.signal = pd.Series(np.where(self.direction < 0, 1, -1), index=index,
                                name='supertrend_signal')

        # alertcondition(...)
        prev_direction = self.direction.shift(1)
        self.to_uptrend = prev_direction > self.direction     # Downtrend -> Uptrend
        self.to_downtrend = prev_direction < self.direction   # Uptrend -> Downtrend
        self.trend_change = prev_direction != self.direction

        return self.supertrend_values