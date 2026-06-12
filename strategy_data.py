import buyhold_data
import data_import
import indicators
import investment_metrics as im
import plotly.subplots as ps
import plotly.graph_objects as go
import numpy as np
import pandas as pd

class instrument_strategy():
    def __init__(self, instrument: str, instrument_type: str, deposit: int):
        """Instrument input is a string format for yfinance ticker (work in progress)
        Instrument type is either stocks or crypto (at the moment)
        Deposit is the int number for your current cash reserves"""
        self.instrument = instrument
        self.deposit = deposit
        self.instrument_type = instrument_type

    def strategy_evaluation(self):
        dataframe = data_import.data_importer(self.instrument)
        dataframe.import_ticker()
        price = dataframe.df

        price['return'] = price['close'].pct_change(fill_method=None)

        benchmark_metrics = buyhold_data.buyhold_benchmark(price, self.deposit, self.instrument_type, 0.05)
        benchmark_returns = benchmark_metrics['return']
        
        
        # --- OBLICZANIE SYGNAŁU (Z poprawkami znoszącymi wehikuł czasu) ---
        price['fast_ema'] = indicators.ema(price, 9)
        price['slow_ema'] = indicators.ema(price, 45)
        
        # 1. Sygnał na koniec dzisiejszego dnia
        price['signal'] = np.where(price['fast_ema'] > price['slow_ema'], 1, 0)
        
        # 2. PRZESUNIĘCIE SYGNAŁU (Likwidacja wehikułu czasu)
        shifted_signal = price['signal'].shift(1).fillna(0)
        
        # 3. Zyski i Kapitał
        price['strat_return'] = price['return'] * shifted_signal
        price['equity'] = self.deposit * (1 + price['strat_return']).cumprod()
        
        # 4. Zwroty wewnątrzdzienne (High/Low)
        price['return_high'] = (price['high'] - price['close'].shift(1)) / price['close'].shift(1)
        price['return_low'] = (price['low'] - price['close'].shift(1)) / price['close'].shift(1)
        
        # 5. Kapitał High/Low z użyciem prawidłowego (przesuniętego) sygnału
        prev_equity = price['equity'].shift(1).fillna(self.deposit)
        price['equity_high'] = np.where(shifted_signal == 1, prev_equity * (1 + price['return_high']).fillna(0), prev_equity)
        price['equity_low'] = np.where(shifted_signal == 1, prev_equity * (1 + price['return_low']).fillna(0), prev_equity)
        
        # 6. BEZPIECZNE wypełnianie braków (tylko dla kolumn kapitałowych!)
        price['equity'] = price['equity'].fillna(self.deposit)
        price['equity_high'] = price['equity_high'].fillna(self.deposit)
        price['equity_low'] = price['equity_low'].fillna(self.deposit)
        # sygnał calculation finished

        strat_metrics = im.metrics(price, self.instrument_type, 0.05,'strat_return',self.deposit, 'equity_high', 'equity_low', 'equity')
        strat_metrics.sharpe_ratio()
        strat_metrics.sortino_ratio()
        strat_metrics.omega_ratio()
        strat_metrics.calmar_ratio()
        strat_metrics.alpha(benchmark_returns)

        return price
        
strat_1 = instrument_strategy('BTC-USD', 'crypto', 12000)
equity_1 = strat_1.strategy_evaluation()

#Przypisujemy 4 zwrócone serie do 4 osobnych kolumn w DataFrame (Zakomentowane)
equity_1['first_b'], equity_1['second_b'], equity_1['first_n'], equity_1['second_n'] = indicators.standard_deviation_bands(equity_1['fast_ema'], 200)

fig = ps.make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        row_heights=[0.6, 0.4], 
        vertical_spacing=0.05
    )

# 2. Główny wykres świecowy Instrumentu (Wiersz 1)
fig.add_trace(go.Candlestick(
    x = equity_1.index,
    open = equity_1['open'],
    close = equity_1['close'],
    low = equity_1['low'],
    high = equity_1['high'],
    name = 'Instrument Price',
    ), row=1, col=1)

# 3. Wykres świecowy Kapitału Strategii (Wiersz 2)
fig.add_trace(go.Candlestick(
    x = equity_1.index,
    open = equity_1['equity'].shift(1).fillna(strat_1.deposit),
    close = equity_1['equity'],
    low = equity_1['equity_low'],
    high = equity_1['equity_high'],
    name = 'Strategy Equity',
    ), row=2, col=1)

# --- ZAKOMENTOWANE FRAGMENTY DO PÓŹNIEJSZEGO UŻYCIA ---

# fig.add_trace(go.Scatter(
#     x = equity_1.index,
#     y = equity_1['fast_ema'],
#     line = dict(color='gray',width=1),
#     name = 'EMA',
# ), row=1, col=1)

# fig.add_trace(go.Scatter(
#     x= equity_1.index,
#     y=equity_1['equity'],
#     name = 'Equity'),row=2,col=1
# )

# # 3. Lista z 4 nowymi kolumnami, które chcesz narysować
# nowe_linie = ['first_b','second_b','first_n','second_n']

# # 4. Automatyczne dodawanie 4 linii w pętli
# for kolumna in nowe_linie:
#     fig.add_trace(go.Scatter(
#         x = equity_1.index,
#         y = equity_1[kolumna],
#         name = kolumna.upper()
#     ), row=1, col=1)


# # 1. Sprawdzamy, gdzie sygnał jest dodatni
# is_positive = equity_1['signal'] > 0

# # 2. Grupujemy dni w ciągłe bloki (żeby nie rysować tysięcy cienkich pasków)
# bloki = (is_positive != is_positive.shift()).cumsum()

# # 3. Iterujemy przez grupy
# for _, grupa in equity_1.groupby(bloki):
    
#     # 4. Sprawdzamy, czy ten konkretny blok to sygnał dodatni (True) czy ujemny/zerowy (False)
#     stan = is_positive.loc[grupa.index[0]]
    
#     start_date = grupa.index[0]
#     end_date = grupa.index[-1]
    
#     if start_date != end_date:
#         # 5. Wybieramy kolor w zależności od stanu
#         kolor_tla = "green" if stan else "red"
        
#         fig.add_vrect(
#             x0=start_date, 
#             x1=end_date,
#             fillcolor=kolor_tla, 
#             opacity=0.15,   
#             layer="below",      
#             line_width=0,       
#             row=1, col=1    
#         )

# --- KONIEC ZAKOMENTOWANYCH FRAGMENTÓW ---

#Kosmetyka i poprawa czytelności
fig.update_layout(
    height=800, # Zwiększamy nieco wysokość całego obrazka, by zmieścić dwa panele
    hovermode='x unified',
    # Wyłączenie "suwaka" (rangeslidera) na obu wykresach. 
    # Przy subplotach domyślny suwak Plotly potrafi całkowicie zepsuć układ.
    xaxis_rangeslider_visible=False,
    xaxis2_rangeslider_visible=False 
)

fig.show()