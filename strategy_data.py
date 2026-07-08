import buyhold_data
import data_import
import indicators
import investment_metrics as im
import plotly.subplots as ps
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import tpi
import optuna

class instrument_strategy():
    def __init__(self, instrument: str, instrument_type: str, deposit: int,risk_free_rate:float = 0.03):
        """Instrument input is a string format for yfinance ticker (work in progress)
        Instrument type is either stocks or crypto (at the moment)
        Deposit is the int number for your current cash reserves"""
        self.instrument = instrument
        self.deposit = deposit
        self.instrument_type = instrument_type
        self.risk_free_rate = risk_free_rate

    def strategy_evaluation(self):
        dataframe = data_import.data_importer(self.instrument)
        dataframe.import_yfinance_ticker()
        price = dataframe.df

        #price['return'] = price['close'].pct_change(fill_method=None)

        benchmark_metrics = buyhold_data.buyhold_benchmark(price, self.deposit, self.instrument_type, self.risk_free_rate)
        benchmark_returns = benchmark_metrics['return']
        
        def objective(trial):
            fast_ma_period = trial.suggest_int('fast_ma',5,100,step=1)
            slow_ma_period = trial.suggest_int('slow_ma',5,100,step=1)
            if fast_ma_period >= slow_ma_period:
                raise optuna.TrialPruned()
            adx_period = trial.suggest_int('adx_period',5,100,step=1)
            threshold = trial.suggest_int('threshold',5,50,step=1)
        
            test_df = price.copy()
            # --- OBLICZANIE SYGNAŁU (Z poprawkami znoszącymi wehikuł czasu) ---
            tpi_signal = tpi.tpi(test_df)
            tpi_signal.calculate_perpetual(slow_ma_period,fast_ma_period)
            tpi_signal.calculate_oscillator(adx_period,threshold)
            tpi_signal.calculate_tpi()
            # 1. Sygnał na koniec dzisiejszego dnia
            price['signal'] = tpi_signal.signal
            
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
            is_liquidated = np.where(price['equity'] < 0,1,0)
            print(price['equity'].min())
            print(is_liquidated.min())
            rect = True if is_liquidated.min() < 0 else False 
            # sygnał calculation finished
            if rect == True:
                raise optuna.TrialPruned()
            strat_metrics = im.metrics(df=price,investment_type = self.instrument_type,risk_free_rate = self.risk_free_rate,returns_column = 'strat_return',starting_equity=self.deposit,high='equity_high',low='equity_low',close='equity',verbose=False)
            sharpe = strat_metrics.sharpe_ratio()
            sortino = strat_metrics.sortino_ratio()
            omega = strat_metrics.omega_ratio()
            calmar = strat_metrics.calmar_ratio()
            alpha = strat_metrics.alpha(benchmark_returns)

           

            if sortino < 1:
                raise optuna.TrialPruned()
            if calmar < 0.5:
                raise optuna.TrialPruned()
            return sortino, calmar, alpha
        
        if __name__ == "__main__":
            # Opcja 'maximize' mówi Optunie, że im większy wynik z return, tym lepiej
            study = optuna.create_study(directions=['maximize','maximize','maximize'])
            
            print("Rozpoczynam poszukiwanie najlepszych parametrów...")
            study.optimize(objective, n_trials=5000, n_jobs=-1) # n_jobs=-1 używa wszystkich rdzeni procesora!
            
            print("\n--- ZAKOŃCZONO OPTYMALIZACJĘ ---")
            best = study.best_trials
            for i in best:
                print('Trial')
                print(i)
                print('End of trial')
            #print(f"Optymalne parametry: {study.best_params}")

        return price
        


strat_1 = instrument_strategy('BTC-USD', 'crypto', 12000,0.03)
equity_1 = strat_1.strategy_evaluation()
