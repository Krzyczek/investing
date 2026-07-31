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
        dataframe.import_csv_file()
        price = dataframe.df

        #price['return'] = price['close'].pct_change(fill_method=None)

        benchmark_metrics = buyhold_data.buyhold_benchmark(price, self.deposit, self.instrument_type, self.risk_free_rate)
        self.benchmark_metrics = benchmark_metrics
        self.benchmark_returns = benchmark_metrics['return']
        
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
            #tpi_signal.calculate_perpetual(slow_ma_period,fast_ma_period)
            #tpi_signal.calculate_oscillator(adx_period,threshold)
            tpi_signal.calculate_tpi(slow_ma_period,fast_ma_period,adx_period,threshold)
            # 1. Sygnał na koniec dzisiejszego dnia
            test_df['signal'] = tpi_signal.signal
            test_df = test_df.loc[test_df.index >= '2018-01-01']
            
            # 2. PRZESUNIĘCIE SYGNAŁU (Likwidacja wehikułu czasu)
            shifted_signal = test_df['signal'].shift(1).fillna(0)
            
            # 3. Zyski i Kapitał
            test_df['strat_return'] = test_df['return'] * shifted_signal
            test_df['equity'] = self.deposit * (1 + test_df['strat_return']).cumprod()
            
            # 4. Zwroty wewnątrzdzienne (High/Low)
            test_df['return_high'] = (test_df['high'] - test_df['close'].shift(1)) / test_df['close'].shift(1)
            test_df['return_low'] = (test_df['low'] - test_df['close'].shift(1)) / test_df['close'].shift(1)
            
            # 5. Kapitał High/Low z użyciem prawidłowego (przesuniętego) sygnału
            prev_equity = test_df['equity'].shift(1).fillna(self.deposit)
            test_df['equity_high'] = np.where(shifted_signal == 1, prev_equity * (1 + test_df['return_high'].fillna(0)), prev_equity)
            test_df['equity_low'] = np.where(shifted_signal == 1, prev_equity * (1 + test_df['return_low'].fillna(0)), prev_equity)
            
            # 6. BEZPIECZNE wypełnianie braków (tylko dla kolumn kapitałowych!)
            test_df['equity'] = test_df['equity'].fillna(self.deposit)
            test_df['equity_high'] = test_df['equity_high'].fillna(self.deposit)
            test_df['equity_low'] = test_df['equity_low'].fillna(self.deposit)
            is_liquidated = np.where(test_df['equity'] < 0,1,0)
            print(test_df['equity'].min())
            print(is_liquidated.min())
            rect = True if is_liquidated.max() == 1 else False 
            # sygnał calculation finished
            if rect == True:
                raise optuna.TrialPruned()
            strat_metrics = im.metrics(df=test_df,investment_type = self.instrument_type,risk_free_rate = self.risk_free_rate,returns_column = 'strat_return',starting_equity=self.deposit,high='equity_high',low='equity_low',close='equity',verbose=False)
            sharpe = strat_metrics.sharpe_ratio()
            sortino = strat_metrics.sortino_ratio()
            omega = strat_metrics.omega_ratio()
            calmar = strat_metrics.calmar_ratio()
            alpha = strat_metrics.alpha(self.benchmark_returns)

           

            
            return sortino, calmar, alpha
        
        if __name__ != "__main__":
            # Opcja 'maximize' mówi Optunie, że im większy wynik z return, tym lepiej
            study = optuna.create_study(directions=['maximize','maximize','maximize'])
            
            print("Rozpoczynam poszukiwanie najlepszych parametrów...")
            study.optimize(objective, n_trials=500, n_jobs=-1) # n_jobs=-1 używa wszystkich rdzeni procesora!
            
            print("\n--- ZAKOŃCZONO OPTYMALIZACJĘ ---")
            best = study.best_trials
            lista=[]
            for trial in best:
                    parametry = trial.params
                    lista.append({'parametry':parametry})
            if not lista:
                lista=[{'parametry':0}]
                
            #print(f"Optymalne parametry: {study.best_params}")
            df_best_trials = pd.DataFrame(lista)
            self.best_trials = df_best_trials

    def best_of_best_selection(self,fast_ma,slow_ma,adx_period,threshold):
        dataframe = data_import.data_importer(self.instrument)
        dataframe.import_csv_file()
        price = dataframe.df

        #indicator parameters - refer to indicators used in tpi
        self.fast_ma = fast_ma
        self.slow_ma=slow_ma
        self.adx_period = adx_period
        self.threshold = threshold
        
        
        benchmark_metrics = buyhold_data.buyhold_benchmark(price.loc['2018-01-01':], self.deposit, self.instrument_type, self.risk_free_rate)
        self.benchmark_metrics = benchmark_metrics
        self.benchmark_returns = benchmark_metrics['return']
                
    
        
                
        test_df = price.copy()
        # --- OBLICZANIE SYGNAŁU (Z poprawkami znoszącymi wehikuł czasu) ---
        tpi_signal = tpi.tpi(test_df)
        #tpi_signal.calculate_perpetual(slow_ma_period,fast_ma_period)
        #tpi_signal.calculate_oscillator(adx_period,threshold)
        tpi_signal.calculate_tpi(self.slow_ma,self.fast_ma,self.adx_period,self.threshold)
        # 1. Sygnał na koniec dzisiejszego dnia
        test_df['signal'] = tpi_signal.signal
        test_df = test_df.loc[test_df.index >= '2018-01-01']
                    
        # 2. PRZESUNIĘCIE SYGNAŁU (Likwidacja wehikułu czasu)
        shifted_signal = test_df['signal'].shift(1).fillna(0)
                    
        # 3. Zyski i Kapitał
        test_df['strat_return'] = test_df['return'] * shifted_signal
        test_df['equity'] = self.deposit * (1 + test_df['strat_return']).cumprod()
                    
        # 4. Zwroty wewnątrzdzienne (High/Low)
        test_df['return_high'] = (test_df['high'] - test_df['close'].shift(1)) / test_df['close'].shift(1)
        test_df['return_low'] = (test_df['low'] - test_df['close'].shift(1)) / test_df['close'].shift(1)
                    
        # 5. Kapitał High/Low z użyciem prawidłowego (przesuniętego) sygnału
        prev_equity = test_df['equity'].shift(1).fillna(self.deposit)
        test_df['equity_high'] = np.where(shifted_signal == 1, prev_equity * (1 + test_df['return_high'].fillna(0)), prev_equity)
        test_df['equity_low'] = np.where(shifted_signal == 1, prev_equity * (1 + test_df['return_low'].fillna(0)), prev_equity)
                    
        # 6. BEZPIECZNE wypełnianie braków (tylko dla kolumn kapitałowych!)
        test_df['equity'] = test_df['equity'].fillna(self.deposit)
        test_df['equity_high'] = test_df['equity_high'].fillna(self.deposit)
        test_df['equity_low'] = test_df['equity_low'].fillna(self.deposit)
       
        strat_metrics = im.metrics(df=test_df,investment_type = self.instrument_type,risk_free_rate = self.risk_free_rate,returns_column = 'strat_return',starting_equity=self.deposit,high='equity_high',low='equity_low',close='equity',verbose=False)
        sharpe = strat_metrics.sharpe_ratio()
        sortino = strat_metrics.sortino_ratio()
        omega = strat_metrics.omega_ratio()
        calmar = strat_metrics.calmar_ratio()
        alpha = strat_metrics.alpha(self.benchmark_returns)
        dict = {'sharpe': sharpe,
                'sortino':sortino,
                'omega':omega,
                'calmar':calmar,
                'alpha':alpha}
        return pd.Series(dict)
        
                   
        
    

