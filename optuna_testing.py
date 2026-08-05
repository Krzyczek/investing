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

        benchmark_metrics = buyhold_data.buyhold_benchmark(price.loc['2018-01-01':], self.deposit, self.instrument_type, self.risk_free_rate)
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
            tpi_signal.calculate_tpi(slow_ma_period,fast_ma_period,adx_period,threshold,'long_short')
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
            rh = test_df['return_high'].fillna(0)
            rl = test_df['return_low'].fillna(0)
            
            conditions = [shifted_signal == 1, shifted_signal == -1]
            # best intraday outcome
            test_df['equity_high'] = np.select(conditions,
                [prev_equity * (1 + rh),      # long: high is best
                prev_equity * (1 - rl)],     # short: low is best
                default=prev_equity)
            
            # worst intraday outcome
            test_df['equity_low'] = np.select(conditions,
                [prev_equity * (1 + rl),      # long: low is worst
                prev_equity * (1 - rh)],     # short: high is worst
                default=prev_equity)
                     
            # 6. BEZPIECZNE wypełnianie braków (tylko dla kolumn kapitałowych!)
            test_df['equity'] = test_df['equity'].fillna(self.deposit)
            test_df['equity_high'] = test_df['equity_high'].fillna(self.deposit)
            test_df['equity_low'] = test_df['equity_low'].fillna(self.deposit)
            
            if (1 + test_df['strat_return'] <= 0).any():
                raise optuna.TrialPruned()   # strategy was liquidated on a short
            running_peak = test_df['equity'].cummax()
            max_drawdown = ((test_df['equity'] - running_peak) / running_peak).min()

            if max_drawdown < -0.60:   # reject strategies that lost >60% from peak
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
            study.optimize(objective, n_trials=10000, n_jobs=-1) # n_jobs=-1 używa wszystkich rdzeni procesora!
            self.study = study
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

    def best_of_best_selection(self,fast_ma,slow_ma,adx_period,threshold,mode):
        dataframe = data_import.data_importer(self.instrument)
        dataframe.import_csv_file()
        price = dataframe.df

        #indicator parameters - refer to indicators used in tpi
        self.fast_ma = fast_ma
        self.slow_ma=slow_ma
        self.adx_period = adx_period
        self.threshold = threshold
        self.mode = mode
        
        
        benchmark_metrics = buyhold_data.buyhold_benchmark(price.loc['2018-01-01':], self.deposit, self.instrument_type, self.risk_free_rate)
        self.benchmark_metrics = benchmark_metrics
        self.benchmark_returns = benchmark_metrics['return']
                
    
        
                
        test_df = price.copy()
        # --- OBLICZANIE SYGNAŁU (Z poprawkami znoszącymi wehikuł czasu) ---
        tpi_signal = tpi.tpi(test_df)
        #tpi_signal.calculate_perpetual(slow_ma_period,fast_ma_period)
        #tpi_signal.calculate_oscillator(adx_period,threshold)
        tpi_signal.calculate_tpi(self.slow_ma,self.fast_ma,self.adx_period,self.threshold,self.mode)
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
        rh = test_df['return_high'].fillna(0)
        rl = test_df['return_low'].fillna(0)

        conditions = [shifted_signal == 1, shifted_signal == -1]
        # best intraday outcome
        test_df['equity_high'] = np.select(conditions,
            [prev_equity * (1 + rh),      # long: high is best
            prev_equity * (1 - rl)],     # short: low is best
            default=prev_equity)

        # worst intraday outcome
        test_df['equity_low'] = np.select(conditions,
            [prev_equity * (1 + rl),      # long: low is worst
            prev_equity * (1 - rh)],     # short: high is worst
            default=prev_equity)
         
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
        

    

    def get_all_pareto_fronts(self, study):
        def dominates(t1: optuna.trial.FrozenTrial, t2: optuna.trial.FrozenTrial, directions):
                """Checks if t1 dominates t2 across all study objectives."""
                better_in_at_least_one = False
                for v1, v2, direction in zip(t1.values, t2.values, directions):
                    if direction == optuna.study.StudyDirection.MINIMIZE:
                        if v1 > v2:
                            return False
                        if v1 < v2:
                            better_in_at_least_one = True
                    else:  # MAXIMIZE
                        if v1 < v2:
                            return False
                        if v1 > v2:
                            better_in_at_least_one = True
                return better_in_at_least_one

        
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        # --- deduplicate by params ---
        seen = set()
        unique = []
        for t in completed:
            key = tuple(sorted(t.params.items()))  # dicts aren't hashable, tuples are
            if key not in seen:
                seen.add(key)
                unique.append(t)
        completed = unique
        # -----------------------------

        if not completed:
            return []

        S  = {t.number: [] for t in completed} #track dominated trials
        n  = {t.number: 0 for t in completed} #track domination counts
        fronts = [[]]

        for p in completed:
            for q in completed:
                if p.number == q.number:
                    continue
                if dominates(p, q, study.directions):
                    S[p.number].append(q)
                elif dominates(q, p, study.directions):
                    n[p.number] += 1

            if n[p.number] == 0:
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in S[p.number]:
                    n[q.number] -= 1
                    if n[q.number] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts


                  
        
    

