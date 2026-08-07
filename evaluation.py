import optuna_testing
import pandas as pd
from optuna_testing import instrument_strategy as istr
import buyhold_data
import data_import
import tpi
import numpy as np
import investment_metrics as im


def eval(deposit : int,instrument,safe_investment:float = 0.03):
    
    strat_1 = optuna_testing.instrument_strategy('CDR.WA',instrument,deposit,safe_investment)
    strat_1.strategy_evaluation()
    

    study = strat_1.study

    pareto_fronts = strat_1.get_all_pareto_fronts(study)

    #temporary solution until robustness test is implemented
    
                
    return pareto_fronts
    #------------------------#
    
def parameter_robustness_test(deposit : int,instrument,pareto_fronts,safe_investment:float = 0.00,mode:str = 'long_short'):
    dataframe = data_import.data_importer(instrument)
    dataframe.import_csv_file()
    price = dataframe.df
    for layer_idx, front in enumerate(pareto_fronts, start=1):
        if layer_idx > 10:
            break
        print(
            f"""Evaluating Front {layer_idx} containing {len(front)} candidates..."""
        )
        base_metrics = {}
        for i in range(len(front)):
            
            params = front[i].params
            fast_ma = params['fast_ma']
            slow_ma = params['slow_ma']
            adx_period = params['adx_period']
            threshold = params['threshold']

            #Strat testing
            benchmark_metrics = buyhold_data.buyhold_benchmark(price.loc['2018-01-01':], deposit, instrument, safe_investment)
            benchmark_returns = benchmark_metrics['return']
                    
        
            
                    
            test_df = price.copy()
            # --- OBLICZANIE SYGNAŁU (Z poprawkami znoszącymi wehikuł czasu) ---
            tpi_signal = tpi.tpi(test_df)
            #tpi_signal.calculate_perpetual(slow_ma_period,fast_ma_period)
            #tpi_signal.calculate_oscillator(adx_period,threshold)
            tpi_signal.calculate_tpi(slow_ma,fast_ma,adx_period,threshold,mode)
            # 1. Sygnał na koniec dzisiejszego dnia
            test_df['signal'] = tpi_signal.signal
            test_df = test_df.loc[test_df.index >= '2018-01-01']
                        
            # 2. PRZESUNIĘCIE SYGNAŁU (Likwidacja wehikułu czasu)
            shifted_signal = test_df['signal'].shift(1).fillna(0)
                        
            # 3. Zyski i Kapitał
            test_df['strat_return'] = test_df['return'] * shifted_signal
            test_df['equity'] = deposit * (1 + test_df['strat_return']).cumprod()
                        
            # 4. Zwroty wewnątrzdzienne (High/Low)
            test_df['return_high'] = (test_df['high'] - test_df['close'].shift(1)) / test_df['close'].shift(1)
            test_df['return_low'] = (test_df['low'] - test_df['close'].shift(1)) / test_df['close'].shift(1)
                        
            # 5. Kapitał High/Low z użyciem prawidłowego (przesuniętego) sygnału
            prev_equity = test_df['equity'].shift(1).fillna(deposit)
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
            test_df['equity'] = test_df['equity'].fillna(deposit)
            test_df['equity_high'] = test_df['equity_high'].fillna(deposit)
            test_df['equity_low'] = test_df['equity_low'].fillna(deposit)


        
            strat_metrics = im.metrics(df=test_df,investment_type = instrument,risk_free_rate = safe_investment,returns_column = 'strat_return',starting_equity=deposit,high='equity_high',low='equity_low',close='equity',verbose=False)
            sharpe = strat_metrics.sharpe_ratio()
            sortino = strat_metrics.sortino_ratio()
            omega = strat_metrics.omega_ratio()
            calmar = strat_metrics.calmar_ratio()
            alpha = strat_metrics.alpha(benchmark_returns)
            candidate_metrics = {'sharpe': sharpe,
                    'sortino':sortino,
                    'omega':omega,
                    'calmar':calmar,
                    'alpha':alpha}
            if not base_metrics:
                base_metrics = candidate_metrics
                base_candidate = i+1
                print(base_metrics)
                continue

            for key, base_value in base_metrics.items():
                if key in candidate_metrics:
                    if candidate_metrics[key] < base_value:
                        print(f"Candidate {i+1} on Front {layer_idx} has worse {key} "
                              f"({candidate_metrics[key]:.4f}) than the base candidate ({base_value:.4f}).")
                    elif candidate_metrics[key] > base_value:
                        print(f"Candidate {i+1} on Front {layer_idx} has better {key} "
                              f"({candidate_metrics[key]:.4f}) than the base candidate ({base_value:.4f}).")
                    else:
                        print(f"Candidate {i+1} on Front {layer_idx} has equal {key} "
                              f"({candidate_metrics[key]:.4f}) to the base candidate.")

                else:
                    print(f"Metric '{key}' not found in candidate metrics.")


            key_metrics = ('sharpe', 'sortino', 'calmar')
            any_key_better = any(candidate_metrics[k] > base_metrics[k] for k in key_metrics)
            none_worse = all(candidate_metrics[k] >= base_metrics[k] for k in base_metrics)

            if any_key_better and none_worse:
                base_metrics = candidate_metrics
                base_candidate = i + 1
                print(f"--> Candidate {i+1} on Front {layer_idx} becomes the new base candidate.")
                    

        
    
pareto = eval(12000,'crypto',0)
parameter_robustness_test(12000, 'crypto', pareto, 0)