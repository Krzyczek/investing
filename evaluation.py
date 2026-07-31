import optuna_testing
import pandas as pd
from optuna_testing import instrument_strategy as istr

def eval(deposit : int,instrument,safe_investment:float = 0.03):
    

    starting_equity = deposit

    strat_1 = optuna_testing.instrument_strategy('CDR.WA',instrument,starting_equity,safe_investment)
    strat_1.strategy_evaluation()


    trials_dataframe = strat_1.best_trials


    return trials_dataframe



def selected_test(evaluation,instrument):
    all_tested_series = []
    for index,item in evaluation.iterrows():
        
        test_parameter = item['parametry']
        try:
            tested = istr('tested',instrument,12000).best_of_best_selection(test_parameter['fast_ma'],test_parameter['slow_ma'],test_parameter['adx_period'],test_parameter['threshold'])
        
            combined_data = { **tested.to_dict(), **test_parameter}
            all_tested_series.append(combined_data)
            
            
       
        except Exception as e:
            print(f"No values found in test_parameter: {test_parameter}. Error: {e}")
            continue
    print("Successfully completed all tests. Returning the selected DataFrame.")    
    selected = pd.DataFrame(all_tested_series)
    return selected

    