import optuna_testing
import pandas as pd
from optuna_testing import instrument_strategy as istr

def eval(deposit : int):
    instrument = int(input("""Select the class of instrument:)
                       1. stocks
                       2. crypto"""))
    if instrument == 1:
        instrument = 'stocks'
    elif instrument == 2:
        instrument = 'crypto'

    starting_equity = deposit
    safe_investment = float(input("What is your yearly return for safe investment (e.g. bonds/savings):"))

    strat_1 = optuna_testing.instrument_strategy('CDR.WA','stocks',starting_equity,safe_investment)
    strat_1.strategy_evaluation()


    trials_dataframe = strat_1.best_trials


    return trials_dataframe

evaluation = eval(12000)
all_tested_series = []
for index,item in evaluation.iterrows():
    test_parameter = item['parametry']
    tested = istr('tested','stocks',12000).best_of_best_selection(test_parameter['fast_ma'],test_parameter['slow_ma'],test_parameter['adx_period'],test_parameter['threshold'])
    all_tested_series.append(tested)
selected = pd.DataFrame(all_tested_series)

print(selected)
    