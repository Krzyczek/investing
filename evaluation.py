import optuna_testing

instrument = int(input("""Select the class of instrument:)
                   1. stocks
                   2. crypto"""))
if instrument == 1:
    instrument = 'stocks'
elif instrument == 2:
    instrument = 'crypto'

starting_equity = int(input("What is your equity in $:"))
safe_investment = float(input("What is your yearly return for safe investment (e.g. bonds/savings):"))

strat_1 = optuna_testing.instrument_strategy('CDR.WA','stocks',12000,0.03)
equity_1 = strat_1.strategy_evaluation()


results_dataframe = equity_1.best_trials

