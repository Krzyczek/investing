import evaluation


def selection():
    instrument = int(input("""Select the class of instrument:)
                           1. stocks
                           2. crypto"""))
    if instrument == 1:
        instrument = 'stocks'
    elif instrument == 2:
        instrument = 'crypto'

    
    safe_investment = float(input("What is your yearly return for safe investment (e.g. bonds/savings):"))
        
    eval = evaluation.eval(12000, instrument, safe_investment)
    selection = evaluation.selected_test(eval, instrument)
    if selection is None or selection.empty:
        print("No valid data found in the selection.")
        return None
    else:
        print(selection)

        metric_columns = ['sharpe','sortino','omega','calmar','alpha']
        selection = selection.drop_duplicates(subset=metric_columns)
        df_zscores = (selection[metric_columns] - selection[metric_columns].mean()) / selection[metric_columns].std()
        selection['avg_zscore'] = df_zscores.mean(axis=1)
        selection_best = selection.sort_values(by='avg_zscore', ascending=False)

        print(selection_best)
        print(selection_best.iloc[0:6])

selection()