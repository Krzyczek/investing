import evaluation


def selection():
    eval = evaluation.eval(12000)
    selection = evaluation.selected_test(eval)
    if selection is None or selection.empty:
        print("No valid data found in the selection.")
        return None
    else:
        print(selection)

        metric_columns = selection.columns.values
        selection = selection.drop_duplicates(subset=metric_columns)
        df_zscores = (selection[metric_columns] - selection[metric_columns].mean()) / selection[metric_columns].std()
        selection['avg_zscore'] = df_zscores.mean(axis=1)
        selection_best = selection.sort_values(by='avg_zscore', ascending=False)

        print(selection_best)
        print(selection_best.iloc[0:6])

selection()