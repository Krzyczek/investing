import investment_metrics as im
import pandas as pd




def buyhold_benchmark(df: pd.DataFrame,starting_equity: float,instrument_type : str,risk_free_investment: float = 0.04):
    
    

    price = df.copy()
    price['bh_equity'] = starting_equity * (1+price['return']).cumprod()
    price['bh_return_high'] = (price['high'] - price['close'].shift(1))/price['close'].shift(1)
    price['bh_return_low'] = (price['low'] - price['close'].shift(1))/price['close'].shift(1)
    prev_equity = price['bh_equity'].shift(1).fillna(starting_equity)
    price['bh_equity_high'] = prev_equity * (1+price['bh_return_high']).fillna(0)
    price['bh_equity_low'] = prev_equity * (1+price['bh_return_low']).fillna(0)



    metrics = im.metrics(df=price,investment_type = instrument_type,risk_free_rate = risk_free_investment,returns_column = 'return',high='bh_equity_high',low='bh_equity_low',close='bh_equity')
    sharpe = metrics.sharpe_ratio()
    sortino = metrics.sortino_ratio()
    omega = metrics.omega_ratio()
    calmar = metrics.calmar_ratio()
    b_return = metrics.alpha()
   

    benchmark_dict = {'sharpe': sharpe.astype(float),
                      'sortino': sortino,
                      'omega': omega,
                      'calmar': calmar,
                      'return':b_return}

    
    return {k: float(v) for k, v in benchmark_dict.items()}