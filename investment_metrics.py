import pandas as pd
import numpy as np

class metrics():
    def __init__(self,df: pd.DataFrame,investment_type: str,risk_free_rate: float,returns_column: str = 'return',starting_equity: float = 1000000,high: str = 'high',low:str = 'low',close: str='close',verbose:bool = False):
        self.df = df
        self.returns = df[returns_column].fillna(0)
        self.close = df[close]
        self.high = df[high]
        self.low = df[low]
        self.investment_type = investment_type
        self.risk_free_rate = risk_free_rate
        self.verbose = verbose
        if self.investment_type == 'crypto':
            self.annualization = 365
        else:
            self.annualization = 252
        self.starting_equity = starting_equity
    def sharpe_ratio(self) -> float:
        '''
        returns is a pandas Series containing returns in a decimal format
        investment_type is the type of investment to consider:
        'stocks' or 'crypto'. That value will impact the annualization parameter
        risk_free_rate is the % rate (in decimal format) of a safe investment (obligations or savings), default is 0.04'''
    
        

        if self.returns.std() == 0:
            return 0.0
        annualized_returns = self.returns.mean() * self.annualization
        annualized_std = self.returns.std()*np.sqrt(self.annualization)
        sharpe = round((annualized_returns - self.risk_free_rate)/annualized_std,6)
        if self.verbose == True:
            print('-'*30)
            print(f'Sharpe: {sharpe}')
        return sharpe

    


    def sortino_ratio(self) -> float:
        '''
        returns is a pandas Series containing returns in a decimal format
        investment_type is the type of investment to consider:
        'stocks' or 'crypto'. That value will impact the annualization parameter
        risk_free_rate is the % rate (in decimal format) of a safe investment (obligations or savings), default is 0.04'''


        annualized_returns = self.returns.mean() * self.annualization
        daily_mar = self.risk_free_rate/self.annualization
        downside_deviation = np.where(self.returns < daily_mar, self.returns - daily_mar,0.0)
        daily_downside_deviation = np.sqrt(np.mean(downside_deviation**2))
        annualized_downside_deviation = daily_downside_deviation * np.sqrt(self.annualization)

        
        if annualized_downside_deviation == 0:
                if self.returns.abs().sum() == 0:
                    return np.nan                      # no activity at all
                excess = annualized_returns - self.risk_free_rate
                return np.inf if excess > 0 else np.nan  # perfect strategy → best score
        
        return round((annualized_returns - self.risk_free_rate) / annualized_downside_deviation, 6)


    def omega_ratio(self) -> float:
        returns_array= np.asarray(self.returns,dtype=float)
    

        daily_mar = self.risk_free_rate/self.annualization
        excess_returns = returns_array - daily_mar
        upside_sum = np.sum(excess_returns[excess_returns > 0])
        downside_sum = np.abs(np.sum(excess_returns[excess_returns <= 0]))
        if downside_sum == 0:
            return np.inf if upside_sum > 0 else np.nan
        omega = round(upside_sum/downside_sum,6)
        if self.verbose == True:
            print(f'Omega: {omega}')
        return omega
    
    def calmar_ratio(self) -> float:
        """
        Oblicza Calmar Ratio uwzględniając wewnątrzdzienne ekstrema (High i Low).
        Zakłada, że DataFrame posiada kolumny 'close', 'high' oraz 'low'.
        """
        if self.close.iloc[-1] <= 0:
            return np.nan
        # --- 1. ROCZNA STOPA ZWROTU (CAGR) ---
        # Liczymy na podstawie cen zamknięcia (Close)
        
        total_years = len(self.df) / self.annualization
        total_return = self.close.iloc[-1] / self.starting_equity
        # Wzór na CAGR: (Kapitał_Końcowy / Kapitał_Początkowy) ^ (1 / Lata) - 1
        cagr = (total_return ** (1 / total_years)) - 1
        
        # --- 2. PESYMISTYCZNY MAX DRAWDOWN ---
        # cummax() zapamiętuje najwyższą wartość, jaką cena (High) osiągnęła do danego dnia
        rolling_peaks = self.high.cummax()
        
        # Mierzymy odległość od historycznego szczytu do dzisiejszego wewnątrzdziennego dołka (Low)
        drawdowns = (self.low - rolling_peaks) / rolling_peaks
        
        # Max DD to najgorszy (najniższy) wynik z całej historii
        max_dd = drawdowns.min() 
        
        # --- 3. WSKAŹNIK CALMARA ---
        # Zabezpieczenie przed dzieleniem przez zero (np. przy idealnie płaskim wykresie)
        if max_dd == 0:
            return np.nan
            
        # Dzielimy zysk przez wartość absolutną Max DD (żeby wynik był dodatni)
        calmar_ratio = cagr / abs(max_dd)
        
        # Ładne drukowanie wyników
        if self.verbose == True:
            print(f"Średni roczny zysk (CAGR): {cagr * 100:.2f}%")
            print(f"Pesymistyczny Max DD:      {max_dd * 100:.2f}%")
            print(f"Calmar Ratio:              {calmar_ratio:.2f}")
        
        return calmar_ratio



    def total_return(self) -> float:
        returns_array = np.array(self.returns, dtype=float)
        total_growth = np.prod(1 + returns_array) - 1
        strategy_return = round(total_growth * 100, 3)
        if self.verbose:
            print(f'Return: {strategy_return}%')
        return strategy_return

    def alpha(self, benchmark_growth: float = 0) -> float:
        alpha = round(self.total_return() - benchmark_growth, 3)
        if self.verbose:
            print(f'Alpha: {alpha}%')
            print('-' * 30)
        return alpha
    

