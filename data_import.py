import yfinance as yf
import pandas as pd
import os

class data_importer():
    def __init__(self,ticker:str): 
        self.ticker = ticker
    def __str__(self):
        return f'what is the result? {self.ticker}'
    
    def import_yfinance_ticker(self):
        df = yf.download(self.ticker, start='2018-01-01', interval='1d')
        # Zabezpieczenie przed błędem w yfinance przy pojedynczym tickerze
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1) # Bezpieczniejsze niż szukanie stringa "Ticker"
         
        df.columns = df.columns.str.lower()
        df['return'] = df['close'].pct_change(fill_method=None)
        self.df = df

    def import_csv_file(self):
        folder_path = "dane_cenowe"
        for i in os.listdir(folder_path):
            try: 
                if i.endswith('.csv'):
                    full_path = os.path.join(folder_path, i)        
                    df = pd.read_csv(full_path,parse_dates=True,index_col='time')
                    df['return'] = df['close'].pct_change(fill_method=None)
                    self.df = df

            except:
                print("No csv files found")
                self.df = pd.DataFrame()
            
    

        

    






