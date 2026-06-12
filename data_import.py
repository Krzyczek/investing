import yfinance as yf

class data_importer():
    def __init__(self,ticker:str): 
        self.ticker = ticker
    def __str__(self):
        return f'what is the result? {self.ticker}'
    
    def import_ticker(self):
        df = yf.download(self.ticker,start='2018-01-01',interval='1d')
        df.columns = df.columns.droplevel("Ticker")
        df.columns = df.columns.str.lower()
        df['return'] = df['close'].pct_change(fill_method=None)
        self.df = df



    




