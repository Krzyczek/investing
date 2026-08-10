import pandas as pd
import numpy as np
import indicators
import data_import


class tpi():
    def __init__(self,df):
        self.df = df
        
    def calculate_perpetual(self,slow_ema_length: int, fast_ema_length:int):
        self.slow_ema_length = slow_ema_length
        self.fast_ema_length = fast_ema_length
        slow_ema = indicators.ema(self.df,self.slow_ema_length)
        fast_ema = indicators.ema(self.df,self.fast_ema_length)
        slow_ema.calculate()
        fast_ema.calculate()
        self.perpetual_signal = np.where(fast_ema.ema > slow_ema.ema,1,-1)

    def calculate_oscillator(self,adx_length:int,threshold: int,aroon_length: int):
        self.adx_length = adx_length
        self.threshold = threshold
        self.aroon_length = aroon_length
        adx = indicators.adx(self.df,self.adx_length)
        adx.calculate()
        aroon = indicators.aroon_oscillator(self.df,self.aroon_length)
        aroon.calculate()
        self.adx_signal = np.where(adx.adx_values > self.threshold,1,np.where(adx.adx_values < self.threshold,-1,0))
        self.aroon_signal = np.where(aroon.osc > 0,1,np.where(aroon.osc < 0,-1,0))
        self.oscillator_signal = (self.adx_signal + self.aroon_signal)/2


    def calculate_tpi(self,slow_ema_length: int, fast_ema_length: int, adx_length: int, threshold: int,aroon_length: int,mode : str = 'long_only'):
        self.calculate_perpetual(slow_ema_length,fast_ema_length)
        self.calculate_oscillator(adx_length,threshold,aroon_length) 
        tpi = (self.perpetual_signal + self.oscillator_signal)/2
        if mode == 'long_short':
            self.signal = np.where(tpi >= 0, 1, np.where(tpi < 0, -1,0))
        else:
            self.signal = np.where(tpi > 0, 1, 0)

