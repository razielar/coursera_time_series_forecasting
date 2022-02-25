import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def plot_time_series(x,y, title= "Title", xlabel= "time", ylabel= "series"):
    plt.plot(x,y,'k-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha= 0.3)

def chunks_statistics(np_chunk_array):
    print("{} | {:7} | {}".format("Chunk", "Mean", "Variance"))
    print("-" * 26)
    for i,j in enumerate(np_chunk_array, 1):
        print("{:5} | {:.6} | {:.6}".format(i, np.mean(j), np.var(j)))

def get_years(timeseries):
    """
    Get the unique number of years from a time-series, under the assumption that the date is in the index
    Usage: get_years(df_ts)
    """
    years= np.unique(np.array(timeseries.index, dtype= 'datetime64[Y]'))
    print("Number of unique years: {}".format(len(years)))
    print('')
    print(years)

def dickey_fuller_test(timeseries):
    """
    NUll hypothesis= time-series is not stationary
    """
    test= adfuller(timeseries)
    df= pd.Series(test[0:4], index= ["Test-statistics", "p-value", "Lags-used", "Observation-used"])
    for key,value in test[4].items():
        df['Critical value (%s)'%key] = value    
    print(df)
    # Rolling
    rolmean= timeseries.rolling(window=12).mean()
    rolstd= timeseries.rolling(window=12).std()
    # Plotting
    original= plt.plot(timeseries, color= "blue", label= "Original Time-series")
    mean= plt.plot(rolmean, color= "red", label= "Rolling Mean")
    std= plt.plot(rolstd, color= "black", label= "Rolling Std")
    plt.legend(loc= "best")
    plt.grid()
    plt.show()
