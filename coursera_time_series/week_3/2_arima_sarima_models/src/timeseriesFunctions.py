import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from dateutil.relativedelta import relativedelta

def plot_time_series(x,y, title= "Title", xlabel= "time", ylabel= "series"):
    plt.plot(x,y,'k-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha= 0.3)

def chunks_statistics(np_chunk_array):
    # np.split
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

def future_preds_df(model,series,num_months):
    """
    Generate a df with model predictions
    """
    pred_first = series.index.max()+relativedelta(months=1)
    pred_last = series.index.max()+relativedelta(months=num_months)
    date_range_index = pd.date_range(pred_first,pred_last,freq = 'MS')
    vals = model.predict(n_periods = num_months)
    return pd.DataFrame(vals,index = date_range_index)

def mape(df_cv):
    """
    MAPE: mean absolute percentage error
    """
    # return abs(df_cv.actual - df_cv.forecast).sum() / df_cv.actual.sum()
    return abs((df_cv.actual - df_cv.forecast)/df_cv.actual).sum() / len(df_cv)

def cross_validate(series, horizon, start, step_size, order= (0,0,0), seasonal_order= (0,0,0,0), trend= None):
    '''
    Function to determine in and out of samples for testing of ARIMA model
    ------
    arguments:
        series (series): time series input
        horizon (int): how far in advance forecast is needed
        start (int): starting location in series
        step_size (int): how often to recalculate forecast
        order (tuple): (p,d,q) order of the model
        seasonal_order (tuple): (P,D,Q,s) seasonal order of model
    Return
    ------
    Dataframe: gives forecast and actuals with date of prediction
    '''
    fcst= []
    actual= []
    date= []
    for i in range(start, len(series)- horizon, step_size):
        # print(i)
        # print(series[i])
        # print(series[:i+1])
        model= SARIMAX(series[:i+1], order= order, seasonal_order= seasonal_order, trend= trend).fit()
        fcst.append(model.forecast(steps= horizon)[-1])
        actual.append(series[i+horizon])
        date.append(series.index[i+horizon])
    return pd.DataFrame({'forecast': fcst, 'actual': actual}, index= date)

def grid_search_ARIMA(series, horizon, start, step_size, orders= [(1,0,0)], seasonal_orders= [(0,0,0,0)], trends= [None]):
    best_mape = np.inf
    best_order = None
    best_seasonal_order = None
    best_trend = None
    for order_ in orders:
        for seasonal_order_ in seasonal_orders:
            for trend_ in trends:
                cv = cross_validate(series,
                                    horizon,
                                    start,
                                    step_size,
                                    order = order_,
                                    seasonal_order = seasonal_order_,
                                    trend=trend_)
                if mape(cv)<best_mape:
                    best_mape = mape(cv)
                    best_order = order_
                    best_seasonal_order = seasonal_order_
                    best_trend = trend_
    return (best_order,best_seasonal_order, best_trend, np.round(best_mape, 6))
