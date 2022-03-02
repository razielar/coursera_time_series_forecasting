# Functions to train a RNN for time-series
import numpy as np


def get_n_last_days(df, series_name, n_days):
    """
    Extract last n_days of an hourly time series
    """
    return df[series_name][-(24*n_days):]

def get_keras_format_series(series, features= 1):
    """
    Convert a series to a numpy array of shape [n_samples, time_steps, features]
    """
    series= np.array(series)
    return series.reshape(series.shape[0], series.shape[1], features)

def get_train_test_data(df, series_name, series_days, input_hours, test_hours, sample_gap= 3, verbose= True):
    """
    Function that splits an hourly time series into train and test with keras format. 
    Arguments
    ---------
    df (pandas df) => df with time series columns
    series_name (str) => column name in df
    series_days (int) => total days to extract
    input_hours (int) => length of sequence input to NN
    test_hours (int) => lengh of held-out terminal sequence
    sample_gap (int) => step size between start of train sequence
    
    return: (train_X, test_X_init, train_y, test_y)
    """
    forecast_series= get_n_last_days(df, series_name, n_days= series_days).values
    train = forecast_series[:-test_hours] # training data is remaining days until amount of test_hours
    test = forecast_series[-test_hours:] # test data is the remaining test_hours
    
    train_X, train_y= [], []
    for i in range(0, train.shape[0]-input_hours, sample_gap):
        # print(i)
        # print(train_X)
        # print(train_y)
        train_X.append(train[i:i+input_hours]) # each training sample is of length input hours
        train_y.append(train[i+input_hours]) # each y is just the next step after training sample
        
    train_X = get_keras_format_series(train_X) # format our new training set to keras format
    train_y = np.array(train_y) # make sure y is an array to work properly with keras
    
    test_X_init = test[:input_hours] 
    test_y = test[input_hours:] 
    
    if verbose:
        print("Train shape: {} Percentage: {:.4f}".format(len(train), len(train)/(series_days*24) ))
        print("Test shape: {} Percentage: {:.4f}".format(len(test), len(test)/(series_days*24)))
    
    return train_X, test_X_init, train_y, test_y

