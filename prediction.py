
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
plt.style.use('fivethirtyeight')

scaler = MinMaxScaler(feature_range=(0, 1))


def get_valid_array(df):
    model = load_model("saved_rnn.h5")
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len, :]
    test_data = scaled_data[training_data_len - 60:, :]

    X_test = []
    Y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predicted_stock_price

    return train, valid


def get_predicted_price(code):
    model = load_model("saved_rnn.h5")
    quote = web.DataReader(
        code, data_source='yahoo', start='2018-01-01', end=dt.datetime.now().strftime("%Y-%m-%d"))
    # create a new dataframe
    new_df = quote.filter(['Close'])

    # get the last 60 day closing price values and convert the dataframe to an array
    last_60_days = new_df[-60:].values

    # scale the data to be values between 0 and 1
    last_60_days_scaled = scaler.fit_transform(last_60_days)
    # create an empty list
    X_test = []
    # append the past 60 days
    X_test.append(last_60_days_scaled)
    # convert the X_test data to a numpy array
    X_test = np.array(X_test)
    # reshape
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # get the predicted scaled price
    pred_price = model.predict(X_test)
    # undo the scaling
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price)

    train, valid = get_valid_array(quote)

    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

    return train, valid, pred_price
