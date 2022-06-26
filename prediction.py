import math
import pandas as pd
import numpy as np
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

scaler = MinMaxScaler(feature_range=(0, 1))


def get_valid_array(df, method):
    model = load_model(f"models/{method.lower()}_model.h5")
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
    actual = data[training_data_len:]

    d = {"Close": predicted_stock_price.flatten()}
    predicted_stock_price = pd.DataFrame(data=d)
    predicted_stock_price.index = actual.index

    return train, actual, predicted_stock_price


def get_predicted_price(code, method, feature):
    model = load_model(f"models/{method.lower()}_model.h5")
    quote = web.DataReader(
        code, data_source='yahoo', start='2018-01-01', end=dt.datetime.now().strftime("%Y-%m-%d"),
    )
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

    train, actual, predicted = get_valid_array(quote, method)

    today = dt.date.today()
    today = dt.datetime(today.year, today.month, today.day)
    today = today + dt.timedelta(days=1)

    d = {"Close": [pred_price.item()]}
    pred_price_ser = pd.DataFrame(
        data=d, index=[today]
    )

    final_predicted_price_data = pd.concat([predicted, pred_price_ser])

    return train, actual, final_predicted_price_data
