import numpy as np
import datetime as dt

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

scaler = MinMaxScaler(feature_range=(0, 1))


def get_predicted_val(model, data):
    val = data[-60:].values
    scaled_data = scaler.fit_transform(val)
    X_test = []
    X_test.append(scaled_data)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    return pred_price


def get_predicted_price(df, method):
    df = df.astype("float")
    data = df.to_dict("records")
    model = load_model(f"models/{method.lower()}_model.h5")
    df = pd.DataFrame.from_records(data)

    close = get_predicted_val(model, df.filter(['c']))
    open = get_predicted_val(model, df.filter(['o']))
    high = get_predicted_val(model, df.filter(['h']))
    low = get_predicted_val(model, df.filter(['l']))

    now = dt.datetime.now()
    next_timeframe_index = now + dt.timedelta(minutes=1)

    return close[0], open[0], high[0], low[0], [next_timeframe_index]
