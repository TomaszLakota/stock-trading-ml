import pandas as pd
from sklearn import preprocessing
import numpy as np

history_points = 50

def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    org = data.values
    print(org[0])
    data = data.drop('time', axis=1)
    # data = data.drop('bxbt', axis=1)
    # data = data.drop('pi', axis=1)
    # data = data.iloc[:, :-1] #drop last column = timestamp
    # data = data.drop('timestamp', axis=1)
    data = data.drop(0, axis=0)


    data = data.values
    print(data[0])

    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)

    ohlcv_histories_normalised = np.array([data_normalised[i:i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.array([data_normalised[:, 0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([data[:, 0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)

    return ohlcv_histories_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser