import pandas as pd
from collections import OrderedDict
import numpy as np
from math import gcd


pd.set_option('future.no_silent_downcasting', True)
dataset = pd.read_csv('./dataset_mood_smartphone.csv')
dataset['time'] = pd.to_datetime(dataset['time'])


def create_features(df, window_size=10):
    """
    Extract and create features using the target_dates (dates where 'mood' was measured)
    as target variables. Window_size amount of days is subtracted from these dates, defining a window
    for the extraction/computation of predictor values.

    :param df:
    :param frame:
    :return:
    """

    X = []
    Y = []
    variables = df['variable'].drop_duplicates()

    for id, data in df.drop(columns=df.columns[0]).sort_values('time').groupby('id'):
        target_dates = list(OrderedDict.fromkeys([x.normalize() for x in data[data['variable'] == 'mood']['time']]))

        for date in target_dates:
            features = {}
            date_window = date - pd.Timedelta(days=window_size - 1)
            data_9 = data[(data['time'] >= date_window) & (data['time'] < date)]
            data_10 = data[(data['time'] >= date) & (data['time'] < date + pd.Timedelta(days=1))]

            features['mean_mood'] = 0 if data_9[data_9['variable'] == 'mood'].empty else\
                data_9[data_9['variable'] == 'mood']['value'].mean()
            features['calls_sum'] = data_9[data_9['variable'] == 'call']['value'].sum()
            features['sms_sum'] = data_9[data_9['variable'] == 'call']['value'].sum()

            X.append(features)
            Y.append(data_10[data_10['variable'] == 'mood']['value'].mean())
    return X, Y


x, y = create_features(dataset, window_size=10)

