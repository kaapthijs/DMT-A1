import pandas as pd
from collections import OrderedDict
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


pd.set_option('future.no_silent_downcasting', True)
dataset = pd.read_csv('./cleaned_dataset.csv')
dataset['time'] = pd.to_datetime(dataset['time'], format='mixed')


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

    for id, data in df.sort_values('time').groupby('id'):
        # We can only compare prediction to days where a mood is recorded
        target_dates = list(OrderedDict.fromkeys([x.normalize() for x in data[data['variable'] == 'mood']['time']]))

        for date in target_dates:
            features = {}
            date_window = date + pd.Timedelta(days=window_size)  # window of [target_date - window_size, target_date + 1)
            if date_window > target_dates[-1]:
                break
            data_n_1 = data[(data['time'] >= date) & (data['time'] < date_window)]
            data_n = data[(data['time'] >= date_window) & (data['time'] < date_window + pd.Timedelta(days=1))]

            score_vars = ['mood', 'circumplex.arousal', 'circumplex.valence']
            for var in score_vars:
                # Mean of variable
                features[f"mean_{var}"] = -10 if data_n_1.loc[data_n_1['variable'] == var].empty else \
                    data_n_1.loc[data_n_1['variable'] == var]['value'].mean()
                # Mean of last 3 instances of var
                features[f"last3_mean_{var}"] = -10 if len(data_n_1.loc[data_n_1['variable'] == var]) < 3 else \
                    data_n_1.loc[data_n_1['variable'] == var]['value'].iloc[:3].fillna(0).mean()

            # for var in [x for x in list(df['variable'].drop_duplicates()) if x not in score_vars]:
            #     # Aggregating the variables
            #     features[f"aggr_{var}"] = 0 if data_n_1.loc[data_n_1['variable'] == var].empty else\
            #         data_n_1.loc[data_n_1['variable'] == var]['value'].sum()

            X.append(features)
            Y.append(data_n[data_n['variable'] == 'mood']['value'].mean())

    return X, Y


x, y = create_features(dataset, window_size=5)

x_train, x_test = [list(point.values()) for point in x[200:]], [list(point.values()) for point in x[:200]]
y_train, y_test = [round(point) for point in y[200:]], [round(point) for point in y[:200]]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
print(accuracy_score(y_test, clf.predict(x_test)))

y_train, y_test = [point for point in y[200:]], [point for point in y[:200]]
reg = LinearRegression().fit(x_train, y_train)
print(reg.score(x_train, y_train))
print(reg.score(x_test, y_test))
