import random

import pandas as pd
from collections import OrderedDict
from sklearn import tree
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def create_features(df, window_size=3):
    """
    Extract and create features using the target_dates (dates where 'mood' was measured)
    as target variables. Window_size amount of days is added to these dates, defining a window
    for the extraction/computation of predictor values.

    :param df:
    :param frame:
    :return:
    """

    df['time'] = pd.to_datetime(df['time'], format='mixed')
    score_vars = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    binary_vars = ['call', 'sms']
    apps = [x for x in list(df['variable'].drop_duplicates()) if x not in score_vars and x not in binary_vars]
    X = []
    Y = []

    for id, data in df.sort_values('time').groupby('id'):
        # We can only compare prediction to days where a mood is recorded
        target_dates = list(OrderedDict.fromkeys([x.normalize() for x in data[data['variable'] == 'mood']['time']]))
        for date in target_dates:
            features = {}
            date_window = date + pd.Timedelta(days=window_size)
            if date_window > target_dates[-1]:
                break
            # Splitting data according to time windows
            data_n_1 = data[(data['time'] >= date) & (data['time'] < date_window)]
            data_n = data[(data['time'] >= date_window) & (data['time'] < date_window + pd.Timedelta(days=1))]

            for var in score_vars:
                # Feature 1: mean of score variables
                features[f"mean_{var}"] = 0 if data_n_1.loc[data_n_1['variable'] == var].empty else \
                    data_n_1.loc[data_n_1['variable'] == var]['value'].mean()
                # Feature 2: standard deviation of score variables
                features[f"sd_{var}"] = 0 if data_n_1.loc[data_n_1['variable'] == var].empty else \
                    data_n_1.loc[data_n_1['variable'] == var]['value'].std(ddof=0)
                values = data_n_1.loc[data_n_1['variable'] == var]['value']
                # Feature 3: slopes of score variables
                features[f"slope_{var}"] = 0 if len(values) < 2 else \
                    np.polyfit(range(len(values)), values, 1)[0]
                # Feature 4: last known value of score variables
                features[f"last_{var}"] = 0 if data_n_1.loc[data_n_1['variable'] == var].empty else \
                    data_n_1.loc[data_n_1['variable'] == var].iloc[-1]['value']

            for var in binary_vars:
                # Feature 5: sum of binary variables
                features[f"sum_{var}"] = 0 if data_n_1.loc[data_n_1['variable'] == var].empty else \
                    data_n_1.loc[data_n_1['variable'] == var]['value'].sum()

            features[f"mean_apps"] = 0
            for var in apps:
                # Feature 6: summed means of all apps variables
                features[f"mean_apps"] += 0 if data_n_1.loc[data_n_1['variable'] == var].empty else \
                    data_n_1.loc[data_n_1['variable'] == var]['value'].mean()
                # Feature 7: last known value of all apps variables
                features[f"last_{var}"] = 0 if data_n_1.loc[data_n_1['variable'] == var].empty else \
                    data_n_1.loc[data_n_1['variable'] == var].iloc[-1]['value']

            X.append(features)
            Y.append(data_n[data_n['variable'] == 'mood']['value'].mean())

    return pd.DataFrame(X), pd.DataFrame(Y, columns=['mood'])

def select_features(X, y, k=5, cc=0.001):
    """
    Select relevant features, using Lasso and GridCV for alpha selection.
    Selects features that have |lasso coefficient| > cc
    Returns list of most relevant features
    :param X: features - pd.Dataframe
    :param y: Labels - pd.Dataframe
    :param k: amount of folds - int
    :param cc: coefficient cutoff - float
    :return:
    """

    # Defining alpha values
    alphas = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

    # Initializing KFold (k=10), Lasso, and GridSearchCV
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    lasso = Lasso()
    lasso_cv = GridSearchCV(lasso, param_grid=alphas, cv=kf)
    lasso_cv.fit(X, y)

    # Fitting Lasso with optimal alpha value
    lasso1 = Lasso(alpha=lasso_cv.best_params_['alpha'])
    lasso1.fit(X, y)

    # plotting the Column Names and Importance of Columns.
    lasso1_coef = np.abs(lasso1.coef_)
    plt.bar(list(X.keys()), lasso1_coef)
    plt.xticks(rotation=90)
    plt.grid()
    plt.title(f"Feature Selection Based on Lasso with alpha = {lasso_cv.best_params_['alpha']}")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    return np.array(list(X.keys()))[lasso1_coef > cc]


if __name__ == '__main__':
    SEED = np.random.randint(100)
    np.random.seed(SEED), random.seed(SEED)
    pd.set_option('future.no_silent_downcasting', True)
    dataset = pd.read_csv('./cleaned_dataset.csv')
    # Creating features
    X, y = create_features(dataset)
    # Splitting in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
    # Selecting optimal features
    features = select_features(X_train, y_train)
    print(f"SEED: {SEED}")
    print(features)
    X_train, X_test = X_train[features], X_test[features]

    # Training classifier
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train.round())
    print(accuracy_score(y_test.round(), clf.predict(X_test)))

    reg = LinearRegression().fit(X_train, y_train)
    print(reg.score(X_train, y_train))
