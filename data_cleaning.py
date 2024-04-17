import pandas as pd
import numpy as np


def remove_outliers(df, variable):
    # Calculate the IQR for the specified variable
    Q1 = df[df['variable'] == variable]['value'].quantile(0.25)
    Q3 = df[df['variable'] == variable]['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Replace outliers with NaN
    # df.loc[(df['variable'] == variable) & ((df['value'] < lower_bound) | (df['value'] > upper_bound)), 'value'] = np.nan

    # Remove outliers
    i = df.loc[(df['variable'] == variable) & ((df['value'] < lower_bound) | (df['value'] > upper_bound))].index
    df = df.drop(i)
    return df


# Function to impute missing values using forward fill imputation for time series
def impute_missing_values_forward_fill(df, variable):
    df.loc[df['variable'] == variable, 'value'] = df[df['variable'] == variable]['value'].fillna(method='ffill')
    return df


# Function to impute missing values using interpolation for time series
def impute_missing_values_interpolation(df, variable):
    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'])
    # Set 'time' column as index
    df.set_index('time', inplace=True)
    # Interpolate missing values
    df.loc[df['variable'] == variable, 'value'] = df[df['variable'] == variable]['value'].interpolate(method='time')
    # Reset index
    df.reset_index(inplace=True)
    return df


# Function to impute missing 'mood' measurements for days with missing data
def impute_missing_mood(df, timestamp_col):
    # Ensure the timestamp column is in datetime format
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.drop(columns='Unnamed: 0')

    # Set the index to the timestamp column
    df.set_index(timestamp_col, inplace=True)

    # Group by id and Date, and check if 'mood' is missing for any day
    missing_mood_days = df[df['variable'] == 'mood'].groupby('id').resample('D').first().isnull()
    df.reset_index(inplace=True)

    # For each id, find the days where 'mood' is missing and impute using forward fill
    for id in missing_mood_days.index.get_level_values(0).unique():
        missing_days = missing_mood_days.loc[id][missing_mood_days.loc[id]['value']]
        for day in missing_days.index:
            # Forward fill the missing 'mood' value for the day
            day_data = df.loc[(df['id'] == id) & (df['time'] == day)]
            df.loc[len(df)] = {'time': day.date(), 'id': id, 'variable': 'mood', 'value': np.nan}
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    impute_missing_values_interpolation(df, 'mood')

    # Reset the index
    df.reset_index(inplace=True)

    return df


def clean_dataset(df, imputation_method, timestamp_col):
    # List of variables to be cleaned
    vars_score = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    vars_binary = ['call', 'sms'] # We don't use these
    variables = ['screen', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment',
                 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social',
                 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']

    # Remove outliers for time variables
    for variable in variables + vars_score:
        df = remove_outliers(df, variable)

    # Impute missing values for each variable
    for variable in variables + vars_score:
        if imputation_method == 'forward_fill':
            df = impute_missing_values_forward_fill(df, variable)
        elif imputation_method == 'interpolation':
            df = impute_missing_values_interpolation(df, variable)

    # Impute missing 'mood' measurements for days with missing data
    df = impute_missing_mood(df, timestamp_col)

    return df


if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('./dataset_mood_smartphone.csv')
    df_cleaned = clean_dataset(df, 'interpolation', 'time')
    # Save the cleaned dataset to a new CSV file
    df_cleaned.to_csv('cleaned_dataset.csv', index=False)
    print("Success")