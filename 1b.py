import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('./dataset_mood_smartphone.csv')

def remove_outliers_iqr(df, column):
    """
    Removes outliers from a DataFrame using the Interquartile Range (IQR) method.

    Parameters:
        df (DataFrame): The pandas DataFrame containing the data.
        column (str): The name of the column from which outliers will be removed.

    Returns:
        DataFrame: A DataFrame with outliers removed based on the specified column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df_filtered

def impute_missing_values(df, method='median', k=None):
    """
    Imputes missing values in a DataFrame using specified imputation method.

    Parameters:
        df (DataFrame): The pandas DataFrame containing the data.
        method (str): The imputation method to be used. Options: 'mean', 'median', 'knn'. Default is 'median'.
        k (int): Number of nearest neighbors to consider for KNN imputation. Required only if method='knn'.

    Returns:
        DataFrame: A DataFrame with missing values imputed based on the specified method.
    """
    # Separate the DataFrame into numeric and non-numeric data
    numeric_data = df.select_dtypes(include='number')
    non_numeric_data = df.select_dtypes(exclude='number')

    # Initialize imputers for numeric and non-numeric data
    if method == 'mean':
        imputer_numeric = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer_numeric = SimpleImputer(strategy='median')
    elif method == 'knn':
        if k is None:
            raise ValueError("Parameter 'k' must be provided for KNN imputation.")
        imputer_numeric = KNNImputer(n_neighbors=k)
    else:
        raise ValueError("Invalid imputation method. Choose from 'mean', 'median', or 'knn'.")

    imputer_non_numeric = SimpleImputer(strategy='most_frequent')

    # Perform imputation on numeric and non-numeric data
    numeric_data_imputed = pd.DataFrame(imputer_numeric.fit_transform(numeric_data), columns=numeric_data.columns)
    non_numeric_data_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(non_numeric_data), columns=non_numeric_data.columns)

    # Combine the imputed numeric and non-numeric data back into one DataFrame
    df_imputed = pd.concat([numeric_data_imputed, non_numeric_data_imputed], axis=1)

    # Reorder the columns to match the original DataFrame's order
    df_imputed = df_imputed[df.columns]

    return df_imputed

def impute_missing_values_time_series(df, time_column, variable_column, value_column, method='linear'):
    """
    Imputes missing values in a time series DataFrame using specified imputation method for numeric data
    and the most frequent value for categorical data.

    Parameters:
        df (DataFrame): The pandas DataFrame containing the time series data.
        time_column (str): The name of the column containing time data.
        variable_column (str): The name of the column containing categorical data.
        value_column (str): The name of the column containing the numeric values to impute.
        method (str): The imputation method to be used for numeric data. Options: 'locf', 'linear'. Default is 'linear'.

    Returns:
        DataFrame: A DataFrame with missing values imputed.
    """
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Ensure the time column is in datetime format
    df_copy[time_column] = pd.to_datetime(df_copy[time_column])

    # Sort the DataFrame by the time index
    df_copy.sort_values(time_column, inplace=True)

    # Impute categorical data with the most frequent value
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df_copy[variable_column] = imputer_categorical.fit_transform(df_copy[[variable_column]]).ravel()

    # Choose the imputation method for numeric data
    if method == 'locf':
        # Last Observation Carried Forward (LOCF)
        df_copy[value_column].fillna(method='ffill', inplace=True)
    elif method == 'linear':
        # Linear Interpolation
        df_copy[value_column].interpolate(method='linear', inplace=True)
    else:
        raise ValueError("Invalid imputation method. Choose from 'locf' or 'linear'.")

   # Reorder the columns to match the original DataFrame's order
    #df_copy = df_copy.reindex(columns=df.columns)

    # Sort the DataFrame by 'id' and 'time' to match the original order
    #df_copy.sort_values(by=[variable_column, time_column], inplace=True)

    return df_copy

def compare_imputation_methods(df, value_column, missing_rate=0.1, random_seed=0):
    """
    Compares linear interpolation and LOCF imputation methods on time series data.

    Parameters:
        df (DataFrame): The pandas DataFrame containing the time series data.
        value_column (str): The name of the column containing the numeric values to impute.
        missing_rate (float): The proportion of values to remove for creating artificial missing data.
        random_seed (int): The seed for the random number generator.

    Returns:
        dict: A dictionary containing the MSE for linear interpolation and LOCF.
    """
    # Introduce artificial missing values for testing
    np.random.seed(random_seed)
    missing_indices = np.random.choice(df.index, size=int(len(df) * missing_rate), replace=False)
    df_with_missing = df.copy()
    df_with_missing.loc[missing_indices, value_column] = np.nan

    # Split the data into two sets: one for training the imputation models and one for testing them
    train_df = df_with_missing.dropna(subset=[value_column])
    test_df = df_with_missing.loc[missing_indices]

    # Apply linear interpolation and LOCF to the training set
    train_df_linear = train_df.interpolate(method='linear')
    train_df_locf = train_df.fillna(method='ffill')

    # Apply the same methods to the test set
    test_df_linear = test_df.interpolate(method='linear')
    test_df_locf = test_df.fillna(method='ffill')

    # Calculate the mean squared error for both methods
    mse_linear = mean_squared_error(train_df[value_column], train_df_linear[value_column])
    mse_locf = mean_squared_error(train_df[value_column], train_df_locf[value_column])

    # Return the MSE of both methods
    return {
        'Linear Interpolation MSE': mse_linear,
        'LOCF MSE': mse_locf
    }

df_filtered = remove_outliers_iqr(df, 'value')

df_imputed_median = impute_missing_values(df_filtered, method='median')
df_imputed_linear = impute_missing_values_time_series(df_filtered, 'time', 'variable', 'value', method='linear')
df_imputed_locf = impute_missing_values_time_series(df_filtered, 'time', 'variable', 'value', method='locf')

results = compare_imputation_methods(df_filtered, 'value')
print(results)

df_cleaned = df_imputed_linear

# Save the cleaned dataset to a new CSV file
#df_cleaned.to_csv('cleaned_dataset.csv', index=False)

print("Success")
