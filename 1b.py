import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

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


df_filtered = remove_outliers_iqr(df, 'value')
df_imputed_median = impute_missing_values(df_filtered, method='median')
df_cleaned = df_imputed_median

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv('cleaned_dataset.csv', index=False)

print("Success")
