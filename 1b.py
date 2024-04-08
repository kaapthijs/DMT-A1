import pandas as pd

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


df_filtered = remove_outliers_iqr(df, 'value')

df_cleaned = df_filtered

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv('cleaned_dataset.csv', index=False)

print("Success")
