import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('./dataset_mood_smartphone.csv')

def remove_outliers(df, variable):
    """
    Removes outliers from a DataFrame using the Interquartile Range (IQR) method.

    Parameters:
        df (DataFrame): The pandas DataFrame containing the data.
        column (str): The name of the column from which outliers will be removed.

    Returns:
        DataFrame: A DataFrame with outliers removed based on the specified column.
    """
    # Calculate the IQR for the specified variable
    Q1 = df[df['variable'] == variable]['value'].quantile(0.25)
    Q3 = df[df['variable'] == variable]['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Replace outliers with NaN
    df.loc[(df['variable'] == variable) & ((df['value'] < lower_bound) | (df['value'] > upper_bound)), 'value'] = np.nan
    
    return df

def clean_dataset(df):
    # List of variables to be cleaned
    variables = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen',
                 'call', 'sms', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment',
                 'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social',
                 'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']
    
    # Remove outliers for each variable
    for variable in variables:
        df = remove_outliers(df, variable)
    
    # Impute missing values for each variable
    # Here we choose forward fill imputation for time series data
    #for variable in variables:
        #df = impute_missing_values_forward_fill(df, variable)
    
    return df

df_cleaned = clean_dataset(df)

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv('cleaned_dataset.csv', index=False)

print("Success")
