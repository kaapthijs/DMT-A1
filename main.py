import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('./dataset_mood_smartphone.csv')
# data['time'] = pd.to_datetime(data['time'])  # Converting time strings into datatime objects


def visualize_data(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Gives a comprehensive overview of the dataset_mood_smartphone dataset.
    :param data:
    :return: None
    """

    # Printing general properties of the dataset
    # Unnamed: 0 (index: int), id (string), time (string), variable (string), value(float)
    print("number of records: ", len(data))
    print("number of users: ", len(data['id'].drop_duplicates()))
    print("number of recorded variables: ", len(data['variable'].drop_duplicates()))
    print(f"total timespan: {(pd.to_datetime(data['time'].max()) - pd.to_datetime(data['time'].min())).days} days")
    print(f"null values: ")
    print(data.isnull().sum())  # Only value has null values

    general_properties, patient_properties = pd.DataFrame(), pd.DataFrame()
    general_properties.loc["number of records", 'max'] = len(data)
    general_properties.loc[f"column: time", 'min'] = data['time'].min()
    general_properties.loc[f"column: time", 'max'] = data['time'].max()

    for x in data['variable'].drop_duplicates():
        general_properties.loc[f"variable: {x}", 'min'] = data.loc[data['variable'] == x]['value'].min()
        general_properties.loc[f"variable: {x}", 'max'] = data.loc[data['variable'] == x]['value'].max()
        general_properties.loc[f"variable: {x}", 'mean'] = data.loc[data['variable'] == x]['value'].mean()

    for id, p_data in data.groupby('id'):
        # Patient properties, statistics (min, max, mean, enz) for each id
        for var in p_data['variable'].drop_duplicates():
            patient_properties.loc[f"min {var} value", id] = p_data.loc[p_data['variable'] == var]['value'].min()
            patient_properties.loc[f"max {var} value", id] = p_data.loc[p_data['variable'] == var]['value'].max()
            patient_properties.loc[f"mean {var} value", id] = p_data.loc[p_data['variable'] == var]['value'].mean()

    for var in ['mood', 'circumplex.arousal', 'circumplex.valence', 'screen', 'appCat.social']:
        fig, axes = plt.subplots(1,1)
        patient_properties.loc[f'mean {var} value'].plot(kind='hist', edgecolor='black', ax=axes)
        plt.title(f'variable: {var}')
        plt.show()

    grouped_data = data[data['variable'] == 'mood'].groupby('id')

    # Calculate the number of subplots needed
    n, ncols = 6, 2  # number of columns for subplots
    nrows = np.ceil(n / ncols).astype(int)  # number of rows for subplots

    # Create a new figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 5))  # adjust the size as needed
    axes = axes.flatten()  # flatten the axes array

    # Loop through each group
    for i, (id, group) in enumerate(grouped_data):
        # Convert the 'time' column to datetime
        group['time'] = pd.to_datetime(group['time'], format='mixed')

        # Sort the data by 'time'
        group = group.sort_values('time')

        # Plot the histogram of 'value' for each group on a separate subplot
        axes[i // 5].hist(group['value'], edgecolor='black', label=f'{id}', alpha=0.5)

        # Add a legend to each subplot
        axes[i // 5].legend()
        axes[i // 5].set_xlabel('mood')
        axes[i // 5].set_ylabel('Frequency')
        axes[i // 5].set_title(f"distributions of mood values for group {i // 5}")

    # Remove unused subplots
    for j in range(i // 5 + 1, len(axes)):
        fig.delaxes(axes[j])

    # Show the plot
    plt.tight_layout()
    plt.show()

    return general_properties, patient_properties

visualize_data(dataset)
