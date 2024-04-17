import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


    for var in ['mood']: #, 'circumplex.arousal', 'circumplex.valence', 'screen', 'appCat.social']:
        fig, axes = plt.subplots(1,1)
        patient_properties.loc[f'mean {var} value'].plot(kind='hist', edgecolor='black', ax=axes)
        plt.title(f'variable: {var}')
        plt.show()

    # Convert 'time' column to datetime and extract date
    data['time'] = pd.to_datetime(dataset['time'])
    data['date'] = dataset['time'].dt.date
    for var in ['mood']: #, 'circumplex.arousal', 'circumplex.valence', 'activity', 'screen']:
        # Filter the data for 'mood' variable
        mood_data = data[data['variable'] == var]
        # Create a pivot table where each row corresponds to a date and each column corresponds to an ID
        pivot_table = mood_data.pivot_table(index='date', columns='id', values='value', aggfunc='count')
        # Create a DataFrame where each value is True if the 'mood' value is missing and False otherwise
        missing_mood = pivot_table.isnull()
        # Select some IDs to visualize
        selected_ids = list(data['id'].unique())  # replace with your selected IDs
        missing_mood_selected = missing_mood[selected_ids]
        # Create a heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(missing_mood_selected, cbar=False, cmap='viridis',mask=missing_mood)
        plt.title(f'Missing {var} Values for Selected IDs')
        plt.xlabel('ID')
        plt.ylabel('Date')
        plt.show()

    # Pie chart showing the proportion of records for each variable. Variable names are only listed in the seperate legend:
    total = data['variable'].value_counts().sum()
    percents = [s * 100 / total for l, s in data['variable'].value_counts().items()]
    fig, ax = plt.subplots()
    # increase figure size
    fig.set_size_inches(11, 6)
    data['variable'].value_counts().plot.pie(startangle=90, ax=ax, legend=False, labels=None)
    plt.title('Proportion of Records for Each Variable')
    ax.legend(['%s, %1.1f %%' % (l, s) for l, s in zip(data['variable'].value_counts().index, percents)], loc='center left', bbox_to_anchor=(1, 0.5))

    plt.ylabel('')
    plt.show()

    # Plotting frequency histograms for the 'mood' variable value for all id's
    # To do this, devide the ids into groups of 5
    # plot the histograms for each group in one plot, seperating ids in the same group by color
    # each seperate plot is put into the same figure that has 3 columns and 2 rows
    # The title of each plot should be the group number
    fig, axes = plt.subplots(2, 3)
    fig.set_size_inches(15, 10)
    for i, ax in enumerate(axes.flat):
        if i == 6:
            break
        ids = data['id'].unique()[i * 5:i * 5 + 5]
        mood_data = data[(data['variable'] == 'mood') & (data['id'].isin(ids))]
        mood_data.pivot_table(index='date', columns='id', values='value', aggfunc='mean').plot.hist(ax=ax, alpha=0.5, edgecolor='black')
        ax.set_title(f'Group {i + 1}')
        ax.set_xticks(np.arange(1, 10, 1))
        ax.set_yticks(np.arange(0, 30, 5))


        ax.legend(title='ID')
    plt.tight_layout()
    plt.show()



    return general_properties, patient_properties

visualize_data(dataset)
