# embedding encoding (deep neural network, cannot code by self, could only be implemented if present in a library)
# for hospital county, we can basically find the average cost of treatment for each hospital and then rank them on that basis for cardinal encoding and use linear regression
# CCSR Diagnosis code and description are redundant
# CCSR Procedure code and description are reduntant
# CCSR Diagn could be taken as square or cube to bias our model more towards it
# CCSR diag and proce ka koi function



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_average_cost(csv_file, category_column, cost_column):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Calculate the average cost for each category
    # df.groupby(category_column)[cost_column].shape
    average_cost = df.groupby(category_column)[cost_column].mean()
    average_cost = average_cost.sort_values()
    # Plot the average cost
    plt.figure(figsize=(10, 6))
    average_cost.plot(kind='bar', color='skyblue')
    plt.title(f'Average {cost_column} by {category_column}')
    plt.xlabel(category_column)
    plt.ylabel(f'Average {cost_column}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_outlier_analysis(csv_file, category_column, value_column):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Calculate the mean and standard deviation for the entire dataset
    mean_value = df[value_column].mean()
    std_dev = df[value_column].std()
    
    # Define the ranges for standard deviation categories
    df['Category'] = df[category_column]
    df['1-2 SD'] = ((df[value_column] > mean_value + std_dev) & (df[value_column] <= mean_value + 2 * std_dev)).astype(int)
    df['2-3 SD'] = ((df[value_column] > mean_value + 2 * std_dev) & (df[value_column] <= mean_value + 3 * std_dev)).astype(int)
    df['>3 SD'] = (df[value_column] > mean_value + 3 * std_dev).astype(int)
    
    # Group by the category column and sum the occurrences in each standard deviation range
    summary = df.groupby('Category')[['1-2 SD', '2-3 SD', '>3 SD']].sum()
    
    # Plot the results
    summary.plot(kind='bar', stacked=True, figsize=(12, 8), color=['lightblue', 'orange', 'red'])
    plt.title(f'Outlier Analysis for {value_column} by {category_column}')
    plt.xlabel(category_column)
    plt.ylabel('Number of Data Points')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Standard Deviation Range')
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_outlier_analysis('data.csv', 'Category', 'Value')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_outlier_analysis_by_category(csv_file, category_column, value_column):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Dictionary to store the summary data for each category
    summary_data = {'Category': [], '1-2 SD': [], '2-3 SD': [], '>3 SD': []}

    # Iterate over each unique category
    for category in df[category_column].unique():
        # Subset the DataFrame for the current category
        category_df = df[df[category_column] == category]
        
        # Calculate the mean and standard deviation for the current category
        mean_value = category_df[value_column].mean()
        std_dev = category_df[value_column].std()
        
        # Calculate the number of points within each standard deviation range
        count_1_2_sd = ((category_df[value_column] > mean_value + std_dev) & 
                        (category_df[value_column] <= mean_value + 2 * std_dev)).sum()
        count_2_3_sd = ((category_df[value_column] > mean_value + 2 * std_dev) & 
                        (category_df[value_column] <= mean_value + 3 * std_dev)).sum()
        count_above_3_sd = (category_df[value_column] > mean_value + 3 * std_dev).sum()
        
        # Store the results
        summary_data['Category'].append(category)
        summary_data['1-2 SD'].append(count_1_2_sd)
        summary_data['2-3 SD'].append(count_2_3_sd)
        summary_data['>3 SD'].append(count_above_3_sd)
    
    # Convert the summary data to a DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Plot the results
    summary_df.set_index('Category').plot(kind='bar', stacked=True, figsize=(12, 8), color=['lightblue', 'orange', 'red'])
    plt.title(f'Outlier Analysis by Category for {value_column}')
    plt.xlabel(category_column)
    plt.ylabel('Number of Data Points')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Standard Deviation Range')
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_outlier_analysis_by_category('data.csv', 'Category', 'Value')


# data = pd.read_csv('data/train.csv')
plot_average_cost('data/train.csv', 'Age Group', 'Total Costs')
# plot_outlier_analysis_by_category('data/train.csv', 'Hospital Service Area', 'Total Costs')
