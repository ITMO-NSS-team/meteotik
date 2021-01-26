import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt


def convert_degrees_to_float(degrees: int = 0, minutes: int = 0,
                             seconds: int = 0) -> float:
    """
    The function converts degrees / minutes / seconds to float

    :param degrees: the degrees of latitude or longitude
    :param minutes: the minutes of latitude or longitude
    :param seconds: the seconds of latitude or longitude

    :return: converted to float value
    """

    # Fraction of seconds in minutes
    seconds_part = seconds/60.0
    minutes = minutes + seconds_part

    # Fraction of minutes in degrees
    minutes_part = minutes/60.0
    result = degrees + minutes_part

    return result


def convert_float_to_degrees(value: float):
    """
    The function converts float to degrees / minutes

    :param value: float value of coordinates

    :return: converted to float value
    TODO add seconds determination
    """

    # Integer part - is degree
    degrees = int(value)
    resid = value - degrees

    minutes = int(round(resid * 60))
    dict_coord = {'degrees':degrees,
                  'minutes':minutes}

    return dict_coord


def equal_mapping(dataframe_left, dataframe_right, merge_column:str = 'Date'):
    """ The function equalizes the dataframes for the selected column

    :param dataframe_left: (meteo stations) dataframe with columns for merging
    :param dataframe_right: (reanalysis) dataframe with columns for merging
    :param merge_column: column name on which dataframes will merged
    """

    left_columns = list(dataframe_left.columns)
    right_columns = list(dataframe_right.columns)
    merged_df = dataframe_left.merge(dataframe_right, on=merge_column)

    new_dataframe_left = merged_df[left_columns]
    new_dataframe_right = merged_df[right_columns]
    return new_dataframe_left, new_dataframe_right


def bias_calculate(real, predict):
    """ Function for calculating bias """
    real = np.ravel(np.array(real))
    predict = np.ravel(np.array(predict))
    bias_arr = predict - real
    return np.mean(bias_arr)


def print_statistics(dataframe_left: pd.DataFrame,
                     dataframe_right: pd.DataFrame,
                     columns_for_compare: dict):
    """
    The function displays some metrics for the compared dataframes

    :param dataframe_left: (meteo stations) dataframe with columns for comparison
    :param dataframe_right: (reanalysis) dataframe with columns for comparison
    :param columns_for_compare: dictionary where keys are column names in
    dataframe_left and values are column names in dataframe_right, which should be
    compared
    """

    # Get all keys (column names) for compare
    column_left_names = list(columns_for_compare.keys())

    for column_left_name in column_left_names:
        column_right_name = columns_for_compare.get(column_left_name)

        column_left = np.array(dataframe_left[column_left_name])
        column_right = np.array(dataframe_right[column_right_name])

        bias = bias_calculate(column_left, column_right)
        mae = mean_absolute_error(column_left, column_right)
        korr = stats.pearsonr(column_left, column_right)

        print(f'Bias for {column_left_name} vs {column_right_name}: {bias:.2f}')
        print(f'MAE for {column_left_name} vs {column_right_name}: {mae:.2f}')
        print(f'Pearson correlation coefficient for {column_left_name} vs '
              f'{column_right_name}: {korr[0]:.2f}\n')


def make_visual_comparison(dataframe_left: pd.DataFrame,
                           dataframe_right: pd.DataFrame,
                           columns_for_compare: dict,
                           dataframe_labels: list = ('Left Dataframe', 'Right Dataframe'),
                           x_min: str = None, x_max: str = None):
    """
    The function displays plots for analytics. Dataframe "dataframe_left"
    should contain 'Date' column with datetime

    :param dataframe_left: (meteo stations) dataframe with columns for comparison
    :param dataframe_right: (reanalysis) dataframe with columns for comparison
    :param columns_for_compare: dictionary where keys are column names in
    dataframe_left and values are column names in dataframe_right, which should be
    compared
    :param dataframe_labels: labels for dataframes
    :param x_min: min x border for visualisation as string in format "%Y-%m-%d"
    :param x_max: max x border for visualisation as string in format "%Y-%m-%d"
    """

    if x_min is None and x_max is None:
        x_min_date = min(dataframe_left['Date'])
        x_max_date = max(dataframe_left['Date'])
    else:
        x_min_date = datetime.datetime.strptime(x_min, "%Y-%m-%d")
        x_max_date = datetime.datetime.strptime(x_max, "%Y-%m-%d")

    # Get all keys (column names) for compare
    column_left_names = list(columns_for_compare.keys())

    for column_left_name in column_left_names:
        column_right_name = columns_for_compare.get(column_left_name)

        column_left = np.array(dataframe_left[column_left_name])
        column_right = np.array(dataframe_right[column_right_name])

        # Comparative plot
        plt.plot(dataframe_left['Date'], column_left,
                 label=dataframe_labels[0], alpha=0.8, c='blue')
        plt.plot(dataframe_left['Date'], column_right,
                 label=dataframe_labels[1], alpha=0.8, c='orange')
        plt.ylabel(f'{column_left_name} vs {column_right_name}', fontsize=12)
        plt.xlabel('Date', fontsize=15)
        plt.legend(fontsize=15)
        plt.grid()
        plt.xlim(x_min_date, x_max_date)
        plt.show()

        # Разница между действительным (метеостанция) и предсказанным (реанализ)
        diff_arr = column_left - column_right
        plt.scatter(dataframe_left['Date'], diff_arr, alpha=0.8, c='green')
        plt.ylabel(f'{column_left_name} vs {column_right_name}', fontsize=12)
        plt.xlabel('Date', fontsize=15)
        plt.xlim(x_min_date, x_max_date)
        plt.title(f'{dataframe_labels[0]} - {dataframe_labels[1]}', fontsize=15)
        plt.grid()
        plt.show()

        with sns.axes_style("whitegrid"):
            sns.kdeplot(np.ravel(diff_arr), shade=False, color="green",
                        kernel='gau', alpha=0.8, linewidth=3)
            plt.hist(np.ravel(diff_arr), 70, density=True, color='green',
                     alpha=0.2)
            plt.xlabel(f'{dataframe_labels[0]} - {dataframe_labels[1]}', fontsize=15)
            plt.ylabel('Probability density function', fontsize=15)
            plt.show()
