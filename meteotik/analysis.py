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
                     columns_for_compare: dict,
                     check_peaks: bool = False):
    """
    The function displays some metrics for the compared dataframes

    :param dataframe_left: (meteo stations) dataframe with columns for comparison
    :param dataframe_right: (reanalysis) dataframe with columns for comparison
    :param columns_for_compare: dictionary where keys are column names in
    dataframe_left and values are column names in dataframe_right, which should be
    compared
    :param check_peaks: is there a need to calculate metrics for peak values
    (> 75 percentile)
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
              f'{column_right_name}: {korr[0]:.2f}')

        if check_peaks is True:
            # Get 75 percentile
            q75 = np.quantile(column_left, 0.75)
            ids = np.argwhere(column_left > q75)

            peak_left = np.ravel(column_left[ids])
            peak_right = np.ravel(column_right[ids])

            bias_peak = bias_calculate(peak_left, peak_right)
            mae_peak = mean_absolute_error(peak_left, peak_right)
            korr_peak = stats.pearsonr(peak_left, peak_right)

            print(f'(Peak) Bias for {column_left_name} vs {column_right_name}: {bias_peak:.2f}')
            print(f'(Peak) MAE for {column_left_name} vs {column_right_name}: {mae_peak:.2f}')
            print(f'(Peak) Pearson correlation coefficient for {column_left_name} vs '
                  f'{column_right_name}: {korr_peak[0]:.2f}')
        print('\n')


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

        # Calculate errors and plot them
        diff_arr = column_left - column_right
        plt.scatter(dataframe_left['Date'], diff_arr, alpha=0.8, c='green')
        plt.ylabel(f'{column_left_name} vs {column_right_name}', fontsize=12)
        plt.xlabel('Date', fontsize=15)
        plt.xlim(x_min_date, x_max_date)
        plt.title(f'{dataframe_labels[0]} - {dataframe_labels[1]}', fontsize=15)
        plt.grid()
        plt.show()

        with sns.axes_style("whitegrid"):
            k = int(len(diff_arr)**0.5)
            sns.kdeplot(np.ravel(diff_arr), shade=False, color="green",
                        kernel='gau', alpha=0.8, linewidth=3)
            plt.hist(np.ravel(diff_arr), k, density=True, color='green',
                     alpha=0.2)
            plt.xlabel(f'{dataframe_labels[0]} - {dataframe_labels[1]}', fontsize=15)
            plt.ylabel('Probability density function', fontsize=15)
            plt.show()

        print('\n')


def qq_comparison(dataframe_left: pd.DataFrame,
                  dataframe_right: pd.DataFrame,
                  columns_for_compare: dict,
                  dataframe_labels: list = ('Left Dataframe', 'Right Dataframe'),
                  check_peaks: bool = False):
    """
    The function allows pairwise matching of values in the form of quantile biplots

    :param dataframe_left: (meteo stations) dataframe with columns for comparison
    :param dataframe_right: (reanalysis) dataframe with columns for comparison
    :param columns_for_compare: dictionary where keys are column names in
    dataframe_left and values are column names in dataframe_right, which should be
    compared
    :param dataframe_labels: labels for dataframes
    :param check_peaks: is there a need to calculate metrics for peak values
    (> 75 percentile)
    """

    # Get all keys (column names) for compare
    column_left_names = list(columns_for_compare.keys())
    for column_left_name in column_left_names:
        column_right_name = columns_for_compare.get(column_left_name)

        column_left = np.array(dataframe_left[column_left_name])
        column_right = np.array(dataframe_right[column_right_name])

        title = ''.join((column_left_name, ' vs ', column_right_name))
        _plot_qq(array_x=column_right, array_y=column_left,
                 name=title, x_label=dataframe_labels[1],
                 y_label=dataframe_labels[0])

        if check_peaks is True:
            # Get 75 percentile
            q75 = np.quantile(column_left, 0.75)
            ids = np.argwhere(column_left > q75)

            peak_left = np.ravel(column_left[ids])
            peak_right = np.ravel(column_right[ids])
            title = ''.join(('(Peak) ', title))
            _plot_qq(array_x=peak_right, array_y=peak_left,
                     name=title, x_label=dataframe_labels[1],
                     y_label=dataframe_labels[0])


def _plot_qq(array_x, array_y, name, x_label, y_label) -> None:
    """
    A function to draw the quantile biplots

    :param array_x: array which will be on x-axis
    :param array_y: array which will be on y-axis
    :param name: title of the plot
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    """

    percs = np.arange(0, 100)
    qn_x = np.percentile(array_x, percs)
    qn_y = np.percentile(array_y, percs)

    # Creating a quantile biplot
    plt.figure()

    min_qn = np.min([qn_x.min(), qn_y.min()])
    max_qn = np.max([qn_x.max(), qn_y.max()])
    x = np.linspace(min_qn, max_qn)

    plt.plot(qn_x, qn_y, ls="", marker="o", markersize=6)
    plt.plot(x, x, color="k", ls="--")
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.grid()
    plt.title(name, fontsize=14)
    plt.show()


def make_report(dataframe_left: pd.DataFrame,
                dataframe_right: pd.DataFrame,
                columns_for_compare: dict,
                check_peaks: bool = False):
    """
    The function generate report dataframe with all necessary information

    :param dataframe_left: (meteo stations) dataframe with columns for comparison
    :param dataframe_right: (reanalysis) dataframe with columns for comparison
    :param columns_for_compare: dictionary where keys are column names in
    dataframe_left and values are column names in dataframe_right, which should be
    compared
    :param check_peaks: is there a need to calculate metrics for peak values
    (> 75 percentile)

    :return : report dataframe
    """

    # Get all keys (column names) for compare
    column_left_names = list(columns_for_compare.keys())

    variables = []
    biases = []
    maes = []
    korrs = []
    for column_left_name in column_left_names:
        column_right_name = columns_for_compare.get(column_left_name)

        column_left = np.array(dataframe_left[column_left_name])
        column_right = np.array(dataframe_right[column_right_name])

        bias = bias_calculate(column_left, column_right)
        mae = mean_absolute_error(column_left, column_right)
        korr = stats.pearsonr(column_left, column_right)

        # Forming var column
        variable_name = ''.join((column_left_name, ' vs ', column_right_name))
        variables.append(variable_name)
        biases.append(bias)
        maes.append(mae)
        korrs.append(korr[0])

        if check_peaks is True:
            # Get 75 percentile
            q75 = np.quantile(column_left, 0.75)
            ids = np.argwhere(column_left > q75)

            peak_left = np.ravel(column_left[ids])
            peak_right = np.ravel(column_right[ids])

            bias_peak = bias_calculate(peak_left, peak_right)
            mae_peak = mean_absolute_error(peak_left, peak_right)
            korr_peak = stats.pearsonr(peak_left, peak_right)

            variable_name = ''.join(('(Peak) ', variable_name))
            variables.append(variable_name)
            biases.append(bias_peak)
            maes.append(mae_peak)
            korrs.append(korr_peak[0])

    df = pd.DataFrame({'Variables': variables,
                       'Bias': biases,
                       'MAE': maes,
                       'Correlation': korrs})
    return df


def _convert_to_polar(arr):
    """
    The function of normalizing the array scale for the range from 0 to 2*Pi

    :param : array to process
    """

    # Normalize to the range from 0 to 1
    arr = (arr / 360)
    # And now transform to the polar coordinates
    arr = arr * (2 * np.pi)
    return arr


def plot_wind_rose(dataframe, vel_col: str, direction_col: str):
    """
    Function for drawing the wind rose plot where color shows the wind velocity

    :param dataframe: dataframe with data for plot
    :param vel_col: name of column with velocity of the wind
    :param direction_col: name of column with direction of the wind
    """

    # Carry out the sampling on the points (from (-1 to 0] - denotes calm)
    ticks = np.array([-1, 0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 361])
    dataframe = dataframe.groupby(pd.cut(dataframe[direction_col], ticks)).agg({direction_col: 'count',
                                                                                vel_col: 'mean'})
    dataframe = dataframe.rename(columns={direction_col: " "})
    dataframe = dataframe.reset_index()
    # Number of observations in total
    all_values = dataframe[' '].sum()

    # The values from the first row refer to calm
    calm_amount = dataframe[' '][0]
    # Now we remove them from the calculations
    dataframe = dataframe.drop(dataframe.index[0])

    # Let's combine the parts in the northern direction
    dataframe['Direction'] = [0, 45, 90, 135, 180, 225, 270, 315, 0]
    dataframe = dataframe.groupby(dataframe['Direction']).agg({' ': 'sum',
                                                               vel_col: 'mean'})
    dataframe = dataframe.reset_index()
    dataframe['The percentage of cases with these directions of the wind'] = _convert_to_polar(dataframe['Direction'])

    # Let's calculate the percentage of directions by points
    dataframe[' '] = (dataframe[' '] / all_values) * 100
    calm_amount_scaled = (calm_amount / all_values) * 100
    dataframe = dataframe.round({vel_col: 2})

    # Inverting the wind direction, because the graph is displayed in an inverted form
    dataframe['The percentage of cases with these directions of the wind'] = _convert_to_polar(np.array([0, 315, 270, 225, 180, 135, 90, 45]))
    with sns.axes_style("whitegrid"):
        days = np.array(dataframe['The percentage of cases with these directions of the wind'])
        d = np.array(dataframe[" "])

        angle_ticks = _convert_to_polar(np.array([0, 315, 270, 225, 180, 135, 90, 45]))

        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
        ax.plot(days, d, c='black', alpha=0.2)
        ax.plot([days[-1], days[0]], [d[-1], d[0]], c='black', alpha=0.2)
        am = ax.scatter(
            dataframe['The percentage of cases with these directions of the wind'], d,
            c=dataframe[vel_col], cmap='coolwarm', s=120)
        plt.xticks(angle_ticks,
                   ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        plt.xlabel('The percentage of cases with these directions of the wind')
        ax.set_theta_zero_location('N')
        fig.colorbar(am)

    print(f'Percentage of observations with calm {calm_amount_scaled:.1f}%')
