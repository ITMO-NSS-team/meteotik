import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import datetime


class MeteoStations:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def calculate_uv(self, vel_col: str, direction_col: str):
        """
        The method allows calculating the U and V components of the wind
        based on the velocity and direction

        :param vel_col: name of column with velocity of the wind
        :param direction_col: name of column with direction of the wind

        :return u_arr: array with U component
        :return v_arr: array with V component
        """
        vel_arr = np.array(self.dataframe[vel_col])
        direction_arr = np.array(self.dataframe[direction_col])

        # Направление должно быть в радианах
        direction_arr = np.radians(direction_arr)

        # Расчет sin и cos для направлений ветра
        direction_sin = np.sin(direction_arr)
        direction_cos = np.cos(direction_arr)

        u_arr = (-1 * vel_arr) * direction_sin
        v_arr = (-1 * vel_arr) * direction_cos
        return u_arr, v_arr

    def plot_wind_rose(self, dataframe, vel_col: str, direction_col: str):
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
        dataframe['The percentage of cases with these directions of the wind'] = \
            self._convert_to_polar(dataframe['Direction'])

        # Let's calculate the percentage of directions by points
        dataframe[' '] = (dataframe[' '] / all_values) * 100
        calm_amount_scaled = (calm_amount / all_values) * 100
        dataframe = dataframe.round({vel_col: 2})

        # Inverting the wind direction, because the graph is displayed in an inverted form
        dataframe['The percentage of cases with these directions of the wind'] = self._convert_to_polar(np.array([0, 315, 270, 225, 180, 135, 90, 45]))
        with sns.axes_style("whitegrid"):
            days = np.array(dataframe['The percentage of cases with these directions of the wind'])
            d = np.array(dataframe[" "])

            angle_ticks = self._convert_to_polar(np.array([0, 315, 270, 225, 180, 135, 90, 45]))

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

    def time_shift(self, dataframe, date_col, shift, add=True):
        """
        The method allows adding or subtracting a few hours to get a time shift

        :param dataframe: dataframe tp process
        :param date_col: name of column with datetime
        :param shift: hours for shifting
        :param add: is there a need to add hours or to subtract them
        :return: pandas series with shifted times
        """
        copy_df = dataframe.copy()

        delta = datetime.timedelta(hours=int(shift))
        new_times = []
        for i_time in dataframe[date_col]:
            if add:
                new_time = i_time + delta
            else:
                new_time = i_time - delta
            new_times.append(new_time)

        copy_df[date_col] = new_times
        copy_df[date_col] = pd.to_datetime(copy_df[date_col])

        return copy_df[date_col]

    @staticmethod
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


class Rp5Station(MeteoStations):

    def __init__(self, dataframe):
        super().__init__(dataframe)

    def convert_wind_direction(self, column):
        """
        The method translates the verbal description of the direction of the
        winds into degrees

        :param : name of column with wind directions
        :return : list with degrees
        """
        direction_dict = {'Штиль': 0, 'северо-северо-востока': 22.5,
                          'северо-востока': 45, 'востоко-северо-востока': 67.5,
                          'востока': 90, 'востоко-юго-востока': 112.5,
                          'юго-востока': 135, 'юго-юго-востока': 157.5,
                          'юга': 180, 'юго-юго-запада': 202.5, 'юго-запада': 225,
                          'западо-юго-запада': 247.5, 'запада': 270,
                          'северо-запада': 315, 'северо-северо-запада': 337.5,
                          'севера': 360}

        new_values = []
        directions_arr = self.dataframe[column]
        for element in directions_arr:
            if element[:5] == 'Штиль':
                code = 0
            else:
                splitted = element.split(' ')
                wind_direction = splitted[-1]

                code = direction_dict.get(wind_direction)

            new_values.append(code)
        return new_values


class MeteoMStation(MeteoStations):

    def __init__(self, dataframe):
        super().__init__(dataframe)

    def prepare_datetime_column(self, year_col, month_col,
                                day_col=None, hour_col=None):
        """
        Function for forming a time column from separate columns of date/time

        :param year_col: name of the year data column
        :param month_col: name of the month data column
        :param day_col: name of the days data column
        :param hour_col: name of the hour data column

        :return: pandas series with datetime strings
        """
        datetime_col = []
        for row_id in range(0, len(self.dataframe)):
            row = self.dataframe.iloc[row_id]
            if day_col is None:
                string_datetime = ''.join((str(row[year_col]), '.', str(row[month_col])))
            elif hour_col is None:
                string_datetime = ''.join((str(row[year_col]), '.',
                                           str(row[month_col]), '.',
                                           str(row[day_col])))
            else:
                string_datetime = ''.join((str(row[year_col]), '.',
                                           str(row[month_col]), '.',
                                           str(row[day_col]), '.',
                                           str(row[hour_col])))
            datetime_col.append(string_datetime)

        df = pd.DataFrame({'Date': datetime_col})
        df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d.%H')
        return df['Date']

