import numpy as np
import pandas as pd
import datetime

from matplotlib import pyplot as plt

from meteotik.reanalysis import uv_to_wind, uv_to_direction


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

    def check_correctness(self, vel_col, direction_col,
                          date_col: str = 'Date') -> None:
        """
        The function allows calculating the wind directions from the U and V components

        :param vel_col: name of column with velocity of the wind
        :param direction_col: name of column with direction of the wind
        :param date_col: name of column with datetime
        """

        # Source arrays with wind velocity and direction
        vel_arr = np.array(self.dataframe[vel_col])
        direction_arr = np.array(self.dataframe[direction_col])

        # Calculate U and V components from velocity and direction
        u_arr, v_arr = self.calculate_uv(vel_col, direction_col)

        # Calculate velocity from U and V
        vel_restored = uv_to_wind(u_arr, v_arr)
        # # Calculate direction from U and V
        direction_restored = uv_to_direction(u_arr, v_arr)

        restored_name = ''.join(('Restored ', vel_col))
        plt.plot(self.dataframe[date_col], vel_arr, label=vel_col)
        plt.plot(self.dataframe[date_col], vel_restored, label=restored_name)
        plt.xlabel(date_col, fontsize=14)
        plt.legend()
        plt.grid()
        plt.show()

        restored_name = ''.join(('Restored ', direction_col))
        plt.plot(self.dataframe[date_col], direction_arr, label=direction_col)
        plt.plot(self.dataframe[date_col], direction_restored, label=restored_name)
        plt.legend()
        plt.xlabel(date_col, fontsize=14)
        plt.grid()
        plt.show()


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

