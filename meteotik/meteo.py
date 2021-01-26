import os
import numpy as np
import pandas as pd


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

