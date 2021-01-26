import os
import numpy as np
import pandas as pd

class MeteoStations:

    def __init__(self):
        pass


class Rp5Station(MeteoStations):

    def __init__(self):
        super().__init__()

    def convert_wind_direction(self, directions_arr):
        """
        The method translates the verbal description of the direction of the
        winds into degrees

        :param : array-like structure with wind directions
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

    def __init__(self):
        super().__init__()
