import os
import numpy as np
import pandas as pd


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

