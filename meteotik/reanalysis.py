import os
import datetime
import calendar
from abc import abstractmethod

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm
import plotly.express as px

from netCDF4 import Dataset



class ReanalysisProcessor:
    """
    Basic class for generating time series from reanalysis grids at points at
    selected coordinates

    :param folder: the path to the folder where the monthly reanalysis files
    are located
    :param files: list of file names in folder, which are need to process. If
    None - all files in folder will be processed
    """

    def __init__(self, folder, files=None):
        self.folder = folder

        # Do we need to process all files in dir or several of them
        if files is None:
            files = os.listdir(self.folder)
        else:
            pass

        files.sort()
        self.files = files

    def print_variables(self) -> None:
        """ Display variables of one netcdf file in the folder """

        file = self.files[0]
        file_path = os.path.join(self.folder, file)
        netcdf_file = Dataset(file_path)

        print(f'Variables in netcdf file {list(netcdf_file.variables)}')
        netcdf_file.close()

    @abstractmethod
    def field_visualisation(self, variable, file_to_vis=None) -> None:
        """
        Method for visualizing the reanalysis field

        :param variable: name of the variable to plot
        :param file_to_vis: the name of the file that you want to visualise
        """
        raise NotImplementedError()

    @abstractmethod
    def field_animation(self, variable, file_to_vis=None) -> None:
        """
        Method for visualizing the reanalysis fields through time

        :param variable: name of the variable to plot
        :param file_to_vis: the name of the file that you want to visualise
        """
        raise NotImplementedError()

    def _siutable_ids(self, latitude, longitude, coordinates):

        # Selecting coordinates
        lat_coord = coordinates.get('lat')
        lon_coord = coordinates.get('lon')

        # Calculate the distance to the point of interest
        lat_dist = latitude - lat_coord
        lon_dist = longitude - lon_coord
        lat_dist = np.abs(lat_dist)
        lon_dist = np.abs(lon_dist)

        # Minimal differences
        lat_index = np.argmin(lat_dist)
        lon_index = np.argmin(lon_dist)

        return lat_index, lon_index

    def _print_borders(self, longitude_name: str, latitude_name: str) -> None:
        """
        The method allows getting the spatial coverage of the reanalysis grid
        and display such metrics on the

        :param longitude_name: the name of the variable that denotes the longitude
        :param latitude_name: the name of the variable that denotes the latitude
        """

        file_path = os.path.join(self.folder, self.files[0])
        netcdf_file = Dataset(file_path)

        lon_vector = np.array(netcdf_file.variables[longitude_name])
        lat_vector = np.array(netcdf_file.variables[latitude_name])

        print(f'"Lower left" corner: latitude - {np.min(lat_vector)}, '
              f'longitude - {np.min(lon_vector)}')
        print(f'"Upper right" corner: latitude - {np.max(lat_vector)}, '
              f'longitude - {np.max(lon_vector)}')

        scope_lat = np.max(lat_vector) - np.min(lat_vector)
        scope_lon = np.max(lon_vector) - np.min(lon_vector)

        print(f'The scope by latitude - {scope_lat}')
        print(f'The scope by longitude - {scope_lon} \n')
        netcdf_file.close()

    def _add_months(self, source_date, months: int = 1) -> datetime.date:
        """
        Function for adding the 1st or several months to the source date

        :param source_date: the current date
        :param months: the number of months for which there is a need to
        increment the source date

        :return : a date that is "months" month longer than the original date
        """

        month = source_date.month - 1 + months
        year = source_date.year + month // 12
        month = month % 12 + 1
        day = min(source_date.day, calendar.monthrange(year, month)[1])
        return datetime.date(year, month, day)


class ERA5Processor(ReanalysisProcessor):
    """ Class for processing ERA5 monthly netcdf files """

    def __init__(self, folder, files=None):
        super().__init__(folder, files)

        # Variables to process data in reanalysis
        self.time_var_name = 'time'
        self.u10_name = 'u10'
        self.v10_name = 'v10'
        self.longitude_name = 'longitude'
        self.latitude_name = 'latitude'

    def show_spatial_coverage(self):
        """
        Wrapped method for displaying spatial coverage of reanalysis fields
        """
        self._print_borders(longitude_name=self.longitude_name,
                            latitude_name=self.latitude_name)

    def field_visualisation(self, variable, file_to_vis=None):
        """ Method for visualizing the reanalysis field """

        if file_to_vis is None:
            file_to_vis = self.files[0]
        else:
            pass

        file_path = os.path.join(self.folder, file_to_vis)
        netcdf_file = Dataset(file_path)

        lon_vector = np.array(netcdf_file.variables[self.longitude_name])
        lat_vector = np.array(netcdf_file.variables[self.latitude_name])

        variable_arr = np.array(netcdf_file.variables[variable])
        variable_arr = variable_arr[0, :, :]

        cmap = cm.get_cmap('coolwarm')
        plt.imshow(variable_arr, cmap=cmap)
        plt.colorbar()
        plt.title(variable, fontsize=14)
        plt.xticks([0, len(lon_vector)-1],
                   [lon_vector[0], lon_vector[-1]],
                   rotation='vertical')
        plt.yticks([0, len(lat_vector)-1],
                   [lat_vector[0], lat_vector[-1]],
                   rotation='horizontal')
        plt.show()

        netcdf_file.close()

    def field_animation(self, variable, file_to_vis=None) -> None:
        """ Method for visualizing the reanalysis fields through time """

        if file_to_vis is None:
            file_to_vis = self.files[0]
        else:
            pass

        file_path = os.path.join(self.folder, file_to_vis)
        netcdf_file = Dataset(file_path)

        variable_arr = np.array(netcdf_file.variables[variable])

        fig = px.imshow(variable_arr, color_continuous_scale='RdBu_r',
                        animation_frame=0,
                        labels=dict(animation_frame="slice"))
        fig.show()

    def prepare_time_series(self,
                            coordinates,
                            change_time_step=False,
                            new_time_step='D'):
        """
        This method allows getting a time series from a point with the
        specified coordinates

        :param coordinates: coordinates of the point from which you want to form a
        time series
        :param change_time_step: is there a need to change the time step
        :param new_time_step: new time step (ignore if change_time_step=False)

        :return: pandas dataframe with "U_reanalysis", "V_reanalysis",
        "Date" columns
        """

        all_us = []
        all_vs = []
        dates = []
        for index, file in enumerate(self.files):
            print(f'Processing... {file}')
            file_path = os.path.join(self.folder, file)
            netcdf_file = Dataset(file_path)

            longitude = netcdf_file.variables[self.longitude_name]
            latitude = netcdf_file.variables[self.latitude_name]

            # Get arrays
            latitude = np.array(latitude)
            longitude = np.array(longitude)

            # Time - the number of hours that have passed since the beginning
            time = netcdf_file.variables[self.time_var_name]
            u10 = netcdf_file.variables[self.u10_name]
            v10 = netcdf_file.variables[self.v10_name]

            # Start date for ERA5 dataframe is 1900-01-01
            beginning_date = datetime.date(1900, 1, 1)
            time = np.array(time)
            start_hour = int(time[0])

            # Convert hours from int to datetime format
            delta_for_start = datetime.timedelta(hours=start_hour)
            delta_for_end = datetime.timedelta(hours=len(time))

            # Get starting date and finish
            start_date = beginning_date + delta_for_start
            end_date = start_date + delta_for_end

            # Generate time ids
            date_range = pd.date_range(start=start_date, end=end_date, freq='H')
            date_range = date_range[:-1]

            # Determining the indexes of the pixel
            if index == 0:
                # It will be enough to determine the pixel indexes only once
                lat_index, lon_index = self._siutable_ids(latitude,
                                                          longitude,
                                                          coordinates)

            u_arr = np.array(u10)
            v_arr = np.array(v10)

            # Preparing time series
            us = u_arr[:, lat_index, lon_index]
            vs = v_arr[:, lat_index, lon_index]

            if change_time_step == True:
                # Changing the time step to a new discreteness
                u_series = pd.Series(us, index=date_range)
                v_series = pd.Series(vs, index=date_range)

                u_series = u_series.resample(new_time_step).mean()
                v_series = v_series.resample(new_time_step).mean()

                us = u_series
                vs = v_series
                date_range = u_series.index
            else:
                pass

            all_us.extend(list(us))
            all_vs.extend(list(vs))
            dates.extend(list(date_range))

            netcdf_file.close()

        dataframe = pd.DataFrame({'U_reanalysis': all_us,
                                  'V_reanalysis': all_vs,
                                  'Date': dates})
        return dataframe


class CFS2Processor(ReanalysisProcessor):
    """ Class for processing CFS2 monthly netcdf files """

    def __init__(self, folder, files=None):
        super().__init__(folder, files)

        # Variables to process data in reanalysis
        self.time_var_name = 'time'
        self.u10_name = '10u'
        self.v10_name = '10v'
        self.longitude_name = 'long'
        self.latitude_name = 'lat'

    def show_spatial_coverage(self):
        """
        Wrapped method for displaying spatial coverage of reanalysis fields
        """

        self._print_borders(longitude_name=self.longitude_name,
                            latitude_name=self.latitude_name)
