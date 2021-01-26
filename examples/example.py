from meteotik.reanalysis import ERA5Processor

""" Example how to use er5 preprocessor for preparing U and W wind components data"""

rean_processor = ERA5Processor(folder='./data/era5')

# Display spatial coverage of the data
rean_processor.show_spatial_coverage()

# Print variables
rean_processor.print_variables()

# How the fields in the reanalysis grids looks like
rean_processor.field_visualisation(variable='u10',
                                   file_to_vis='wind_bechevinski_2019-10-01.nc')

# Animation of fields
rean_processor.field_animation(variable='u10')

# Prepare time series from point
dataframe = rean_processor.prepare_time_series(coordinates={'lat':55.2,'lon':165.98},
                                               change_time_step=False)
print(dataframe)