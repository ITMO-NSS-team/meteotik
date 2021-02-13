# ![meteotik_logo.png](https://raw.githubusercontent.com/ITMO-NSS-team/meteotik/main/media/meteotik_logo.png)

# meteotik
Module for processing reanalysis grids and comparative analysis of 
time series with meteorological parameters

## Requirements
    'python>=3.7',
    'pandas>=1.1.0',
    'netcdf4==1.5.4',
    'numpy',
    'seaborn',
    'scipy',
    'matplotlib',
    'sklearn',
    'plotly'

## Install module
    pip install git+https://github.com/ITMO-NSS-team/meteotik

## Reason to use
This module allows comparing time series with meteoparameters obtained from ERA5 or 
CFS2 reanalysis with data from weather stations. With this module you can generate 
reports, make visualizations, calculate the wind components U and V from the velocity
 and direction (and perform reverse operations also) and much more.

## Documentation
All necessary documentation you can find in docstring or in examples.

#### Supported reanalysis sources

☑️ ERA5 - ERA5Processor class
☑️ CFS2 - CFS2Processor class
☑️ BCustom interpolated grids - InterpolatedProcessor class  

#### Supported weather stations sources

☑️ [rp5](https://rp5.ru/%D0%9F%D0%BE%D0%B3%D0%BE%D0%B4%D0%B0_%D0%B2_%D0%BC%D0%B8%D1%80%D0%B5)
☑️ [Roshydromet](http://meteo.ru/data/163-basic-parameters)

## Examples 
* [ERA5 comparison with weather station (Roshydromet) example](https://github.com/ITMO-NSS-team/meteotik/blob/main/examples/ERA5_example.ipynb) 
(in russian)
* [CFS2 comparison with weather station (rp5) example](https://github.com/ITMO-NSS-team/meteotik/blob/main/examples/CFS2_example.ipynb) 
(in russian)