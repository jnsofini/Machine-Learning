# Time Series Analysis
This repository contains all projects that are based on time series. The projects uses python to get data, prepare, visualize, and forecast. See the corresponding folder for the data, notebooks of a project.

## 1. Canadian Collision Data
The aim is to identify the common cause of accidents and identify at risk drivers and also identify possible locations with higher accident numbers. These location can then be mapped by the city or responsible authority to reduce the number of fatal accidents.
Insurance companies can also prepare appropriately

The data used can be found in [Canada Open Data](https://open.canada.ca/data/en/dataset/1eb9eba7-71d1-4b30-9fb1-30cbdab7e63a) containing list of accidents from 1999 to 2015. The followings are covered in this notebook

   - Steps to getting the data
   - Display some features of raw data
   - Prepare or clean the information
   - Visualize fatal and non-fatal accidents
   - Check seasonality effect

A plot of the variation of accidents between 1999 and 2015 is shown below. It shows a visble yearly seasonality with an decreasing trend, showing that accidents have reduced over the years. Below is a plot for both fatal and non-fatal accidents
![](https://github.com/jnsofini/Time-Series-Analysis/blob/master/figs/accident_trend.png).

In addition, a plot of the monthly accident is made. It can be noticed that the peak period for accidents is in the sumer months of July and August. On the other hand the lowest are record in March April. See below for the variation
![](https://github.com/jnsofini/Time-Series-Analysis/blob/master/figs/monthly_accident_trend.png).

## 2. Canadian Temperature.
This time series forcasting deals with the analysing temperature in Canadian series. There are multiple notebooks for the project. In this set of not books on exploration notebooks, however, we will be working exclusively with temperature data. The data is from the NOAA weather station, and we will pick one station in Prince Albert, a city in SK, Canada. The data, as well as its properties, can be found in the NOAA website. There are three other stations in Canada, but we will focus on this one which is part of the GCOS Surface Network (GSN). In the next project, we will work with multiple cities.

### Part 1 Temperature_time_series_exploration (Getting the data).ipynb
This notebook focus only on getting the time series. It can be considered as a basic intro but several aspects are covered. This notebook only focus on one city as well. The principle derived here applies to most of the other cities
Temperature Time Series Visualization

#### Part 2 Time_series_exploration (Exploration analysis).ipynb
In this note book the followings are covered
  
 -   Learn the steps to create a Time Series dataset from the NOAA data
 -   Display some features of raw data
 -   Prepare or clean the information
 -   Visualize temperature variations over a period from 1973 to present
 -   Check seasonality and stationarity
 -   Additional focus on Dickey-Fuller test

Data visualization shows the followingvariation for monthly temperatures for Prince Albert. The minimum temperature:  -50.0 deg celcius and maximum temperature:  38.8 deg celcius have ever been recorded. 
![](https://github.com/jnsofini/Time-Series-Analysis/blob/master/figs/daily_temperature.png).

Later on the a test for the stationarity is a plot the autocorrelation function (ACF). 
