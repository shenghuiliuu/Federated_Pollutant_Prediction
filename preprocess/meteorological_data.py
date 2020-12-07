import os
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy.spatial import distance
import numpy as np
import utils
from preprocess import replace_missing_value

entry_point = "https://opendata-download-metobs.smhi.se"
from utils import *


def get_data(par, station):
    api = "api/version/1.0/parameter/%d/station/%d/period/corrected-archive/data.csv" % (par, station)
    url = os.path.join(entry_point, api)
    print(url)
    r = requests.get(url)
    return r.text


def get_line_number(phrase, text):
    for i, line in enumerate(text.splitlines(), 1):
        if phrase in line:
            return i


def get_weather_data(name, key, long, lag):
    api = "api/version/1.0/parameter/%d.xml" % (key)
    r = requests.get(os.path.join(entry_point, api))

    soup = BeautifulSoup(r.text, 'xml')
    min_dist = 1000000
    best_df = pd.DataFrame(
        data = {name : np.nan, "Start" : pd.date_range('2014-01-01', '2020-01-01', freq='1H', closed='left')}
     )
    best_df.index = best_df['Start']
    stations = list(soup.find_all("station"))
    stations.sort(key=lambda x : distance.euclidean([long, lag], [float(x.longitude.text), float(x.latitude.text)]))
    for station in (soup.find_all("station")):
        dis = distance.euclidean([long, lag], [float(station.longitude.text), float(station.latitude.text)])
        if int(station.to.text[:4]) > 2019 and dis < 0.7:
            id = int(station.id.text)
            raw = get_data(key, id)
            if "2017-01-01;00:00:00" not in raw or "2017-01-01;01:00:00" not in raw or dis > min_dist:
                continue
            line_number = get_line_number("Datum", raw)
            df = pd.read_csv(StringIO(raw), skiprows=line_number - 1, sep=';')
            df['Datum'] = pd.to_datetime(df['Datum'] + ' ' + df['Tid (UTC)'])
            df.drop(df.columns[[3, 4, 5]], axis=1, inplace=True)

            df.drop(columns=['Tid (UTC)'], inplace=True)

            df = df.rename({'Datum': 'Start'}, axis='columns')
            df.set_index("Start", inplace=True)
            df.index = df.index + pd.Timedelta('0 days 1 hours')
            begin_index = pd.to_datetime("2014-1-1 00:00:00")
            end_index = pd.to_datetime("2019-12-31 23:00:00")
            df = df[begin_index: end_index]

            station_id = int(station.id.text)
            print(station_id)
            min_dist = dis
            if best_df is None:
                best_df = df
            else:
                best_df.fillna(df, inplace=True)
            
            
    print(best_df.describe())
    print(min_dist)

    return best_df


def main():
    positions = pd.read_csv('../data/station_long_lat.csv')
    for id in utils.STATIONS:
        long = positions[positions['Station ID'] == id]["Longitude"].values[0]
        lag = positions[positions['Station ID'] == id]["Latitude"].values[0]
        path = '../data/preprocessed_data/' + str(id) + '.csv'
        station_data = pd.read_csv(path, index_col=0)
        station_data.index = pd.to_datetime(station_data.index)
        station_data = station_data[OUTPUTS]
        for name, key in utils.METEO_FEATURES.items():
            weather_data = get_weather_data(name, key, long, lag)
            print(weather_data.head())
            
            station_data = pd.concat([station_data, weather_data], axis=1)
        station_data = station_data[INPUTS]
        station_data = replace_missing_value(station_data)
        station_data.to_csv(path)
        print(station_data.head())

if __name__ == "__main__":
    main()
