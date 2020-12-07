import glob
import os
import re
from shutil import rmtree

import pandas as pd

import utils


def remove_comments(csv_path):
    pd_frame = pd.read_csv(csv_path, sep=';', comment='#', skiprows=1)
    columns_names = list(pd_frame.columns)

    regex = re.compile("\([^()]*\)")
    new_name = [e.replace(re.search(regex, e).group(), "").strip() if re.search(regex, e) else e.strip() for e in
                columns_names]
    pd_frame.columns = new_name
    pd_frame.to_csv(csv_path.replace(utils.RAW_DIR, utils.ALL_DIR))


def load_by_year(year=2019, dir=utils.ALL_DIR):
    csv_files = glob.glob(os.path.join(dir, str(year), "*.csv"))
    stations = {}
    for path in csv_files:
        id = re.findall(r'\d+', path)[1]
        data = pd.read_csv(path)
        stations[id] = data

    for station in stations.values():
        station["Start"] = pd.to_datetime(station["Start"])
        if utils.PRE_DIR != dir:
            station["Slut"] = pd.to_datetime(station["Slut"])
            assert ((station["Slut"] - station["Start"]).eq(pd.to_timedelta("01:00:00"))).all()
            station.drop(columns=["Slut"], inplace=True)
        station.set_index("Start", inplace=True)
    return stations


# Replacing missing values.
def replace_missing_value(station):
    
    for feature in station.columns:
        time = station.index[0]
        end_time = station.index[-1]
        while time < end_time:
            if time in station.index and pd.isna(station.loc[time, feature]):
                aft_time = time
                while aft_time in station.index and pd.isna(station.loc[aft_time, feature]):
                    aft_time += pd.to_timedelta("01:00:00")
                if aft_time - time > pd.to_timedelta("12:00:00"):
                    station.drop(station[(station.index >= time) & (station.index < aft_time)].index, inplace=True)
                    time = aft_time
                elif aft_time in station.index:
                    station.loc[time, feature] = station.loc[aft_time, feature]
                    time += pd.to_timedelta("01:00:00")
                else:
                    station.drop(index=time, inplace=True)
                    time += pd.to_timedelta("01:00:00")
                
            else:
                time += pd.to_timedelta("01:00:00")
        if pd.isna(station.loc[end_time, feature]):
            station.loc[end_time, feature] = station.loc[end_time - pd.to_timedelta("01:00:00"), feature]
    return station


def normalize_data(station):
    nor_station = (station - station.mean()) / station.std()
    return nor_station


def preprocess_data(csv_path):
    station = pd.read_csv(csv_path, index_col=0)
    if not all(elem in station.columns for elem in utils.OUTPUTS):
        return
    print("preprocessing data station %s ..." % (re.findall(r'\d+', csv_path)[1]))
    for name in station.columns:
        if name not in utils.OUTPUTS and name not in ["Slut", "Start"]:
            station.drop(columns=[name], inplace=True)
    station["Slut"] = pd.to_datetime(station["Slut"])
    station["Start"] = pd.to_datetime(station["Start"])
    assert ((station["Slut"] - station["Start"]).eq(pd.to_timedelta("01:00:00"))).all()
    station.set_index("Start", inplace=True)
    station.drop(columns=["Slut"], inplace=True)

    # Drop rows where "PM2.5" are missing from the beginning
    idx = max(station[col].first_valid_index() for col in [station.columns])
    station = station.truncate(before=idx)

    station = replace_missing_value(station)
    # station = normalize_data(station)
    station.to_csv(csv_path.replace(utils.ALL_DIR, utils.PRE_DIR))


def concat_by_years(years):
    for station in utils.STATIONS:
        data_list = []
        for year in years:
            pattern = os.path.join(utils.PRE_DIR, str(year), "*" + str(station) + "*.csv")
            csv_files = glob.glob(pattern)
            data = pd.read_csv(csv_files[0])
            data["Start"] = pd.to_datetime(data["Start"])
            data.set_index("Start", inplace=True)
            data_list.append(data)
        station_df = pd.concat(data_list, verify_integrity=True)
        station_df.to_csv(os.path.join(utils.PRE_DIR, str(station) + ".csv"))


# concat_by_years()

if __name__ == "__main__":
    years = range(2014, 2020)
    # download(years=years)
    for year in years:
        for dir in [utils.ALL_DIR, utils.PRE_DIR]:
            des_dir = os.path.join(dir, str(year))
            if os.path.exists(des_dir):
                rmtree(des_dir)
            os.makedirs(des_dir)
        csv_files = glob.glob(os.path.join(utils.RAW_DIR, str(year), "*.csv"))
        for file in csv_files:
            id = re.findall(r'\d+', file)[1]
            if int(id) not in utils.STATIONS:
                continue
            remove_comments(file)
            preprocess_data(file.replace(utils.RAW_DIR, utils.ALL_DIR))

    concat_by_years(years)
    
