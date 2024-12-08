#!/usr/bin/env python
# coding: utf-8
import gc
import os
import traceback
from multiprocessing import Pool, cpu_count
from os import cpu_count
import random
import pandas as pd
import datetime
from datetime import datetime, timedelta
import re
import glob
import logging as log
from tqdm import tqdm
from metar_taf_parser.parser.parser import MetarParser, TAFParser
from metar_taf_parser.model.enum import CloudQuantity, CloudType
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pdf
from functools import partial
from datetime import timedelta
from typing import Optional, List

log.basicConfig(
    level=log.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    filename='data_loader.log',
    filemode='w',
)

def load_actual_arrivals(airport_code, data_dir='data'):
    # runways_files = glob.glob(os.path.join(data_dir, 'FUSER_test', airport_code, '**', f'{airport_code}_*runways_data_set.csv'), recursive=True)
    runways_files = glob.glob(os.path.join(data_dir, airport_code, '**', f'{airport_code}_*runways_data_set.csv'), recursive=True)

    arrival_times = []

    for file in tqdm(runways_files, desc='Loading true arrival times'):
        df = pd.read_csv(file, parse_dates=['arrival_runway_actual_time'])
        arrival_times.append(df)

    arrivals_df = pd.concat(arrival_times)
    arrivals_df['arrival_runway_actual_time'] = pd.to_datetime(arrivals_df['arrival_runway_actual_time'])
    arrivals_df.set_index('arrival_runway_actual_time', inplace=True)

    return arrivals_df

def load_estimated_arrivals(airport_code, data_dir='data'):
    tfm_files = glob.glob(os.path.join(data_dir, airport_code, '**', f'{airport_code}_*TFM_track_data_set.csv'), recursive=True)
    est_arrivals = []

    for file in tqdm(tfm_files, desc='Loading estimated arrivals'):
        df = pd.read_csv(file, parse_dates=['timestamp', 'arrival_runway_estimated_time'])
        df = df[['timestamp', 'arrival_runway_estimated_time']].dropna()
        est_arrivals.append(df)

    est_arrivals_df = pd.concat(est_arrivals)
    est_arrivals_df['timestamp'] = pd.to_datetime(est_arrivals_df['timestamp'])
    est_arrivals_df['arrival_runway_estimated_time'] = pd.to_datetime(est_arrivals_df['arrival_runway_estimated_time'])

    return est_arrivals_df

def filter_file_time(files, start, end, interval=timedelta(hours=1)):
    """
    Filters TAF files based on overlapping time intervals.

    Parameters:
        files (list): List of file paths.
        start (datetime): Start datetime of the desired range.
        end (datetime): End datetime of the desired range.
        interval (timedelta): Duration each file covers (default is 1 hour).

    Returns:
        list: Filtered list of file paths whose intervals overlap with the desired range.
    """
    filtered_files = []

    for filename in files:
        basename = os.path.basename(filename)
        # Adjust regex based on your file naming convention
        match = re.match(r'.*\.(\d{8})\.(\d{2})Z\.txt', basename)
        if match:
            date_part = match.group(1)  # e.g., '20220901'
            time_part = match.group(2)  # e.g., '00'
            try:
                dt = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H")
                file_start = dt
                file_end = dt + interval

                # Check if the file's interval overlaps with the desired range
                if (file_start <= end) and (file_end >= start):
                    filtered_files.append(filename)
            except ValueError as ve:
                log.error(f"Skipping file {filename}: {ve}")
        else:
            log.error(f"Filename {basename} does not match the expected pattern.")
    return filtered_files

def get_df(data, start, end, start_time_col, end_time_col):
    df = pd.DataFrame(data)
    # df.dropna(subset=['timestamp'], inplace=True)
    df[start_time_col] = pd.to_datetime(df[start_time_col])
    df[end_time_col] = pd.to_datetime(df[end_time_col])
    df = df[(df[start_time_col] >= start) & (df[end_time_col] <= end)]
    return df


bins = [0, 2000, 6000, 12000, 20000]
bin_labels = ['0_2000', '2001_6000', '6001_12000', '12001_20000']

# Define condition priority
condition_priority = {'FEW': 1, 'SCT': 2, 'BKN': 3, 'OVC': 4}

def bin_clouds(clouds):
    # Initialize a dictionary for the final results
    result = {}
    for label in bin_labels:
        result[f'clouds_{label}_condition'] = 'NONE'

    # Dictionary to track the highest priority condition found in each bin
    best_conditions = {label: ('NONE', 0) for label in bin_labels}
    # Format: { bin_label: (condition, priority) }

    for cloud in clouds:
        height = cloud.height
        quantity = cloud.quantity
        if quantity == CloudQuantity.SKC or quantity == CloudQuantity.NSC:
            return result
        if cloud.type == CloudType.CB:
            return {alt: 'Cumulonimbus' for alt in result.keys()}

        # Determine the bin for this cloud
        if height is None:
            continue
        bin_label = pd.cut([height], bins=bins, labels=bin_labels, include_lowest=True)[0]
        if bin_label is pd.NA:
            # If height doesn't fall into a defined bin, skip this cloud
            continue

        # Check if this cloud's condition is recognized and has a priority
        if quantity in condition_priority:
            current_best_condition, current_best_priority = best_conditions[bin_label]
            candidate_priority = condition_priority[quantity]

            # Update if we found a higher priority condition
            if candidate_priority > current_best_priority:
                best_conditions[bin_label] = (quantity, candidate_priority)

    # Update the result with the best conditions found
    for label in bin_labels:
        condition, _ = best_conditions[label]
        result[f'clouds_{label}_condition'] = condition

    return result

def parse_vis(vis):
    if vis is not None:
        try:
            dist = float(re.match(r'^[^\d]*(\d+)', '> 2300m').groups()[0])
            if vis.distance[-2:].lower() == 'km':
                dist *= 1000.0
            return vis.distance
        except:
            return np.nan
    else:
        return np.nan

def parse_metar_string(metar_str):
    try:
        parser = MetarParser()
        # return {'metar': parser.parse(metar_str)}
        metar = parser.parse(metar_str)
        # TODO: check units
        line = {
            'station': metar.station,
            'wind_dir': metar.wind.degrees if metar.wind is not None else np.nan,
            'wind_speed': metar.wind.speed if metar.wind is not None else 0,
            'temperature_c': metar.temperature,
            'dewpoint_c': metar.dew_point,
            'altimeter': metar.altimeter,
            'visibility': parse_vis(metar.visibility),
            # Add more fields as needed
        }
        clouds = bin_clouds(metar.clouds)
        line.update(clouds)
        return line
    except ValueError as ve:
        log.error(f"Skipping metar string {metar_str}: {ve}")
        return {}
    # try:
    #     m = Metar.Metar(metar_str)
    #     wind_speed = m.wind_speed.value() if m.wind_speed else np.nan
    #     wind_dir = m.wind_dir.value() if m.wind_dir else np.nan
    #     visibility = m.vis.value() if m.vis else np.nan
    #     temperature = m.temp.value(units='C') if m.temp else np.nan
    #     dewpoint = m.dewpt.value(units='C') if m.dewpt else np.nan
    #     pressure = m.press.value('hPa') if m.press else np.nan
    #     try:
    #         weather = m.present_weather() if m.weather else ''
    #     except KeyError as e:
    #         print(f'Error parsing weather data {e}, setting to None')
    #         weather = None
    #     # cloud = ';'.join([','.join([str(i) for i in alt]) for alt in m.sky])
    #     cloud = m.sky
    #     return {
    #         'wind_speed': wind_speed,
    #         'wind_dir': wind_dir,
    #         'visibility': visibility,
    #         'temperature': temperature,
    #         'dewpoint': dewpoint,
    #         'pressure': pressure,
    #         'weather': weather,
    #         'cloud': cloud
    #     }
    # except Metar.ParserError:
    #     return {
    #         'wind_speed': None,
    #         'wind_dir': None,
    #         'visibility': None,
    #         'temperature': None,
    #         'dewpoint': None,
    #         'pressure': None,
    #         'weather': None,
    #         'cloud': None
    #     }

def parse_metar_file(file):
    metar_data = []
    # encoding = detect_file_encoding(file)
    encoding = 'utf-8'
    try:
        with open(file, 'r', encoding=encoding, errors='ignore') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                if i + 1 >= len(lines):
                    # Skip if there's an incomplete pair
                    continue
                date_line = lines[i].strip()
                data_line = lines[i + 1].strip()
                # print(date_line, data_line, sep='\n')
                date = pd.to_datetime(date_line, format='%Y/%m/%d %H:%M')
                # if airport_code in data_line:
                try:
                    parsed_data = parse_metar_string(data_line)
                    parsed_data['timestamp'] = date
                    metar_data.append(parsed_data)
                except Exception as e:
                    log.error(f"Error reading line {file}:{i} skipping: {e}")
                    # print(traceback.format_exc())
    except Exception as e:
        log.error(f"Error reading file {file} with encoding {encoding} skipping: {e}")
    df = pd.DataFrame(metar_data)
    df.astype({'station': 'str', 'wind_dir': 'float16', 'wind_speed': 'float16', 'visibility': 'str',
               'temperature_c': 'float16', 'dewpoint_c': 'float16', 'altimeter': 'float16',
               'clouds_0_2000_condition': 'category', 'clouds_2001_6000_condition': 'category',
               'clouds_6001_12000_condition': 'category', 'clouds_12001_20000_condition': 'category'})
    return df

def parse_taf_line(line):
    tokens = ' '.join(line).split()
    parser = TAFParser()
    for i in reversed(range(len(tokens))):
        taf_str = ' '.join(tokens[:i])
        try:
            taf = parser.parse(taf_str)
            ret = {
                'station': taf.station,
                # TODO: should we include valid to (taf.validity.end_day /end_hour?
                'forecast_wind_dir_deg': taf.wind.degrees if taf.wind is not None else np.nan,
                'forecast_wind_speed_kt': taf.wind.speed if taf.wind is not None else 0,
                'forecast_visibility': parse_vis(taf.visibility),
            }
            ret.update(bin_clouds(taf.clouds))
            return ret
        # Add more fields as needed
        except Exception as e:
            continue
    log.error(f"Error parsing TAF: {' '.join(line)}")
    return {}

def parse_taf_file(file):
    taf_data = []
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    current_timestamp = None
    current_taf = []

    # for line in tqdm(lines, leave=False, desc=file):
    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        # TODO: maybe we should consider ammendments
        # Detect timestamp lines (e.g., '2022/12/06 12:10')
        timestamp_match = re.match(r'^(\d{4}/\d{2}/\d{2} \d{2}:\d{2})$', line)
        if timestamp_match:
            if current_taf:
                # Save the previous TAF report before starting a new one
                taf_line = parse_taf_line(current_taf)
                try:
                    taf_line['timestamp'] = current_timestamp
                except Exception as e:
                    pass
                taf_data.append(taf_line)
                current_taf = []

            # Update the current timestamp
            try:
                current_timestamp = pd.to_datetime(timestamp_match.group(1), format='%Y/%m/%d %H:%M')
            except ValueError as ve:
                log.error(f"Invalid timestamp format in file {file}: {ve}")
                current_timestamp = None
            continue

        # If not a timestamp line, it's part of the TAF report
        if current_timestamp:
            current_taf.append(line)

    # After the loop, save the last TAF report
    if current_taf:
        taf_line = parse_taf_line(current_taf)
        taf_line['timestamp'] = current_timestamp
        taf_data.append(taf_line)
        
    df = pd.DataFrame(taf_data)
    df.astype({'clouds_0_2000_condition': 'category', 'clouds_12001_20000_condition': 'category',
               'clouds_2001_6000_condition': 'category', 'clouds_6001_12000_condition': 'category', 'forecast_visibility': 'category', # TODO: make string
               'forecast_wind_dir_deg': 'float16', 'forecast_wind_speed_kt': 'float16', 'station': 'category'}
              )

    return df

def load_fuser(airport_code, type, data_dir, types, desc='', leave=False):
    # runways_files = glob.glob(os.path.join(data_dir, 'FUSER_test', airport_code, '**', f'{airport_code}_*runways_data_set.csv'), recursive=True)
    files = glob.glob(os.path.join(data_dir, airport_code, f'{airport_code}*{type}_data_set.csv'), recursive=True)
    time_cols   =   [col for col, val in types.items() if val == 'datetime']
    non_time_cols = {col:val for col, val in types.items() if val != 'datetime'}

    with Pool(max(1, cpu_count() - 1)) as pool:
        results = list(tqdm(
                pool.imap(partial(pd.read_csv,
                                  usecols=list(types.keys()),
                                  dtype=non_time_cols,
                                  # parse_dates=time_cols # TODO:
                                  ), files)
            , total=len(files), desc=desc, leave=leave))

    # for file in tqdm(files, desc=desc, leave=leave):
    #     data.append(pd.read_csv(file))
    # arrivals_df = pd.concat(data)
    # arrivals_df[time_col] = pd.to_datetime(arrivals_df[time_col])
    # arrivals_df.set_index(time_col, inplace=True)
    # return arrivals_df
    df = pd.concat(results)
    # if drop_cols:
    #     df.drop(drop_cols, axis=1, inplace=True)

    return df.drop_duplicates()

def load_data(start, end, files, interval, parser, desc='', leave=False):
    filtered_files = filter_file_time(files, start, end, interval=interval)

    # for file in tqdm(taf_files, desc="Processing TAF files"):
    #     results.append(parse_taf_file(file))
    with Pool(max(1, cpu_count() - 1)) as pool:
        results = list(tqdm(pool.imap(parser, filtered_files), total=len(filtered_files), desc=desc, leave=leave))

    # Flatten the results into one array # TODO: use numpy or something
    data = pd.concat(results)
    return data

def load_all_data(files, parser, desc='', leave=False):
    # for file in tqdm(taf_files, desc="Processing TAF files"):
    #     results.append(parse_taf_file(file))
    with Pool(max(1, cpu_count() - 1)) as pool:
        results = list(tqdm(pool.imap(parser, files), total=len(files), desc=desc, leave=leave))

    # Flatten the results into one array # TODO: use numpy or something
    data = pd.concat(results)
    return data


def preprocess_metar_taf(data_dir, out_dir, test=False):
    if test:
        mode = 'test'
        metar_files = glob.glob(os.path.join(data_dir, 'METAR_test', 'metar.*.txt'))
        taf_files = glob.glob(os.path.join(data_dir, 'TAF_test', 'taf.*.txt'))
    else:
        mode = 'train'
        metar_files = glob.glob(os.path.join(data_dir, 'METAR_train', '**', 'metar.*.txt'))
        taf_files = glob.glob(os.path.join(data_dir, 'TAF_train', 'taf.*.txt'))
    # metar_df = load_data(start,
    #                      end,
    #                      metar_files,
    #                      timedelta(hours=1),
    #                      parse_metar_file, desc='Metar Data'
    #                      )

    metar_df = load_all_data(
        metar_files,
        parse_metar_file, desc='Metar Data'
    )
    # metar_df.to_hdf(os.path.join(out_dir, f'metar_{mode}.h5'), key=f'metar_{mode}', format='table')
    metar_df.to_parquet(os.path.join(out_dir, 'metar.parquet'))
    del metar_df

    # taf_df = load_data(start,
    #                    end,
    #                    taf_files,
    #                    timedelta(hours=6),
    #                    parse_taf_file, desc='TAF Data')

    taf_df = load_all_data(
        taf_files,
        parse_taf_file, desc='TAF Data')

    # taf_df.to_hdf(os.path.join(out_dir, f'taf_{mode}.h5'), key=f'taf_{mode}', format='table')
    taf_df.to_parquet(os.path.join(out_dir, 'taf.parquet'))
    del metar_df


def preprocess_fuser(airport, data_dir, out_dir, test=False, leave=True):
    if test:
        mode = 'test'
        fuser_path = os.path.join(data_dir, 'FUSER_test')
    else:
        mode = 'train'
        fuser_path = data_dir

    os.makedirs(out_dir, exist_ok=True)

    # TODO: load 'configs', fun facts about the airport

    first_position_df = load_fuser(airport,
                           'first_position',
                           types={'gufi': 'str',
                                  'time_first_tracked': 'datetime',
                                  },
                           data_dir=fuser_path,
                           desc=f'Loading first position',
                           leave=leave,
                           )
    first_position_df.set_index('gufi', inplace=True)
    first_position_df.to_parquet(os.path.join(out_dir, 'first_position.parquet'))

    lamp_df = load_fuser(airport,
                         'LAMP',
                         types={'timestamp': 'datetime', 'forecast_timestamp': 'datetime', 'temperature': 'float16',
                                'wind_direction': 'float16', 'wind_speed': 'float16', 'wind_gust': 'float16',
                                'cloud_ceiling': 'float16', 'visibility': 'float16', 'cloud': 'category',
                                'lightning_prob': 'category', 'precip': 'bool'},
                         data_dir=fuser_path,
                         desc=f'Loading lamp',
                         leave=leave
                         )
    lamp_df.to_parquet(os.path.join(out_dir, 'lamp.parquet'))

    # Actual arrival times
    runway_df = load_fuser(airport,
                           'runways',
                           types={'gufi': 'str',
                                  'arrival_runway_actual_time': 'datetime',
                                  'arrival_runway_actual': 'category',
                                  },
                           data_dir=fuser_path,
                           desc=f'Loading runways',
                           leave=leave,
                           )
    runway_df = runway_df[runway_df['arrival_runway_actual_time'].notna()]
    runway_df.drop_duplicates(subset='gufi', inplace=True)
    runway_df = runway_df.set_index('gufi')
    # runway_df.to_hdf(os.path.join(out_dir, 'runway_df.h5'), key='runway_df', format='table')
    runway_df.to_parquet(os.path.join(out_dir, 'runway_df.parquet'))
    del runway_df

    # Predicted arrival times
    tfm_df = load_fuser(airport,
                        'TFM_track',
                        types={'gufi': 'str',
                               'timestamp': 'datetime',
                               'arrival_runway_estimated_time': 'datetime',
                               },
                        data_dir=fuser_path,
                        desc=f'Loading TFM_track',
                        leave=leave)
    tfm_df.rename(columns={'timestamp': 'timestamp_arrival_runway_estimate'}, inplace=True)
    tfm_df.set_index('gufi', inplace=True)
    # tfm_df = tfm_df.groupby(level='gufi').agg(list)
    # tfm_df.to_hdf(os.path.join(out_dir, 'tfm_df.h5'), key='tfm_df', format='table')
    tfm_df.to_parquet(os.path.join(out_dir, 'tfm_df.parquet'))
    # print('merging TFM data')
    # fuser_data = tfm_df.join(flights_df, on='gufi')
    # # Free the memory of the un used df
    del tfm_df
    gc.collect()

    MFS_df = load_fuser(airport,
                        'MFS',
                        data_dir=fuser_path,
                        desc=f'Loading MFS',
                        leave=leave,
                        types={'gufi': 'str',
                               'aircraft_engine_class': 'category',
                               'aircraft_type': 'category',
                               'arrival_aerodrome_icao_name': 'category',
                               'major_carrier': 'category',
                               'flight_type': 'category',
                               },
                        )
    MFS_df = MFS_df.set_index('gufi')
    # fuser_data = fuser_data.join(MFS_df, on='gufi')
    # MFS_df.to_hdf(os.path.join(out_dir, 'MFS_df.h5'), key='MFS_df', format='table')
    MFS_df.to_parquet(os.path.join(out_dir, 'MFS_df.parquet'))
    del MFS_df
    gc.collect()
    TBFM_df = load_fuser(airport,
                         'TBFM',
                         types={'gufi': 'str',
                                'timestamp': 'datetime',
                                'arrival_runway_sta': 'datetime', },
                         data_dir=fuser_path,
                         desc=f'Loading TBFM',
                         leave=leave,
                         )
    TBFM_df = TBFM_df.set_index('gufi')
    TBFM_df.rename(columns={'timestamp': 'arrival_runway_sta_time_stamp'}, inplace=True)
    # TBFM_df = TBFM_df.dropna(subset=['arrival_runway_sta']).groupby('gufi').agg(list)
    # TBFM_df.to_hdf(os.path.join(out_dir, 'TBFM_df.h5'), key='TBFM_df', format='table')
    TBFM_df.to_parquet(os.path.join(out_dir, 'TBFM_df.parquet'))
    del TBFM_df
    gc.collect()

    # TODO:
    fuser_types = {'ETD', 'LAMP', 'configs', 'first_position'}

    # fuser_data = fuser_data.join(TBFM_df, on='gufi')
    # fuser_data.to_hdf('fuser_train.h5', key='fuser_data', format='table')


def fill_missing_estimated_times(df, fallback):
    """
    Fills missing values in the 'arrival_runway_estimated_time' column using other columns
    in a predefined order of preference.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the relevant columns.

    Returns:
        pd.DataFrame: The DataFrame with filled 'arrival_runway_estimated_time'.
    """
    # Define the columns to use for filling in order of preference
    fill_columns = [
        'arrival_runway_sta_time_stamp',
        'arrival_runway_sta',
        'timestamp_arrival_runway_estimate'
    ]

    for col in fill_columns:
        # Use the values from the current column to fill nulls in 'arrival_runway_estimated_time'
        df['arrival_runway_estimated_time'] = df['arrival_runway_estimated_time'].fillna(df[col])

    # Optional: Handle cases where all fallback columns are null
    # You can fill remaining nulls with a placeholder or leave them as is
    df['arrival_runway_estimated_time'].fillna(fallback, inplace=True)  # Example: placeholder value

    return df

class NASAAirportDataset(Dataset):
    def __init__(self,
                 airport_code: str,
                 data_dir,
                 scale_min = -1.0,
                 scale_max = 1.0,
                 to_tensor = True,
                 transform=None,
                 target_transform=None):
        """
        A PyTorch Dataset for NASA Airport Throughput Prediction Challenge.

        Parameters:
            airport_code (str): ICAO code of the airport (e.g., 'KJFK').
            start_dt (datetime): Start datetime for the data to load.
            end_dt (datetime): End datetime for the data to load.
            data_dir (str): Base directory for the data.
            transform (callable, optional): Optional transform to be applied
                to the features.
            target_transform (callable, optional): Optional transform to be applied
                to the targets.
        """
        self.airport_code = airport_code
        self.data_dir = data_dir
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.to_tensor = to_tensor

        self.first_position_df = pd.read_parquet(os.path.join(data_dir, 'fuser', airport_code, 'first_position.parquet'))
        self.first_position_df['time_first_tracked'] = pd.to_datetime(self.first_position_df['time_first_tracked'],
                                                                      format='mixed')

        self.lamp_df = pd.read_parquet(os.path.join(data_dir, 'fuser', airport_code, 'lamp.parquet'))
        self.lamp_df['timestamp'] = pd.to_datetime(self.lamp_df['timestamp'], format='mixed')
        self.lamp_df['forecast_timestamp'] = pd.to_datetime(self.lamp_df['forecast_timestamp'], format='mixed')

        self.runway_df = pd.read_parquet(os.path.join(data_dir, 'fuser', airport_code, 'runway_df.parquet'))
        self.runway_df['arrival_runway_actual_time'] = pd.to_datetime(self.runway_df['arrival_runway_actual_time'],
                                                                      format='mixed')

        self.mfs_df = pd.read_parquet(os.path.join(data_dir, 'fuser', airport_code, 'MFS_df.parquet'))
        self.tbfm_df = pd.read_parquet(os.path.join(data_dir, 'fuser', airport_code, 'TBFM_df.parquet'))
        self.tbfm_df['arrival_runway_sta_time_stamp'] = pd.to_datetime(self.tbfm_df['arrival_runway_sta_time_stamp'],
                                                                      format='mixed')
        self.tbfm_df['arrival_runway_sta'] = pd.to_datetime(self.tbfm_df['arrival_runway_sta'],
                                                                      format='mixed')
        self.tfm_df = pd.read_parquet(os.path.join(data_dir, 'fuser', airport_code, 'tfm_df.parquet'))
        self.tfm_df['timestamp_arrival_runway_estimate'] = pd.to_datetime(
            self.tfm_df['timestamp_arrival_runway_estimate'],
            format='mixed')
        self.tfm_df['arrival_runway_estimated_time'] = pd.to_datetime(self.tfm_df['arrival_runway_estimated_time'],
                                                                      format='mixed')

        for df_name in ['mfs_df', 'tbfm_df', 'tfm_df', 'runway_df', 'lamp_df', 'first_position_df']:
            df = getattr(self, df_name)

            # Convert object columns to category
            new_cat_cols = df.select_dtypes(include=['object']).columns
            for col in new_cat_cols:
                df[col] = df[col].astype('category')

            cat_cols = df.select_dtypes(include=['category']).columns
            df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)

            for one_hot_col in df.select_dtypes(include=['bool']).columns:
                df[one_hot_col] = df[one_hot_col].astype('int')

            # Identify numeric columns (excluding datetime)
            datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if c not in datetime_cols]

            # Scale numeric columns to [-1, 1]
            for col in numeric_cols:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max != col_min:
                    df[col] = ((df[col] - col_min) / (col_max - col_min)) * (self.scale_max - self.scale_min) + self.scale_min
                else:
                    # Constant column, just set it to 0
                    df[col] = (self.scale_min + self.scale_max) / 2
            setattr(self, df_name, df)


        self.transform = transform
        self.target_transform = target_transform

        # Load and preprocess the data

    def __len__(self):
        return len(self.runway_df)

    def __getitem__(self, timestamp):
        end_time = timestamp + timedelta(hours=3)
        runway_rows = self.runway_df.loc[(self.runway_df['arrival_runway_actual_time'] > timestamp) & (
                    self.runway_df['arrival_runway_actual_time'] < end_time)].copy()

        runway_rows = runway_rows.join(self.mfs_df, on='gufi')
        # This needs to be done here as to get the latest flight data
        runway_rows = runway_rows.join(self.tbfm_df, on='gufi').groupby(level='gufi').agg(max)
        runway_rows = runway_rows.join(self.tfm_df, on='gufi').groupby(level='gufi').agg(max)

        runway_rows = fill_missing_estimated_times(runway_rows, end_time)

        # Sort by time for merge_asof
        runway_rows_sorted = runway_rows.reset_index().sort_values('arrival_runway_estimated_time')
        lamp_sorted = self.lamp_df.sort_values('forecast_timestamp')

        # Perform asof merge to find nearest forecast_timestamp for each arrival_runway_estimated_time
        merged = pd.merge_asof(
            runway_rows_sorted,
            lamp_sorted,
            left_on='arrival_runway_estimated_time',
            right_on='forecast_timestamp',
            direction='nearest'
        )

        # Keep only rows that have a match from lamp_df (i.e., forecast_timestamp is not NaN)
        merged.dropna(subset=['forecast_timestamp'], inplace=True)

        # If needed, restore the original index. Assuming 'gufi' was the unique index:
        if 'gufi' in runway_rows.index.names:
            merged.set_index('gufi', inplace=True, drop=True)

        runway_rows = merged
        runway_rows.join(self.first_position_df[self.first_position_df['time_first_tracked'] < timestamp].notna(),
                         on='gufi')

        runway_rows.drop([], inplace=True)

        # Convert all datetime columns in x to scaled time deltas
        datetime_cols = runway_rows.select_dtypes(include=['datetime64[ns]']).columns
        # 3 hours = 10800 seconds. We'll map: timestamp-3h -> -1, timestamp -> 0, timestamp+3h -> 1
        time_window_seconds = 3 * 3600.0

        for col in datetime_cols:
            # Convert to time delta in seconds relative to timestamp
            deltas = (runway_rows[col] - timestamp).dt.total_seconds()
            # Scale to [-1, 1] by dividing by 10800 (3 hours)
            runway_rows[col] = deltas / time_window_seconds
            # TODO: How do we deal with missing tiemes?!?!? setting to max value for now
            runway_rows[col].fillna(self.scale_max, inplace=True)

        y = runway_rows['arrival_runway_actual_time']
        x = runway_rows.drop('arrival_runway_sta_time_stamp', axis=1)

        if self.to_tensor:
            x = x.to_numpy(dtype='float32')
            x = torch.tensor(x, dtype=torch.float32)

            y = y.to_numpy(dtype='float32')
            y = torch.tensor(y, dtype=torch.float32)

            if self.transform:
                x = self.transform(x)
            if self.target_transform:
                y = self.target_transform(y)

        return x, y


if __name__ == '__main__':

    # this will take a while
    # train_ds = NASAAirportDataset('KCLT', 'data/preprocess/train')

    preprocess_metar_taf('data', os.path.join('data','preprocess', 'test'), test=False)
    preprocess_metar_taf('data', os.path.join('data','preprocess', 'test'), test=True)


    start = datetime(2022, 9, 30, 0, 0, 0)  # Start of the range
    end = datetime(2023, 9, 30, 23, 0)  # End of the range
    n_samples = 100
    time_samples = [start + timedelta(seconds=random.randint(0, int((end - start).total_seconds()))) for _ in
                    range(n_samples)]

    for i in time_samples:
        print('Flights at time', i)
        inputs, targets = train_ds[i]
        print(inputs)
        print(targets)
        # for flight_data in train_ds[i]:

    fuser_types = set(
        [re.match(r'.*\.(.*)_data_set.csv', os.path.basename(file)).group(1) for file in glob.glob("data/KATL/*.csv")])
    airports = set([os.path.basename(file) for file in glob.glob("data/FUSER_test/*")])
    print(fuser_types)

    for term in (pbar := tqdm(airports)):
        pbar.set_description(f'Loading: {term}')
        # print('preprocessing:', term)
        preprocess_fuser(term, 'data', os.path.join('data','preprocess', 'train', 'fuser', term), test=False, leave=False)
        preprocess_fuser(term, 'data', os.path.join('data','preprocess', 'test',  'fuser', term), test=True,  leave=False)

