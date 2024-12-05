#!/usr/bin/env python
# coding: utf-8


import os
from multiprocessing import Pool, cpu_count
from os import cpu_count
import pandas as pd
import datetime
from datetime import datetime, timedelta
import re
import glob
import logging as log
from tqdm import tqdm
from metar_taf_parser.parser.parser import MetarParser, TAFParser

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

# from metar import Metar

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

def get_df(data, start, end):
    df = pd.DataFrame(data)
    df.dropna(subset=['timestamp'], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
    df.set_index('timestamp', inplace=True)
    return df

def parse_metar_string(metar_str):
    try:
        parser = MetarParser()
        return {'metar': parser.parse(metar_str)}
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
    except Exception as e:
        log.error(f"Error reading file {file} with encoding {encoding} skipping: {e}")
    return metar_data

def parse_taf_line(line):
    tokens = ' '.join(line).split()
    parser = TAFParser()
    for i in reversed(range(len(tokens))):
        taf_str = ' '.join(tokens[:i])
        try:
            return parser.parse(taf_str)
        except Exception as e:
            continue
    log.error(f"Error parsing TAF: {' '.join(line)}")
    return None

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
                taf_data.append({'timestamp': current_timestamp, 'taf': parse_taf_line(current_taf)})
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
        taf_data.append({'timestamp': current_timestamp, 'taf': parse_taf_line(current_taf)})

    return taf_data

def load_data(start, end, files, interval, parser, desc='', leave=False):
    """
    Loads and parses TAF data for a specific airport within a time range.

    Parameters:
        # airport_code (str): ICAO code of the airport (e.g., 'SCEL').
        start (datetime): Start datetime.
        end (datetime): End datetime.
        data_dir (str): Base directory containing TAF files.

    Returns:
        pd.DataFrame: DataFrame with 'timestamp' and 'taf' columns.
    """
    files = filter_file_time(files, start, end, interval=interval)

    # for file in tqdm(taf_files, desc="Processing TAF files"):
    #     results.append(parse_taf_file(file))
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(parser, files), total=len(files), desc=desc, leave=leave))

    # Flatten the results into one array # TODO: use numpy or something
    data = [item for row in results for item in row]
    return get_df(data, start, end)


if __name__ == '__main__':
    fuser_types = set([re.match(r'.*\.(.*)_data_set.csv', os.path.basename(file)).group(1) for file in glob.glob("data/KATL/*.csv")])
    airports = set([os.path.basename(file) for file in glob.glob("data/FUSER_test/*")])

    airport = airports.pop()
    data = {}
    for fuser in fuser_types:
        data[fuser] = load_fuser(airport,
                                 fuser,
                                 data_dir='data',
                                 desc=f'Loading {fuser}',
                                 leave=True)


    actual_arrival_df = load_actual_arrivals('KJFK')
    print(actual_arrival_df)
    actual_arrival_df = load_fuser('KJFK', 'runways', data_dir='data', desc='Loading actual arrival times', leave=True)

    print(actual_arrival_df)
    print(actual_arrival_df)
    est_arrival_df = load_estimated_arrivals('KJFK')
    start = datetime(2022, 9, 1, 10, 0)
    end = datetime(2022, 9, 1, 10, 30)
    metar_df = load_data(start,
                         end,
                         glob.glob('./data/METAR_train/**/metar.*.txt'),
                         timedelta(hours=1),
                         parse_metar_file, desc='Metar Data')
    metar = metar_df.iloc[0]['metar']
    print(f"Station: {metar.station}")
    print(f"Observation Time: {metar.day:02d} {metar.time.hour:02d}:{metar.time.minute:02d} UTC")
    print(f"Wind: {metar.wind.direction} at {metar.wind.speed} {metar.wind.unit}")
    print(f"Visibility: {metar.visibility.distance}")
    print(f"Temperature: {metar.temperature}°C")
    print(f"Dew Point: {metar.dew_point}°C")
    print(f"Pressure: {metar.altimeter} {metar.altimeter}")
    if metar.remarks:
        print(f"Remarks: {metar.remarks}")

    # taf_df = load_taf_data(datetime(2022, 9, 1, 10, 0), datetime(2022, 9, 30, 11, 30), './data/TAF_train')
    # taf_df = load_data(datetime(2022, 9, 1, 10, 0),
    #                    datetime(2022, 9, 30, 11, 30),
    #                    glob.glob('./data/TAF_train/taf.*.txt'),
    #                    timedelta(hours=6),
    #                    parse_taf_file)
    #
    # print(taf_df)
    #
    #
    # print(len(taf_df.loc[taf_df['taf'].isnull()]))
    # print(len(taf_df.loc[taf_df['taf'].notnull()]))
    # print(len(taf_df.loc[taf_df['taf'].isnull()]) / len(taf_df.loc[taf_df['taf'].notnull()]))

    pass