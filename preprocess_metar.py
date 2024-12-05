import logging
import os
import numpy as np
import pandas as pd
import glob
from metar import Metar
import multiprocessing as mp
import logging as log
import sys
import threading
import time
from functools import partial

from tqdm import tqdm

print(sys.version)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    filename='preprocess.log',
    filemode='w',
)

file_locks = {}  # Dictionary to manage locks for each output file

# Function to get a lock for each file
def get_file_lock(file_path):
    if file_path not in file_locks:
        file_locks[file_path] = threading.Lock()
    return file_locks[file_path]

def write_hdf5(data, output_path):
    metar_df = pd.DataFrame(data)
    file_lock = get_file_lock(output_path)
    with file_lock:  # Ensure only one thread writes to the file at a time
        with pd.HDFStore(output_path, mode='a', complevel=9, complib='bzip2') as hdf:
            for dest_airport, group in metar_df.groupby('airport_code'):
                key = f'/{dest_airport}'
                if key in hdf:
                    existing_df = hdf[key]
                    updated_df = pd.concat([existing_df, group], ignore_index=True)
                    hdf.put(key, updated_df, format='table')
                else:
                    hdf.put(key, group, format='table')


def parse_metar_string(metar_str):
    try:
        m = Metar.Metar(metar_str)
        wind_speed = m.wind_speed.value() if m.wind_speed else np.nan
        wind_dir = m.wind_dir.value() if m.wind_dir else np.nan
        visibility = m.vis.value() if m.vis else np.nan
        temperature = m.temp.value(units='C') if m.temp else np.nan
        dewpoint = m.dewpt.value(units='C') if m.dewpt else np.nan
        pressure = m.press.value('hPa') if m.press else np.nan
        try:
            weather = m.present_weather() if m.weather else ''
        except KeyError as e:
            logging.error(f'Error parsing weather data {e}, setting to None')
            weather = None
        cloud = ';'.join([','.join([str(i) for i in alt]) for alt in m.sky])
        return {
            'wind_speed': wind_speed,
            'wind_dir': wind_dir,
            'visibility': visibility,
            'temperature': temperature,
            'dewpoint': dewpoint,
            'pressure': pressure,
            'weather': weather,
            'cloud': cloud
        }
    except Metar.ParserError:
        return {
            'wind_speed': None,
            'wind_dir': None,
            'visibility': None,
            'temperature': None,
            'dewpoint': None,
            'pressure': None,
            'weather': None,
            'cloud': None
        }

def process_metar_file(file_path, output_path):
    encoding = 'utf-8'
    try:
        parsed_data = []
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                if i+1 >= len(lines):
                    continue
                date_line = lines[i].strip()
                data_line = lines[i+1].strip()
                try:
                    timestamp = pd.to_datetime(date_line, format='%Y/%m/%d %H:%M')
                except ValueError:
                    timestamp = pd.NaT
                if pd.isna(timestamp):
                    logging.error(f"Skipping {date_line} in {file_path}:{i}")
                    continue
                airport_code = data_line.split()[0] if len(data_line.split()) > 0 else None
                if airport_code:
                    parsed_weather = parse_metar_string(data_line)
                    parsed_weather['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(timestamp) else None
                    parsed_weather['airport_code'] = airport_code
                    parsed_data.append(parsed_weather)
                else:
                    logging.error(f'no airport code line: {i}, filename: {file_path}')
                    logging.error(data_line)

        if parsed_data:
            write_hdf5(parsed_data, output_path)

    except Exception as e:
        logging.error(f"Error processing file {file_path} skipping: {e}")

def preprocess_metar_data(data_dir, output_path):
    metar_files = glob.glob(os.path.join(data_dir, '**', '*.txt'))
    log.info(f"Found {len(metar_files)} METAR files to process.")

    pool = mp.Pool(mp.cpu_count() - 1)
    process_file = partial(process_metar_file, output_path=output_path)
    # pool.map(process_file, metar_files)
    for _ in tqdm(pool.imap_unordered(process_file, metar_files), total=len(metar_files),
                  desc="Processing METAR files"):
        pass

    pool.close()
    pool.join()

    log.info("Preprocessing of METAR data completed.")

if __name__ == "__main__":
    out_train = './data/METAR_train/processed.h5.bz2'
    out_test = './data/METAR_test/processed.h5.bz2'

    preprocess_metar_data('./data/METAR_train', out_train)
    preprocess_metar_data('./data/METAR_test', out_test)
