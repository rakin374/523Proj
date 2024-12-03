import os
import h5py
import numpy as np
import pandas as pd
import glob
from metar import Metar
import multiprocessing as mp
from tqdm import tqdm


# Ensure the processed directory exists

def initialize_hdf5(file_path):
    with h5py.File(file_path, 'w') as h5f:
        # Create a group for airports
        h5f.create_group('airports')
    print(f"HDF5 file created at {file_path}")


def write_to_hdf5(h5f, airport_code, data):
    airport_group = h5f['airports']

    # If airport group doesn't exist, create datasets
    if airport_code not in airport_group:
        airport_group.create_group(airport_code)
        airport_group[airport_code].create_dataset(
            'timestamp',
            shape=(0,),
            maxshape=(None,),
            dtype='S19',  # ISO format string
            chunks=True,
            compression="gzip"
        )
        airport_group[airport_code].create_dataset(
            'wind_speed',
            shape=(0,),
            maxshape=(None,),
            dtype='f4',
            chunks=True,
            compression="gzip"
        )
        airport_group[airport_code].create_dataset(
            'wind_dir',
            shape=(0,),
            maxshape=(None,),
            dtype='f4',
            chunks=True,
            compression="gzip"
        )
        airport_group[airport_code].create_dataset(
            'visibility',
            shape=(0,),
            maxshape=(None,),
            dtype='f4',
            chunks=True,
            compression="gzip"
        )
        airport_group[airport_code].create_dataset(
            'temperature',
            shape=(0,),
            maxshape=(None,),
            dtype='f4',
            chunks=True,
            compression="gzip"
        )
        airport_group[airport_code].create_dataset(
            'dewpoint',
            shape=(0,),
            maxshape=(None,),
            dtype='f4',
            chunks=True,
            compression="gzip"
        )
        airport_group[airport_code].create_dataset(
            'pressure',
            shape=(0,),
            maxshape=(None,),
            dtype='f4',
            chunks=True,
            compression="gzip"
        )
        airport_group[airport_code].create_dataset(
            'weather',
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding='utf-8'),
            chunks=True,
            compression="gzip"
        )
        airport_group[airport_code].create_dataset(
            'cloud',
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding='utf-8'),
            chunks=True,
            compression="gzip"
        )
        print(f"Created datasets for airport {airport_code}")

    # Reference to the airport's group
    group = airport_group[airport_code]

    # Prepare data for appending
    num_new = len(data)
    for key in ['timestamp', 'wind_speed', 'wind_dir', 'visibility', 'temperature', 'dewpoint', 'pressure', 'weather',
                'cloud']:
        if key == 'timestamp':
            # Convert datetime to ISO format strings
            values = [d['timestamp'].isoformat() for d in data]
            dtype = 'S19'
        elif key in ['weather', 'cloud']:
            # Encode strings as UTF-8
            values = [d[key] for d in data]
            dtype = h5py.string_dtype(encoding='utf-8')
        else:
            values = [d[key] for d in data]
            dtype = 'f4'

        dataset = group[key]
        old_size = dataset.shape[0]
        new_size = old_size + num_new
        dataset.resize((new_size,))

        if key == 'timestamp':
            dataset[old_size:new_size] = np.array(values, dtype='S19')
        elif key in ['weather', 'cloud']:
            dataset[old_size:new_size] = values
        else:
            dataset[old_size:new_size] = np.array(values, dtype='f4')

    print(f"Appended {num_new} records to airport {airport_code}")


def parse_metar_string(metar_str):
    try:
        m = Metar.Metar(metar_str)
        wind_speed = m.wind_speed.value() if m.wind_speed else None
        wind_dir = m.wind_dir.value() if m.wind_dir else None
        visibility = m.vis.value() if m.vis else None
        temperature = m.temp.value(units='C') if m.temp else None
        dewpoint = m.dewpt.value(units='C') if m.dewpt else None
        pressure = m.press.value('hPa') if m.press else None
        try:
            weather = m.present_weather() if m.weather else None
        except KeyError as e:
            # print(f'Error parsing weather data {e}, setting to None')
            weather = None
        cloud = m.sky if m.sky else None
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

def process_metar_file(file_path, h5f, chunk_size=1000):
    encoding = 'utf-8'
    parsed_data = []
    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if i+1 >= len(lines):
                    # Skip if there's an incomplete pair
                    continue
                date_line = lines[i].strip()
                data_line = lines[i+1].strip()
                # Extract date and time from date_line
                # Example date_line: '2022/09/25 00:00'
                try:
                    timestamp = pd.to_datetime(date_line, format='%Y/%m/%d %H:%M')
                except ValueError:
                    # Handle unexpected date format
                    timestamp = pd.NaT
                if pd.isna(timestamp):
                    continue
                # Check if the data_line corresponds to the desired airport
                # Assuming airport_code is globally defined or passed as an argument
                # For flexibility, you might want to pass airport_code as a parameter
                # Here, we assume multiple airports are processed, so we extract the airport code from data_line
                # Example data_line: 'KJFK 250000Z ...'
                airport_code = data_line.split()[0] if len(data_line.split()) > 0 else None
                if airport_code:
                    parsed_weather = parse_metar_string(data_line)
                    parsed_weather['timestamp'] = timestamp
                    parsed_weather['airport_code'] = airport_code
                    parsed_data.append(parsed_weather)
                else:
                    print(f'no airport code line: {i}, filename: {file_path}')
                    print(data_line)
        if parsed_data:
            write_to_hdf5(h5f, airport_code, parsed_data)
        else:
            raise Exception()
    except Exception as e:
        print(f"Error processing file {file_path} skipping: {e}")
    return parsed_data

def preprocess_metar_data(data_dir, output_dir):
    metar_files = glob.glob(os.path.join(data_dir, '**', '*.txt'))
    print(f"Found {len(metar_files)} METAR files to process.")

    # Use multiprocessing to speed up processing
    pool = mp.Pool(mp.cpu_count() - 1)

    all_parsed_data = tqdm(pool.imap(process_metar_file, metar_files), desc="Processing METAR files", total=len(metar_files))
    tuple(all_parsed_data)
    pool.close()
    pool.join()

    # Flatten the list of lists
    all_parsed_data = [item for sublist in all_parsed_data for item in sublist]

    # Convert to DataFrame
    metar_df = pd.DataFrame(all_parsed_data)
    metar_df.dropna(subset=['timestamp', 'airport_code'], inplace=True)

    # Save per airport for efficiency
    for airport, group in metar_df.groupby('airport_code'):
        airport_df = group.drop(columns=['airport_code'])
        airport_file = os.path.join(output_dir, f"{airport}_metar.parquet")
        airport_df.to_parquet(airport_file, index=False)
        print(f"Saved processed METAR data for {airport} to {airport_file}")

    print("Preprocessing of METAR data completed.")

if __name__ == "__main__":
    out_train = './data/METAR_train/processed'
    out_test = './data/METAR_test/processed'
    os.makedirs(out_train, exist_ok=True)
    preprocess_metar_data('./data/METAR_train', out_train)
    os.makedirs(out_test, exist_ok=True)
    preprocess_metar_data('./data/METAR_test', out_test)

