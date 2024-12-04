import logging
import os
import h5py
import numpy as np
import pandas as pd
import glob
from metar import Metar
import multiprocessing as mp
from tqdm import tqdm
from sqlalchemy import *
from sqlalchemy.exc import SQLAlchemyError
import logging as log
import sys
from functools import partial

print(sys.version)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    filename='preprocess.log',
    filemode='w',
    # handlers=[
    #     logging.FileHandler("preprocess_metar.log"),
    #     logging.StreamHandler(sys.stdout)
    # ]
)

def insert_into_db(data, engine):
    chunck = 1000
    df = pd.DataFrame(data)
    if df.empty:
        return
    try:
        with engine.begin() as connection:
            log.info('Inserting into database')
            for i in range(0, len(df), chunck):
                df.iloc[i:min(i + chunck, len(df))].to_sql('metar_data', con=connection, if_exists='append', index=False, method='multi')
            log.info('done')
    except SQLAlchemyError as e:
        logging.error(f"Database insertion error: {e}")

def optimize_sqlite(engine):
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode = OFF;"))
        conn.execute(text("PRAGMA synchronous = OFF;"))
        conn.execute(text("PRAGMA cache_size = 100000;"))
        conn.execute(text("PRAGMA temp_store = MEMORY;"))

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

def process_metar_file(file_path):
    encoding = 'utf-8'
    parsed_data = []
    engine = create_engine(f'sqlite:///processed_metar.db', connect_args={'check_same_thread': False}, pool_pre_ping=True)
    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
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
                    logging.error(f"Skipping {date_line} in {file_path}:{i}")
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
                    logging.error(f'no airport code line: {i}, filename: {file_path}')
                    logging.error(data_line)
        if parsed_data:
            insert_into_db(parsed_data, engine)
        else:
            raise Exception()
    except Exception as e:
        logging.error(f"Error processing file {file_path} skipping: {e}")
    return None

def preprocess_metar_data(data_dir, output_dir, engine):
    metar_files = glob.glob(os.path.join(data_dir, '**', '*.txt'))
    log.info(f"Found {len(metar_files)} METAR files to process.")

    # Use multiprocessing to speed up processing
    pool = mp.Pool(mp.cpu_count() - 1)

    all_parsed_data = list(
        tqdm(
            pool.imap(process_metar_file, metar_files),
             desc="Processing METAR files",
             total=len(metar_files)))
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
        log.info(f"Saved processed METAR data for {airport} to {airport_file}")

    log.info("Preprocessing of METAR data completed.")

def create_engine_with_optimizations(db_path):
    # Create SQLAlchemy engine with optimizations
    engine = create_engine(f'sqlite:///{db_path}',
                           connect_args={'check_same_thread': False},
                           pool_pre_ping=True)
    return engine

def insert_into_db_optimized(data, engine):
    df = pd.DataFrame(data)
    if df.empty:
        return
    try:
        with engine.begin() as connection:
            df.to_sql('metar_data', con=connection, if_exists='append', index=False, method='multi')
    except SQLAlchemyError as e:
        log.error(f"Database insertion error: {e}")


if __name__ == "__main__":
    out_train = './data/METAR_train/processed'
    out_test = './data/METAR_test/processed'

    engine = create_engine_with_optimizations('processed_metar.db')
    optimize_sqlite(engine)

    os.makedirs(out_train, exist_ok=True)
    preprocess_metar_data('./data/METAR_train', out_train, engine)
    os.makedirs(out_test, exist_ok=True)
    preprocess_metar_data('./data/METAR_test', out_test, engine)

