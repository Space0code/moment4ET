import pandas as pd
import numpy as np
import re

from pathlib import Path


def load_tobii_file(tobii_file):
    """
    Load tobii data from individual .tsv file.
    Skips metadata before 'TimeStamp', extracts timestamp from filename.
    
    :param input_path: path to file
    """

    file = Path(tobii_file)

    # Find where 'TimeStamp' appears (start of the file)
    header_line_idx = None
    with open(file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip().startswith("TimeStamp"):
                header_line_idx = i
                break

    if header_line_idx is None:
        raise ValueError(f"No 'TimeStamp' header found in {file.name}")

    try:
        df = pd.read_csv(file, sep='\t', header=header_line_idx, low_memory=False)
    except Exception as e:
        print(f"Failed to read {file.name}: {e}")
            

    # Add filename time info
    match = re.search(r'(\d{4}-\d{2}-\d{2}\$\d{2}-\d{2}-\d{2}-\d{6})', file.name)
    ts_str = match.group(1).replace('$', 'T')
    file_dt = pd.to_datetime(ts_str, format='%Y-%m-%dT%H-%M-%S-%f', errors='coerce') if match else pd.NaT

    df['source_file'] = file.name

    # Ensure numeric and drop invalid
    df['TimeStamp'] = pd.to_numeric(df['TimeStamp'], errors='coerce')
    bad = df['TimeStamp'].isna().sum()
    if bad:
        print(f"[WARN] {file.name}: dropping {bad} rows with invalid TimeStamp")
        df = df.dropna(subset=['TimeStamp'])

     # Convert relative timestamp to absolute time
    df['timestamp'] = file_dt + pd.to_timedelta(df['TimeStamp'], unit='ms') # timestamp in miliseconds

    return df


def load_tobii_data(tobii_folder):
    """
    Load tobii .tsv files for one participant.
    Skips metadata before 'TimeStamp', extracts timestamp from filename.

    :param tobii_folder: Path to folder with tobii .tsv files.
    """
    tobii_folder = Path(tobii_folder)
    tsv_files = sorted(tobii_folder.glob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No Tobii files found in {tobii_folder}")

    dfs = []

    for file in tsv_files:
        try:
            dfs.append(load_tobii_file(file))
        except Exception as e:
            print(f"[WARN] Skipping {file.name}: {e}")

    if not dfs:
        raise ValueError(f"No valid Tobii data loaded from {tobii_folder}")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values('timestamp')

    print(f"Loaded {len(df_all):,} rows from {len(tsv_files)} files.")
    return df_all



def tobii_flag_missing(df_tobii):
    """
    Add a column to tobii data (as loaded with tobii_load_data) marking missing values.
    
    :param df_tobii: Loaded tobii data (multiple files).
    """
    df_tobii = df_tobii.copy()

    df_tobii['tobii_missing'] = (df_tobii['PupilValidityLeft'] == 0) & (df_tobii['PupilValidityRight'] == 0) # pupilvaliditiy = 0 -> missing value 
    df_tobii['tobii_missing'] = df_tobii['tobii_missing'].map({True: 'missing', False: 'present'}).fillna('missing')
    
    return df_tobii



def resample_tobii(df_tobii, freq_hz=10):
    """
    Resample tobii loaded data onto desired frequency.
    
    :param df_tobii: Loaded tobii data.
    :param freq_hz: Desired resampling frequency.
    """

    df_tobii['timestamp'] = pd.to_datetime(df_tobii['timestamp'], errors='coerce')
    df_tobii = df_tobii.dropna(subset=['timestamp'])

    df_tobii = df_tobii.sort_values('timestamp')
    df_tobii = df_tobii.drop_duplicates(subset=['timestamp'])

    # set index
    df_tobii = df_tobii.set_index('timestamp')

    # resample frequency
    period_ms = int(1000 / freq_hz)
    freq_str = f'{period_ms}ms'

    # if after cleaning the index is still not monotonic, fix it
    if not df_tobii.index.is_monotonic_increasing:
        df_tobii = df_tobii.sort_index()

    # split numeric / non-numeric columns
    numeric_cols = df_tobii.select_dtypes(include='number').columns
    non_numeric_cols = df_tobii.select_dtypes(exclude='number').columns

    df_num = df_tobii[numeric_cols].resample(freq_str).max() # max or mean

    # non-numeric = forward-fill
    df_non = df_tobii[non_numeric_cols].resample(freq_str).ffill()

    # merge and reset index
    df_resampled = pd.concat([df_num, df_non], axis=1).reset_index()

    return df_resampled



def add_missing_tobii_column(df_rgb, df_tobii, tolerance_ms=120):
    """
    Adds a column 'tobii_missing' to df_rgb based on Tobii signal presence.
    Tolerates small timestamp misalignments (e.g. ±120 ms).
    """
    df_rgb = df_rgb.copy()
    df_tobii = df_tobii.copy()
    df_rgb['timestamp'] = pd.to_datetime(df_rgb['timestamp'])
    df_tobii['timestamp'] = pd.to_datetime(df_tobii['timestamp'])

    df_rgb = df_rgb.sort_values('timestamp')
    df_tobii = df_tobii.sort_values('timestamp')

    # merge nearest timestamps within tolerance
    merged = pd.merge_asof(
        df_rgb,
        df_tobii[['timestamp', 'tobii_missing']],
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(milliseconds=tolerance_ms)
    )

    # fill nans 
    merged['tobii_missing'] = merged['tobii_missing'].fillna('missing')
    return merged


def prepare_raw_tobii(tobii_data):

    # drop unneccesary cols
    tobii_data = tobii_data.drop(columns=['TimeStamp', 'Event'])

    # drop if timestamp is NaN
    tobii_data = tobii_data.dropna(subset=['timestamp']).reset_index(drop=True)

    # replace -1 with NaNs
    tobii_data.replace(-1, np.nan, inplace=True) # -1 is missing value

    # crerate mask for missing gaze 
    tobii_data['mask_gaze_fused'] = (
        tobii_data['GazePointX'].notna() & tobii_data['GazePointY'].notna()
    ).astype('float32')

    num_cols = [
    'GazePointX','GazePointY',
    'GazePointXLeft','GazePointYLeft',
    'GazePointXRight','GazePointYRight',
    'PupilSizeLeft','PupilSizeRight',
    'DistanceLeft','DistanceRight','AverageDistance',
]
    tobii_data[num_cols] = tobii_data[num_cols].fillna(0) # fill nans with 0 for training
    
    # normalise gaze to screen size
    tobii_data['GazePointX'] /= tobii_data['screen_w']
    tobii_data['GazePointY'] /= tobii_data['screen_h']
    tobii_data['GazePointXLeft'] /= tobii_data['screen_w']
    tobii_data['GazePointYLeft'] /= tobii_data['screen_h']
    tobii_data['GazePointXRight'] /= tobii_data['screen_w']
    tobii_data['GazePointYRight'] /= tobii_data['screen_h']

    return tobii_data.drop(columns=['screen_h', 'screen_w'])