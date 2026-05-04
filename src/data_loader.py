"""Data loading utilities: parse features file, channel map, and index images."""
from pathlib import Path
import json
import pandas as pd
from typing import Dict


def load_channel_map(path: Path) -> Dict[str, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Channel map not found: {path}")
    with open(path, 'r') as fh:
        mapping = json.load(fh)
    return mapping


def load_features_table(path: Path, skiprows: int = 2) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    # Read TSV, skip first `skiprows` lines
    df = pd.read_csv(path, sep='\t', header=0, skiprows=skiprows)
    # Normalize column name for object id
    # Possible column names: 'Object Number' or 'ObjectNumber' etc.
    candidates = [c for c in df.columns if 'object' in c.lower() and 'number' in c.lower()]
    if not candidates:
        # fallback to first column
        df = df.rename(columns={df.columns[0]: 'Object Number'})
    else:
        df = df.rename(columns={candidates[0]: 'Object Number'})
    return df


def index_images(image_folder: Path, df: pd.DataFrame, channel_map: Dict[str, str]) -> pd.DataFrame:
    """
    Add image path columns to dataframe for each mapped channel.

    Files are expected as: {ObjectNumber}_Ch{ChannelNumber}.ome.tif
    """
    image_folder = Path(image_folder)
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    # For each channel key like 'Ch1', create a column with path or NaN
    for ch_key in channel_map.keys():
        col_name = channel_map[ch_key]
        df[col_name + '_path'] = None

    for idx, row in df.iterrows():
        obj = str(row['Object Number'])
        for ch_key, ch_name in channel_map.items():
            # build filename
            # example: 0_Ch1.ome.tif
            fname = f"{obj}_{ch_key}.ome.tif" if ch_key.startswith('Ch') else f"{obj}_{ch_key}.ome.tif"
            p = image_folder / fname
            if p.exists():
                df.at[idx, ch_name + '_path'] = str(p)
            else:
                # try without .ome (some files may be .tif)
                p2 = image_folder / fname.replace('.ome', '')
                if p2.exists():
                    df.at[idx, ch_name + '_path'] = str(p2)
    return df
