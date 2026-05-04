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

    # attempt multiple filename patterns to be robust to padding/extension differences
    pad_lengths = [0, 4, 5, 6]
    exts = ['.ome.tif', '.tif']
    for idx, row in df.iterrows():
        raw_obj = row['Object Number']
        # ensure string form
        obj_str = str(raw_obj)
        for ch_key, ch_name in channel_map.items():
            found = None
            # try without modification
            for ext in exts:
                fname = f"{obj_str}_{ch_key}{ext}"
                p = image_folder / fname
                if p.exists():
                    found = p
                    break
            # try int cast (remove decimals)
            if found is None:
                try:
                    obj_int = int(float(raw_obj))
                    for ext in exts:
                        fname = f"{obj_int}_{ch_key}{ext}"
                        p = image_folder / fname
                        if p.exists():
                            found = p
                            break
                except Exception:
                    pass
            # try zero-padded variants
            if found is None:
                for pad in pad_lengths:
                    if pad <= 0:
                        continue
                    obj_pad = str(raw_obj).zfill(pad)
                    for ext in exts:
                        fname = f"{obj_pad}_{ch_key}{ext}"
                        p = image_folder / fname
                        if p.exists():
                            found = p
                            break
                    if found:
                        break

            if found:
                df.at[idx, ch_name + '_path'] = str(found)
    return df
