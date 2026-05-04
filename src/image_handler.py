"""
Image handling utilities for OME-TIFF and other image formats.
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import Union, Optional


def load_ome_tiff(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load OME-TIFF image file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the OME-TIFF file
        
    Returns
    -------
    np.ndarray
        Image data as numpy array
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with tifffile.TiffFile(file_path) as tif:
        image_data = tif.asarray()
    
    return image_data


def save_image(image_data: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Save image data to a file.
    
    Parameters
    ----------
    image_data : np.ndarray
        Image data to save
    output_path : str or Path
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    tifffile.imwrite(str(output_path), image_data)
