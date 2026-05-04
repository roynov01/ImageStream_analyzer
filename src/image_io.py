"""Image IO utilities: load channels and create overlays."""
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image
from typing import Optional, Tuple


def load_image(path: Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    # tifffile returns numpy arrays
    arr = tifffile.imread(str(p))
    # Ensure 2D (if multi-page, take first)
    if arr.ndim > 2:
        # try to collapse channel axis if present
        arr = arr.squeeze()
    return arr.astype(np.float32)


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn = img.min()
    mx = img.max()
    if mx == mn:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def overlay_images(bf: np.ndarray, dapi: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Return an RGB PIL Image with DAPI over BF using alpha transparency."""
    bf_n = normalize_image(bf)
    dapi_n = normalize_image(dapi)

    # Create RGB: BF as gray -> mapped to RGB, DAPI as blue channel
    h, w = bf_n.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = bf_n  # R
    rgb[..., 1] = bf_n  # G
    rgb[..., 2] = bf_n  # B as baseline

    # Add DAPI into blue channel with alpha
    rgb[..., 2] = (1 - alpha) * rgb[..., 2] + alpha * dapi_n

    rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(rgb)
