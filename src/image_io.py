"""Image IO utilities: load channels and create overlays."""
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from skimage import io as skio


def load_image(path: Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    # Try multiple readers to handle problematic OME-TIFF variants
    # 1) tifffile
    try:
        arr = tifffile.imread(str(p))
    except Exception:
        arr = None

    # 2) scikit-image (imageio backend)
    if arr is None:
        try:
            arr = skio.imread(str(p))
        except Exception:
            arr = None

    # 3) PIL fallback (reads first page/frame)
    if arr is None:
        try:
            with Image.open(str(p)) as im:
                arr = np.array(im)
        except Exception as e:
            raise RuntimeError(f"Unable to read image {p}: {e}")

    # Ensure 2D: if multi-page or multi-channel, try to select a sensible plane
    arr = np.asarray(arr)
    if arr.ndim > 2:
        # common shapes: (pages, H, W) or (H, W, channels)
        if arr.shape[0] <= 4 and arr.shape[0] != arr.shape[1]:
            # assume first axis is pages -> take first page
            arr = arr[0]
        else:
            # squeeze singleton dimensions
            arr = arr.squeeze()
            if arr.ndim > 2:
                # as a last resort, take first channel/page
                arr = arr[0]

    return arr.astype(np.float32)


def normalize_image(img: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.5) -> np.ndarray:
    """Contrast-stretch an image for display using robust percentiles."""
    img = img.astype(np.float32)
    lo = float(np.nanpercentile(img, low_percentile))
    hi = float(np.nanpercentile(img, high_percentile))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(img))
        hi = float(np.nanmax(img))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def subtract_background(img: np.ndarray, background_percentile: float = 20.0) -> np.ndarray:
    """Remove a soft background floor before display/compositing."""
    img = img.astype(np.float32)
    bg = float(np.nanpercentile(img, background_percentile))
    return np.clip(img - bg, 0.0, None)


def overlay_images(bf: np.ndarray, dapi: np.ndarray, alpha: float = 0.65) -> Image.Image:
    """Return an RGB PIL Image with BF as grayscale and DAPI as a cyan overlay."""
    bf_n = normalize_image(bf)
    dapi_n = normalize_image(subtract_background(dapi))

    h, w = bf_n.shape
    rgb = np.stack([bf_n, bf_n, bf_n], axis=-1)

    # Use the DAPI intensity as a transparency mask so the BF remains visible.
    mask = np.clip(dapi_n * alpha, 0.0, 1.0)[..., None]
    dapi_rgb = np.zeros_like(rgb)
    dapi_rgb[..., 1] = dapi_n
    dapi_rgb[..., 2] = dapi_n
    rgb = rgb * (1.0 - mask) + dapi_rgb * mask

    rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(rgb)
