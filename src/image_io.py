"""Image IO utilities: load channels and create overlays."""
from pathlib import Path
import xml.etree.ElementTree as ET

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


def get_pixel_size_um(path: Path) -> float | None:
    """Return the OME pixel size in microns when available, otherwise None."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with tifffile.TiffFile(str(p)) as tf:
            page = tf.pages[0] if tf.pages else None
            xml = getattr(tf, 'ome_metadata', None)
            if xml:
                root = ET.fromstring(xml)
                pixels = root.find('.//{*}Pixels')
                if pixels is not None:
                    for key in ('PhysicalSizeX', 'PhysicalSizeY'):
                        raw_val = pixels.attrib.get(key)
                        if raw_val is None:
                            continue
                        try:
                            value = float(raw_val)
                        except Exception:
                            continue
                        unit = (pixels.attrib.get(f'{key}Unit') or '').strip().lower()
                        if not unit or unit in {'um', 'µm', 'micrometer', 'micrometre'}:
                            return value

            if page is not None:
                resolution_unit = int(page.tags.get('ResolutionUnit').value) if 'ResolutionUnit' in page.tags else None
                x_resolution = page.tags.get('XResolution')
                if x_resolution is not None and resolution_unit in {2, 3}:
                    try:
                        x_num, x_den = x_resolution.value
                        if x_den:
                            pixels_per_unit = float(x_num) / float(x_den)
                            if pixels_per_unit > 0:
                                if resolution_unit == 2:
                                    inches_per_pixel = 1.0 / pixels_per_unit
                                    return inches_per_pixel * 25400.0
                                if resolution_unit == 3:
                                    cm_per_pixel = 1.0 / pixels_per_unit
                                    return cm_per_pixel * 10000.0
                    except Exception:
                        pass

                # ImageStream/Amnis files often expose a calibrated field width in this vendor tag.
                vendor_width = page.tags.get(33012)
                if vendor_width is not None and page.imagewidth:
                    try:
                        field_width_um = float(vendor_width.value)
                        pixel_size_um = field_width_um / float(page.imagewidth)
                        if pixel_size_um > 0:
                            return pixel_size_um
                    except Exception:
                        pass
    except Exception:
        return None
    return None


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


def overlay_images(
    bf: np.ndarray,
    dapi: np.ndarray,
    alpha: float = 0.65,
    dapi_min: float | None = None,
    dapi_max: float | None = None,
    dapi_scale: float | None = None,
    dapi_gain: float = 1.0,
) -> Image.Image:
    """Return an RGB image with BF as base and DAPI overlaid in cyan.

    If ``dapi_min``/``dapi_max`` are provided, DAPI normalization is fixed across cells:
    dapi_norm = clip((dapi_bg_subtracted - dapi_min) / (dapi_max - dapi_min), 0, 1).
    Otherwise, if ``dapi_scale`` is provided, use the older scale/gain path.
    """
    bf_n = normalize_image(bf)
    dapi_bg = subtract_background(dapi)
    if dapi_min is not None and dapi_max is not None and np.isfinite(dapi_min) and np.isfinite(dapi_max) and dapi_max > dapi_min:
        dapi_n = np.clip((dapi_bg - float(dapi_min)) / (float(dapi_max) - float(dapi_min)), 0.0, 1.0)
    elif dapi_scale is not None and np.isfinite(dapi_scale) and dapi_scale > 0:
        dapi_n = np.clip((dapi_bg / float(dapi_scale)) * float(dapi_gain), 0.0, 1.0)
    else:
        dapi_n = np.clip(normalize_image(dapi_bg) * float(dapi_gain), 0.0, 1.0)

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
