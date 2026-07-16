"""
extract.py — Dominant color extraction via K-means clustering.

Loads an image with OpenCV, resizes it for speed, reshapes pixel data,
and runs scikit-learn K-means to find the top-N dominant colors along
with each color's percentage share of the image.

Also provides RGB histogram counts (scope-style) and image metadata.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .utils import closest_color_name, rgb_to_hex

RGB = Tuple[int, int, int]


def load_image_rgb(image_source: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """
    Decode an image to a full-resolution RGB uint8 array (no resize).

    Accepts a file path, raw bytes (e.g. from a Streamlit uploader), or a
    NumPy array. Arrays from OpenCV are assumed BGR and converted to RGB.
    """
    if isinstance(image_source, np.ndarray):
        image = image_source
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    if isinstance(image_source, (bytes, bytearray)):
        buffer = np.frombuffer(image_source, dtype=np.uint8)
        image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Could not decode image bytes. Use a valid JPG or PNG.")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_bgr = cv2.imread(str(image_source), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image at path: {image_source}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_max(image_rgb: np.ndarray, max_size: int = 400) -> np.ndarray:
    """Resize so the longest side is at most `max_size` pixels (keeps aspect ratio)."""
    height, width = image_rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_size:
        return image_rgb
    scale = max_size / float(longest)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_and_preprocess(image_source: Union[str, bytes, np.ndarray], max_size: int = 400) -> np.ndarray:
    """
    Load an image and resize so the longest side is at most `max_size` pixels.

    Returns an RGB uint8 array ready for clustering.
    """
    return resize_max(load_image_rgb(image_source), max_size=max_size)


def compute_rgb_histograms(image_rgb: np.ndarray) -> dict:
    """
    Compute per-channel histograms (256 bins, range 0–255).

    Returns a dict with keys r, g, b — each a length-256 int array of counts.
    Used for DaVinci-style RGB scope / histogram display.
    """
    channels = cv2.split(image_rgb)
    hist_r = cv2.calcHist([channels[0]], [0], None, [256], [0, 256]).flatten().astype(int)
    hist_g = cv2.calcHist([channels[1]], [0], None, [256], [0, 256]).flatten().astype(int)
    hist_b = cv2.calcHist([channels[2]], [0], None, [256], [0, 256]).flatten().astype(int)
    return {"r": hist_r, "g": hist_g, "b": hist_b}


def get_image_info(
    image_source: Union[str, bytes, np.ndarray],
    processed_rgb: np.ndarray,
    filename: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
) -> dict:
    """
    Collect human-readable metadata about the uploaded / analyzed image.

    Includes original & analysis dimensions, mean RGB, average luminance,
    format guess from filename, and file size when available.
    """
    full_rgb = load_image_rgb(image_source)
    orig_h, orig_w = full_rgb.shape[:2]
    proc_h, proc_w = processed_rgb.shape[:2]

    mean_r, mean_g, mean_b = [float(x) for x in full_rgb.reshape(-1, 3).mean(axis=0)]
    # Rec. 601 luminance
    luminance = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b

    fmt = "unknown"
    if filename:
        suffix = str(filename).rsplit(".", 1)
        if len(suffix) == 2:
            fmt = suffix[1].upper()

    size_kb = None
    if file_size_bytes is not None:
        size_kb = round(file_size_bytes / 1024.0, 2)
    elif isinstance(image_source, (bytes, bytearray)):
        size_kb = round(len(image_source) / 1024.0, 2)

    return {
        "filename": filename or "—",
        "format": fmt,
        "file_size_kb": size_kb,
        "original_width": int(orig_w),
        "original_height": int(orig_h),
        "analysis_width": int(proc_w),
        "analysis_height": int(proc_h),
        "channels": int(full_rgb.shape[2]) if full_rgb.ndim == 3 else 1,
        "pixel_count": int(orig_w * orig_h),
        "mean_r": round(mean_r, 1),
        "mean_g": round(mean_g, 1),
        "mean_b": round(mean_b, 1),
        "mean_luminance": round(float(luminance), 1),
    }


def extract_dominant_colors(
    image_source: Union[str, bytes, np.ndarray],
    n_colors: int = 5,
    max_size: int = 400,
    random_state: int = 42,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Extract the top `n_colors` dominant colors from an image using K-means.

    Steps:
      1. Load & resize the image (max 400px) for faster clustering.
      2. Reshape HxWx3 pixels into an (N, 3) float array.
      3. Fit KMeans with `n_colors` clusters on the pixel RGB values.
      4. Rank clusters by membership count and compute percentage shares.
      5. Attach nearest CSS3 color names and hex codes for display.

    Returns
    -------
    image_rgb : np.ndarray
        The preprocessed RGB image used for clustering.
    results : list of dict
        Each dict has keys: rgb, hex, name, percentage, count.
        Sorted by percentage descending.
    """
    image_rgb = load_and_preprocess(image_source, max_size=max_size)

    # Flatten to one pixel per row: shape (num_pixels, 3)
    pixels = image_rgb.reshape(-1, 3).astype(np.float64)

    # Clamp k so we never ask for more clusters than unique pixels
    unique_count = min(len(pixels), len(np.unique(pixels, axis=0)))
    k = max(1, min(int(n_colors), unique_count))

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    # Count how many pixels belong to each cluster
    counts = np.bincount(labels, minlength=k)
    total = counts.sum()

    # Sort clusters by dominance (largest share first)
    order = np.argsort(-counts)

    results: List[dict] = []
    for idx in order:
        rgb_float = centers[idx]
        rgb: RGB = (
            int(np.clip(round(rgb_float[0]), 0, 255)),
            int(np.clip(round(rgb_float[1]), 0, 255)),
            int(np.clip(round(rgb_float[2]), 0, 255)),
        )
        percentage = float(counts[idx]) / float(total) * 100.0
        results.append(
            {
                "rgb": rgb,
                "hex": rgb_to_hex(rgb),
                "name": closest_color_name(rgb),
                "percentage": round(percentage, 2),
                "count": int(counts[idx]),
            }
        )

    return image_rgb, results
