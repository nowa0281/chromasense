"""
utils.py — RGB-to-human-readable color name helpers.

Maps any RGB triple to the nearest CSS3 / web color name using
Euclidean distance in RGB space (nearest-neighbor matching).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import webcolors


RGB = Tuple[int, int, int]


def _css3_color_map() -> Dict[str, RGB]:
    """
    Build a name -> (R, G, B) dictionary of CSS3 / HTML color names.

    Prefer webcolors' CSS3 names; fall back to a small hard-coded set
    if the installed webcolors version exposes a different API.
    """
    # webcolors >= 1.13 uses names("css3"); older versions use CSS3_NAMES_TO_HEX
    try:
        names: Iterable[str] = webcolors.names("css3")
        mapping: Dict[str, RGB] = {}
        for name in names:
            rgb = webcolors.name_to_rgb(name)
            mapping[name] = (rgb.red, rgb.green, rgb.blue)
        return mapping
    except (AttributeError, ValueError, TypeError):
        # Fallback for older webcolors APIs
        mapping = {}
        for name, hex_value in webcolors.CSS3_NAMES_TO_HEX.items():
            rgb = webcolors.hex_to_rgb(hex_value)
            mapping[name] = (rgb.red, rgb.green, rgb.blue)
        return mapping


# Module-level cache so we only build the CSS3 map once
_CSS3_COLORS: Dict[str, RGB] = _css3_color_map()


def euclidean_distance(rgb1: RGB, rgb2: RGB) -> float:
    """
    Compute Euclidean distance between two RGB triples.

    Used as the similarity metric for nearest-neighbor color matching.
    """
    return math.sqrt(
        (rgb1[0] - rgb2[0]) ** 2
        + (rgb1[1] - rgb2[1]) ** 2
        + (rgb1[2] - rgb2[2]) ** 2
    )


def closest_color_name(rgb: RGB) -> str:
    """
    Return the CSS3 color name closest to the given RGB value.

    Iterates over all known CSS3 colors and picks the one with the
    smallest Euclidean distance in RGB space.
    """
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    target: RGB = (r, g, b)

    best_name = "unknown"
    best_distance = float("inf")

    for name, candidate in _CSS3_COLORS.items():
        distance = euclidean_distance(target, candidate)
        if distance < best_distance:
            best_distance = distance
            best_name = name

    return best_name


def rgb_to_hex(rgb: RGB) -> str:
    """Convert an RGB triple to a #RRGGBB hex string for display."""
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def format_color_label(rgb: RGB) -> str:
    """
    Build a human-friendly label: closest CSS name + hex code.

    Example: "steelblue (#4682b4)"
    """
    name = closest_color_name(rgb)
    return f"{name} ({rgb_to_hex(rgb)})"


def batch_closest_names(colors: List[RGB]) -> List[str]:
    """Map a list of RGB triples to their closest CSS3 color names."""
    return [closest_color_name(c) for c in colors]
