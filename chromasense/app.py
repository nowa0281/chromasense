"""
ChromaSense — Streamlit app for dominant color extraction & classification.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.ticker import FuncFormatter

# Ensure the project root is on sys.path so `src` imports work when launched
# via `streamlit run app.py` from the chromasense directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classify import (  # noqa: E402
    ensure_dataset_and_print_metrics,
    predict_color_category,
)
from src.extract import (  # noqa: E402
    compute_rgb_histograms,
    extract_dominant_colors,
    get_image_info,
    load_image_rgb,
    resize_max,
)

MAX_HISTORY = 12
_SESSION_HISTORY_KEY = "report_history"
MAX_UPLOAD_MB = 25
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


@st.cache_resource(show_spinner="Training color classifier…")
def get_trained_model():
    """
    Train (once per session) the Random Forest classifier.

    Uses Streamlit's cache_resource so we don't retrain on every rerun.
    Also ensures the synthetic CSV exists under data/.
    """
    model, metrics, report = ensure_dataset_and_print_metrics()
    return model, metrics, report


def go_home() -> None:
    """Return to the home/upload screen without leaving the app."""
    for key in ("img_bytes", "img_name", "_last_upload_key"):
        st.session_state.pop(key, None)
    st.session_state["home_upload_gen"] = st.session_state.get("home_upload_gen", 0) + 1
    st.session_state["side_upload_gen"] = st.session_state.get("side_upload_gen", 0) + 1


def store_upload(uploaded_file) -> bool:
    """
    Validate size, store bytes in session, and enter analysis view.

    Returns True when the upload was accepted.
    """
    if uploaded_file is None:
        return False

    data = uploaded_file.getvalue()
    upload_key = f"{uploaded_file.name}:{len(data)}"
    if st.session_state.get("_last_upload_key") == upload_key:
        return False

    size_mb = len(data) / (1024 * 1024)
    if len(data) > MAX_UPLOAD_BYTES:
        st.error(
            f"**{uploaded_file.name}** is {size_mb:.1f} MB. "
            f"Maximum upload size is **{MAX_UPLOAD_MB} MB**."
        )
        return False

    st.session_state["img_bytes"] = data
    st.session_state["img_name"] = uploaded_file.name
    st.session_state["_last_upload_key"] = upload_key
    return True


def _report_bar_label(result: dict) -> str:
    """Compact dominant-color label for the JPEG report (hex + truncated name)."""
    name = str(result["name"])
    if len(name) > 18:
        name = name[:17] + "…"
    return f"{result['hex']}\n{name}"


def _format_pixel_count(value: float, _pos) -> str:
    """Readable y-axis ticks for large histogram counts."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(int(value))


def build_full_report_jpeg(
    image_bytes: bytes,
    image_rgb: np.ndarray,
    results: list,
    info: dict,
    n_colors: int,
) -> bytes:
    """
    Compose a single JPEG containing the full analysis snapshot.

    Clean page layout:
      - Header title
      - Image (left) + image details (right)
      - Palette strip
      - Dominant-color bars + RGB histogram side by side
      - Classified colors as a neat table
    """
    display_rgb = resize_max(load_image_rgb(image_bytes), max_size=720)
    hists = compute_rgb_histograms(image_rgb)
    x = np.arange(256)
    n = len(results)

    fig_height = 14 + max(0, n - 6) * 0.4
    bar_row_height = max(2.8, 0.42 * n + 1.4)
    fig = plt.figure(figsize=(13, fig_height), facecolor="#f7f7f7")

    # Manual gridspec with padding so labels / titles don't collide
    outer = fig.add_gridspec(
        5,
        1,
        height_ratios=[0.45, 3.4, 0.95, bar_row_height, 2.5],
        hspace=0.32,
        left=0.16,
        right=0.97,
        top=0.95,
        bottom=0.05,
    )

    # --- Header ---
    ax_head = fig.add_subplot(outer[0])
    ax_head.set_xlim(0, 1)
    ax_head.set_ylim(0, 1)
    ax_head.axis("off")
    ax_head.text(
        0.0,
        0.62,
        "ChromaSense Report",
        fontsize=18,
        fontweight="bold",
        color="#1a1a1a",
        va="center",
    )
    ax_head.text(
        0.0,
        0.12,
        f"{info.get('filename', 'image')}   ·   "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   ·   "
        f"k = {n_colors}",
        fontsize=10,
        color="#555555",
        va="center",
    )

    # --- Image + details side by side ---
    mid = outer[1].subgridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18)
    ax_img = fig.add_subplot(mid[0, 0])
    ax_img.imshow(display_rgb)
    ax_img.set_title("Source image", fontsize=11, pad=8, loc="left", color="#222222")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_color("#cccccc")

    ax_info = fig.add_subplot(mid[0, 1])
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")
    ax_info.set_title("Image details", fontsize=11, pad=8, loc="left", color="#222222")

    detail_rows = [
        ("Filename", str(info.get("filename", "—"))),
        ("Format", str(info.get("format", "—"))),
        ("Resolution", f"{info['original_width']} × {info['original_height']}"),
        ("File size", f"{info.get('file_size_kb', '—')} KB"),
        ("Pixels", f"{info['pixel_count']:,}"),
        ("Channels", str(info.get("channels", 3))),
        ("Mean RGB", f"({info['mean_r']}, {info['mean_g']}, {info['mean_b']})"),
        ("Avg luminance", str(info["mean_luminance"])),
        ("Analyzed at", f"{info['analysis_width']} × {info['analysis_height']}"),
    ]
    row_h = 0.09
    start_y = 0.88
    for i, (label, value) in enumerate(detail_rows):
        y = start_y - i * row_h
        # Alternating row background for readability
        if i % 2 == 0:
            ax_info.add_patch(
                plt.Rectangle(
                    (0.0, y - 0.035),
                    1.0,
                    row_h,
                    facecolor="#eeeeee",
                    edgecolor="none",
                    transform=ax_info.transAxes,
                    clip_on=False,
                )
            )
        ax_info.text(0.04, y, label, fontsize=9, color="#666666", va="center", fontweight="bold")
        ax_info.text(0.42, y, value, fontsize=9, color="#1a1a1a", va="center")

    # --- Palette ---
    ax_pal = fig.add_subplot(outer[2])
    ax_pal.set_xlim(0, max(n, 1))
    ax_pal.set_ylim(0, 1.35)
    ax_pal.axis("off")
    ax_pal.set_title("Color palette", fontsize=11, pad=6, loc="left", color="#222222")
    for i, r in enumerate(results):
        ax_pal.add_patch(
            plt.Rectangle(
                (i + 0.04, 0.28),
                0.92,
                0.72,
                facecolor=tuple(c / 255.0 for c in r["rgb"]),
                edgecolor="#333333",
                linewidth=0.7,
            )
        )
        lum = 0.299 * r["rgb"][0] + 0.587 * r["rgb"][1] + 0.114 * r["rgb"][2]
        ax_pal.text(
            i + 0.5,
            0.64,
            r["hex"],
            ha="center",
            va="center",
            fontsize=8,
            color="white" if lum < 140 else "black",
            fontweight="bold",
        )
        ax_pal.text(
            i + 0.5,
            0.08,
            f"{r['percentage']:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
        )
        pal_name = str(r["name"])
        if len(pal_name) > 14:
            pal_name = pal_name[:13] + "…"
        ax_pal.text(
            i + 0.5,
            1.12,
            pal_name,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )

    # --- Bars + histogram ---
    bottom = outer[3].subgridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)

    ax_bar = fig.add_subplot(bottom[0, 0])
    labels = [_report_bar_label(r) for r in results]
    percentages = [r["percentage"] for r in results]
    bar_colors = [tuple(c / 255.0 for c in r["rgb"]) for r in results]
    y_pos = list(range(n))
    bars = ax_bar.barh(y_pos, percentages, color=bar_colors, edgecolor="#333333", linewidth=0.5, height=0.72)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=7.5)
    ax_bar.invert_yaxis()
    xmax = max(percentages) * 1.32 if percentages else 100
    ax_bar.set_xlim(0, xmax)
    ax_bar.set_xlabel("Share of image (%)", fontsize=9, color="#333333", labelpad=6)
    ax_bar.set_title("Dominant colors", fontsize=11, pad=8, loc="left", color="#222222")
    ax_bar.tick_params(axis="x", labelsize=8, colors="#333333")
    ax_bar.tick_params(axis="y", labelsize=7.5, colors="#222222", pad=2)
    for bar, pct, r in zip(bars, percentages, results):
        label_x = min(bar.get_width() + xmax * 0.02, xmax * 0.97)
        ax_bar.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}% · {r['category']}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#222222",
            clip_on=True,
        )

    ax_hist = fig.add_subplot(bottom[0, 1])
    ax_hist.set_facecolor("#121212")
    ax_hist.fill_between(x, hists["r"], color="#ff3344", alpha=0.45, linewidth=0)
    ax_hist.fill_between(x, hists["g"], color="#33dd66", alpha=0.45, linewidth=0)
    ax_hist.fill_between(x, hists["b"], color="#3399ff", alpha=0.45, linewidth=0)
    ax_hist.plot(x, hists["r"], color="#ff6677", linewidth=0.8, label="R")
    ax_hist.plot(x, hists["g"], color="#66ee88", linewidth=0.8, label="G")
    ax_hist.plot(x, hists["b"], color="#66bbff", linewidth=0.8, label="B")
    ax_hist.set_xlim(0, 255)
    ymax = max(max(hists["r"]), max(hists["g"]), max(hists["b"]), 1)
    ax_hist.set_ylim(0, ymax * 1.08)
    ax_hist.set_xlabel("Level (0–255)", fontsize=9, color="#eeeeee", labelpad=8)
    ax_hist.set_ylabel("Pixel count", fontsize=9, color="#eeeeee", labelpad=8)
    ax_hist.set_title("RGB histogram", fontsize=11, pad=8, loc="left", color="#eeeeee")
    ax_hist.tick_params(axis="both", labelsize=7.5, colors="#dddddd", pad=3)
    ax_hist.yaxis.set_major_formatter(FuncFormatter(_format_pixel_count))
    for spine in ax_hist.spines.values():
        spine.set_color("#555555")
    legend = ax_hist.legend(loc="upper right", fontsize=8, framealpha=0.45, labelcolor="#eeeeee")
    legend.get_frame().set_facecolor("#222222")
    legend.get_frame().set_edgecolor("#555555")

    # --- Classified colors table ---
    ax_tbl = fig.add_subplot(outer[4])
    ax_tbl.axis("off")
    ax_tbl.set_title("Classified colors", fontsize=11, pad=8, loc="left", color="#222222")

    col_labels = ["#", "CSS name", "RF category", "RGB", "Hex", "Share %"]
    cell_text = []
    cell_colors = []
    for i, r in enumerate(results, start=1):
        cell_text.append(
            [
                str(i),
                r["name"],
                r["category"],
                f"({r['rgb'][0]}, {r['rgb'][1]}, {r['rgb'][2]})",
                r["hex"],
                f"{r['percentage']:.1f}",
            ]
        )
        # Light tint of the actual color on the hex cell column via row bg
        tint = tuple(min(1.0, c / 255.0 * 0.25 + 0.75) for c in r["rgb"])
        cell_colors.append(["#ffffff", "#ffffff", "#ffffff", "#ffffff", tint, "#ffffff"])

    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#e8e8e8"] * len(col_labels),
        loc="upper center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(fontweight="bold", color="#333333")
        # Give CSS name / category a bit more width feel via left align
        if col in (1, 2) and row > 0:
            cell._loc = "left"
            cell.PAD = 0.04

    fig.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.05, hspace=0.34)

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="jpeg",
        dpi=150,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.25,
        pil_kwargs={"quality": 93},
    )
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_bar_chart(results: list) -> plt.Figure:
    """
    Build a horizontal bar chart of dominant colors with percentage labels.

    Bar faces are colored with the actual RGB of each dominant color so the
    chart doubles as a visual palette summary.
    """
    labels = [f"{r['name']}\n{r['hex']}" for r in results]
    percentages = [r["percentage"] for r in results]
    bar_colors = [tuple(c / 255.0 for c in r["rgb"]) for r in results]

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.55 * len(results) + 1)))
    y_pos = range(len(results))
    bars = ax.barh(list(y_pos), percentages, color=bar_colors, edgecolor="#333333", linewidth=0.6)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # Highest share at the top
    ax.set_xlabel("Share of image (%)")
    ax.set_xlim(0, max(percentages) * 1.25 if percentages else 100)
    ax.set_title("Dominant Color Distribution")

    # Annotate each bar with its percentage
    for bar, pct in zip(bars, percentages):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def build_palette_swatch(results: list) -> plt.Figure:
    """
    Build a horizontal color palette swatch strip.

    Each cell shows the dominant color; hex codes are overlaid for clarity.
    """
    n = len(results)
    fig, ax = plt.subplots(figsize=(8, 1.6))
    for i, r in enumerate(results):
        ax.add_patch(
            plt.Rectangle(
                (i, 0),
                1,
                1,
                facecolor=tuple(c / 255.0 for c in r["rgb"]),
                edgecolor="#222222",
                linewidth=0.8,
            )
        )
        # Choose light or dark text depending on luminance
        luminance = 0.299 * r["rgb"][0] + 0.587 * r["rgb"][1] + 0.114 * r["rgb"][2]
        text_color = "white" if luminance < 140 else "black"
        ax.text(
            i + 0.5,
            0.5,
            r["hex"],
            ha="center",
            va="center",
            fontsize=8,
            color=text_color,
            fontweight="bold",
        )

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Color Palette", pad=8)
    fig.tight_layout()
    return fig


def build_rgb_histogram_figure(image_rgb: np.ndarray) -> plt.Figure:
    """
    Build a DaVinci-style RGB color histogram on a dark background.

    Overlays R, G, and B channel distributions (levels 0–255).
    """
    hists = compute_rgb_histograms(image_rgb)
    x = np.arange(256)

    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor="#1a1a1a")
    ax.set_facecolor("#0d0d0d")
    ax.fill_between(x, hists["r"], color="#ff3344", alpha=0.45, linewidth=0)
    ax.fill_between(x, hists["g"], color="#33dd66", alpha=0.45, linewidth=0)
    ax.fill_between(x, hists["b"], color="#3399ff", alpha=0.45, linewidth=0)
    ax.plot(x, hists["r"], color="#ff6677", linewidth=0.8, label="Red")
    ax.plot(x, hists["g"], color="#66ee88", linewidth=0.8, label="Green")
    ax.plot(x, hists["b"], color="#66bbff", linewidth=0.8, label="Blue")
    ax.set_xlim(0, 255)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Level (0–255)", color="#cccccc")
    ax.set_ylabel("Pixel count", color="#cccccc")
    ax.set_title("RGB Histogram (scope)", color="#eeeeee", fontsize=11)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    legend = ax.legend(loc="upper right", framealpha=0.3, labelcolor="#dddddd")
    legend.get_frame().set_facecolor("#222222")
    legend.get_frame().set_edgecolor("#555555")

    fig.tight_layout()
    return fig


def _normalize_history_entry(item: dict) -> dict:
    """Normalize a history entry so the UI never crashes on partial data."""
    top = item.get("top_colors") or []
    if not isinstance(top, list):
        top = []
    return {
        "timestamp": str(item.get("timestamp") or "—"),
        "filename": str(item.get("filename") or "untitled"),
        "n_colors": item.get("n_colors") or len(top) or "—",
        "dimensions": str(item.get("dimensions") or "—"),
        "mean_rgb": str(item.get("mean_rgb") or "—"),
        "mean_luminance": item.get("mean_luminance", "—"),
        "format": str(item.get("format") or "—"),
        "file_size_kb": item.get("file_size_kb"),
        "top_colors": [
            {
                "name": str(c.get("name") or "—"),
                "category": str(c.get("category") or "—"),
                "hex": str(c.get("hex") or "#888888"),
                "percentage": float(c.get("percentage") or 0),
            }
            for c in top
            if isinstance(c, dict)
        ],
    }


def init_report_history() -> None:
    """Ensure per-browser session history exists (not shared across users)."""
    if _SESSION_HISTORY_KEY not in st.session_state:
        st.session_state[_SESSION_HISTORY_KEY] = []


def get_report_history() -> list:
    """Return this browser session's recent reports (cleared on hard refresh)."""
    init_report_history()
    return st.session_state[_SESSION_HISTORY_KEY]


def clear_report_history() -> None:
    """Clear recent reports for this browser session only."""
    st.session_state[_SESSION_HISTORY_KEY] = []


def append_history_entry(entry: dict) -> list:
    """
    Prepend a new report to this session's history (newest first).

    Stored in st.session_state only — never written to disk — so each visitor
    sees only their own reports and a hard refresh starts with empty history.

    Deduplicates consecutive identical filename+n_colors+top hex to avoid
    flooding history on Streamlit reruns of the same analysis.
    """
    history = get_report_history()
    normalized = _normalize_history_entry(entry)
    if history:
        prev = history[0]
        same = (
            prev.get("filename") == normalized.get("filename")
            and prev.get("n_colors") == normalized.get("n_colors")
            and prev.get("top_colors") == normalized.get("top_colors")
        )
        if same:
            return history

    history.insert(0, normalized)
    history = history[:MAX_HISTORY]
    st.session_state[_SESSION_HISTORY_KEY] = history
    return history


def _swatch_html(colors: list, max_n: int = 5) -> str:
    """Build inline HTML color chips for recent-report cards."""
    chips = []
    for c in colors[:max_n]:
        hex_code = c.get("hex") or "#888888"
        title = f"{c.get('name', '')} · {c.get('category', '')} · {c.get('percentage', 0)}%"
        chips.append(
            f'<span title="{title}" style="display:inline-block;width:28px;height:28px;'
            f'border-radius:6px;background:{hex_code};border:1px solid #333;'
            f'margin-right:6px;vertical-align:middle;"></span>'
        )
    return "".join(chips) if chips else "<span style='color:#888'>—</span>"


def render_history_cards(history: list, *, compact: bool = False, clear_key: str = "clear_history") -> None:
    """
    Render recent reports as readable cards (works better than a wide dataframe on phones).

    Each card shows timestamp, filename, dimensions, mean RGB, luminance, and color chips.
    """
    if not history:
        st.caption("No analyses yet — upload an image to start building history.")
        return

    for i, item in enumerate(history):
        top = item.get("top_colors") or []
        top_name = top[0]["name"] if top else "—"
        top_hex = top[0]["hex"] if top else "#888888"
        size_kb = item.get("file_size_kb")
        size_txt = f"{size_kb} KB" if size_kb is not None else "—"

        header = f"**{item.get('filename', 'untitled')}**"
        if compact:
            st.markdown(f"{header}")
            st.caption(f"{item.get('timestamp', '—')} · {item.get('dimensions', '—')}")
            st.markdown(_swatch_html(top, 5), unsafe_allow_html=True)
            st.caption(f"Top: {top_name} `{top_hex}` · k={item.get('n_colors', '—')}")
            st.markdown("---")
        else:
            with st.container(border=True):
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.markdown(header)
                    st.caption(item.get("timestamp", "—"))
                    st.markdown(_swatch_html(top, 5), unsafe_allow_html=True)
                with c2:
                    st.markdown(
                        f"**Size:** {item.get('dimensions', '—')}  \n"
                        f"**Format:** {item.get('format', '—')} · {size_txt}  \n"
                        f"**Colors (k):** {item.get('n_colors', '—')}  \n"
                        f"**Mean RGB:** {item.get('mean_rgb', '—')}  \n"
                        f"**Luminance:** {item.get('mean_luminance', '—')}"
                    )

                if top:
                    lines = [
                        f"`{c['hex']}` **{c['name']}** → {c['category']} ({c['percentage']:.1f}%)"
                        for c in top
                    ]
                    st.markdown(" · ".join(lines) if len(lines) <= 3 else "  \n".join(lines))

    if st.button("Clear history", key=clear_key, use_container_width=True):
        clear_report_history()
        st.rerun()


def inject_app_theme() -> None:
    """Apply a full dark UI theme — no white surfaces."""
    st.markdown(
        """
        <style>
        [data-testid="stAppDeployButton"],
        .stDeployButton { display: none !important; }

        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        .main .block-container,
        section.main {
            background-color: #12171f !important;
            color: #e6edf5 !important;
        }
        [data-testid="stHeader"] {
            background: #12171f !important;
        }
        [data-testid="stToolbar"] {
            background: #12171f !important;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child,
        section[data-testid="stSidebar"] {
            background-color: #0e131a !important;
            border-right: 1px solid #2a3444 !important;
        }
        .block-container {
            padding-top: 1.25rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 100% !important;
            width: 100% !important;
        }
        /* Let charts/images stretch with the viewport */
        [data-testid="stImage"] img,
        [data-testid="stPyplot"] img {
            max-width: 100% !important;
            width: 100% !important;
            height: auto !important;
        }

        /* Text */
        h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown, .stCaption,
        [data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"] {
            color: #e6edf5 !important;
        }
        .stCaption, [data-testid="stCaptionContainer"] {
            color: #9aa8bc !important;
        }

        /* Inputs / uploader / buttons — charcoal, never white */
        [data-testid="stFileUploaderDropzone"] {
            min-height: 148px !important;
            padding: 1.2rem 1rem !important;
            border-radius: 14px !important;
            border: 1.5px dashed #4a5a70 !important;
            background: #1c2430 !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #5ec4a8 !important;
            background: #232d3c !important;
        }
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small {
            color: #c5d0de !important;
            background: transparent !important;
        }
        [data-testid="stFileUploaderDropzone"] svg {
            fill: #8fa0b5 !important;
            color: #8fa0b5 !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #1c2430 !important;
            border-color: #2f3b4d !important;
            border-radius: 14px !important;
        }

        /* Expander / info / metrics */
        [data-testid="stExpander"],
        [data-testid="stExpander"] details,
        [data-testid="stAlert"] {
            background-color: #1c2430 !important;
            color: #e6edf5 !important;
            border-color: #2f3b4d !important;
        }
        [data-testid="stMetric"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] {
            color: #e6edf5 !important;
        }

        /* Dataframe / tables */
        [data-testid="stDataFrame"],
        .stDataFrame, .stTable {
            background-color: #1c2430 !important;
        }

        /* Slider labels */
        div[data-testid="stSlider"] label, div[data-testid="stSlider"] p {
            color: #c5d0de !important;
        }

        /* Buttons */
        .stButton > button, .stDownloadButton > button {
            background: #243044 !important;
            color: #e6edf5 !important;
            border: 1px solid #3a4a60 !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #2d3d55 !important;
            border-color: #5ec4a8 !important;
        }

        /* Hero on home */
        .cs-hero {
            background: linear-gradient(145deg, #182232 0%, #1f3148 55%, #254058 100%);
            border: 1px solid #334860;
            border-radius: 18px;
            padding: 1.25rem 1.3rem 1.1rem 1.3rem;
            margin: 0 0 1rem 0;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
        }
        .cs-hero h3 {
            margin: 0 0 0.4rem 0;
            color: #f0f5fb !important;
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .cs-hero p {
            margin: 0;
            color: #a9b8cc !important;
            font-size: 0.95rem;
            line-height: 1.45;
        }
        .cs-chip-row { margin-top: 0.8rem; display: flex; flex-wrap: wrap; gap: 0.4rem; }
        .cs-chip {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 650;
            color: #101820 !important;
        }
        .cs-chip.teal { background: #5ec4a8; }
        .cs-chip.amber { background: #e0b35a; }
        .cs-chip.coral { background: #e08b7e; }

        .cs-k-value {
            font-size: 1.65rem;
            font-weight: 750;
            color: #5ec4a8 !important;
            letter-spacing: -0.03em;
            margin: 0.2rem 0 0.15rem 0;
        }
        .cs-k-hint { color: #9aa8bc !important; font-size: 0.88rem; margin-bottom: 0.25rem; }

        @media (max-width: 1100px) {
            .block-container {
                padding-left: 1.25rem !important;
                padding-right: 1.25rem !important;
            }
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 0.85rem !important;
                padding-left: 0.85rem !important;
                padding-right: 0.85rem !important;
            }
            .cs-hero { padding: 1.1rem 1rem; border-radius: 16px; }
            .cs-hero h3 { font-size: 1.25rem; }
            [data-testid="stFileUploaderDropzone"] { min-height: 165px !important; }
            div[data-testid="stVerticalBlock"]:has(.cs-back-home-marker) {
                position: sticky;
                top: 0.35rem;
                z-index: 999;
                background: #12171f;
                padding-bottom: 0.35rem;
                margin-bottom: 0.25rem;
            }
            div[data-testid="stVerticalBlock"]:has(.cs-back-home-marker) .stButton > button {
                min-height: 2.75rem !important;
                font-weight: 650 !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_landing(history: list) -> tuple:
    """
    Optimized home screen: modern analyze card, large upload, color slider, recent reports.

    Returns (uploaded_file_or_None, n_colors).
    """
    st.markdown(
        """
        <div class="cs-hero">
          <h3>Analyze an image</h3>
          <p>Drop a JPG/PNG or pick from your gallery. ChromaSense extracts the palette, names each color, and classifies it.</p>
          <div class="cs-chip-row">
            <span class="cs-chip teal">K-means palette</span>
            <span class="cs-chip amber">CSS color names</span>
            <span class="cs-chip coral">RF categories</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Upload**")
    st.caption(f"JPG or PNG · max {MAX_UPLOAD_MB} MB")
    home_upload = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        key=f"home_upload_{st.session_state.get('home_upload_gen', 0)}",
        label_visibility="collapsed",
        help=f"JPG or PNG from camera roll or files (max {MAX_UPLOAD_MB} MB)",
    )

    default_k = int(st.session_state.get("n_colors", 8))
    default_k = max(3, min(16, default_k))

    st.markdown("**Dominant colors**")
    st.caption("Choose how many colors to extract (3–16). Default is 8.")
    n_colors = st.slider(
        "Number of dominant colors",
        min_value=3,
        max_value=16,
        value=default_k,
        step=1,
        key="home_n_colors_v2",
        label_visibility="collapsed",
    )
    st.session_state["n_colors"] = n_colors
    st.markdown(
        f'<div class="cs-k-value">{n_colors}</div>'
        f'<div class="cs-k-hint">Extracting {n_colors} dominant colors</div>',
        unsafe_allow_html=True,
    )

    with st.expander("How it works", expanded=False):
        st.markdown(
            "- **K-means** finds dominant colors in the photo\n"
            "- **CSS nearest name** labels each color\n"
            "- **Random Forest** maps RGB → basic category (red, blue, …)\n"
            "- Download a full **JPEG report** after analysis"
        )

    st.subheader(f"Recent reports ({len(history)})")
    st.caption(
        "Private to this browser tab only. Other visitors cannot see your history. "
        "Refreshing the page clears it."
    )
    if history:
        st.caption("Past analyses on this device — palette chips and image details.")
        render_history_cards(history, compact=False, clear_key="clear_history_home")
    else:
        st.info("Your last analyses will show up here with palette chips and image details.")

    return home_upload, n_colors


def main() -> None:
    """Streamlit entry point: mobile landing + sidebar controls + analysis panel."""
    st.set_page_config(
        page_title="ChromaSense",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",  # keep left sidebar always open
    )
    inject_app_theme()

    st.title("ChromaSense")
    st.caption(
        "Dominant color extraction (K-means) + color category classification (Random Forest)"
    )

    history = get_report_history()

    # --- Sidebar controls (always visible) ---
    with st.sidebar:
        st.header("Controls")
        st.caption(f"JPG/PNG · max {MAX_UPLOAD_MB} MB")
        side_upload = st.file_uploader(
            "Upload an image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            key=f"side_upload_{st.session_state.get('side_upload_gen', 0)}",
            help=f"Maximum file size: {MAX_UPLOAD_MB} MB",
        )
        default_k = int(st.session_state.get("n_colors", 8))
        default_k = max(3, min(16, default_k))
        side_n = st.slider(
            "Number of dominant colors",
            min_value=3,
            max_value=16,
            value=default_k,
            step=1,
            key="side_n_colors_v2",
        )
        st.session_state["n_colors"] = side_n
        st.caption(f"Current: **{side_n}** colors (max 16)")
        st.markdown("---")
        st.caption("Upload here or on the home screen. Sidebar stays open for quick controls.")
        st.markdown("#### Recent reports")
        st.caption("Only visible in this browser tab — cleared on refresh.")
        render_history_cards(history[:5], compact=True, clear_key="clear_history_side")

    # Prefer whichever uploader the user used last; persist bytes so the home
    # widget can disappear without losing the active image.
    if side_upload is not None:
        if store_upload(side_upload):
            st.rerun()

    # Train / load classifier (cached)
    model, metrics, report = get_trained_model()

    with st.expander("Classifier metrics (held-out test split)", expanded=False):
        cols = st.columns(2)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
        cols[1].metric("F1-score", f"{metrics['f1']:.3f}")
        cols2 = st.columns(2)
        cols2[0].metric("Precision", f"{metrics['precision']:.3f}")
        cols2[1].metric("Recall", f"{metrics['recall']:.3f}")
        st.text(report)

    has_image = "img_bytes" in st.session_state and st.session_state["img_bytes"]

    if not has_image:
        home_upload, n_colors = render_landing(history)
        if store_upload(home_upload):
            st.rerun()
        return

    # Active image from session (home or sidebar)
    image_bytes = st.session_state["img_bytes"]
    image_name = st.session_state.get("img_name", "upload.jpg")
    n_colors = int(st.session_state.get("n_colors", 8))
    n_colors = max(3, min(16, n_colors))

    # Return to home without leaving the app (use this instead of browser Back on mobile).
    st.markdown('<div class="cs-back-home-marker"></div>', unsafe_allow_html=True)
    clear_col, _ = st.columns([1, 3])
    with clear_col:
        if st.button("← Back to home", use_container_width=True):
            go_home()
            st.rerun()

    try:
        image_rgb, results = extract_dominant_colors(image_bytes, n_colors=n_colors)
        info = get_image_info(
            image_bytes,
            image_rgb,
            filename=image_name,
            file_size_bytes=len(image_bytes),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    # Attach Random Forest category for each dominant color
    for r in results:
        r["category"] = predict_color_category(model, r["rgb"])

    # Save to recent-report history (deduped against identical consecutive runs)
    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": image_name,
        "n_colors": n_colors,
        "dimensions": f"{info['original_width']} x {info['original_height']}",
        "mean_rgb": f"({info['mean_r']}, {info['mean_g']}, {info['mean_b']})",
        "mean_luminance": info["mean_luminance"],
        "format": info.get("format", "—"),
        "file_size_kb": info.get("file_size_kb"),
        "top_colors": [
            {
                "name": r["name"],
                "category": r["category"],
                "hex": r["hex"],
                "percentage": r["percentage"],
            }
            for r in results[:5]
        ],
    }
    append_history_entry(history_entry)

    # --- Top download: full analysis snapshot as JPEG ---
    stem = Path(image_name).stem or "chromasense"
    jpeg_bytes = build_full_report_jpeg(
        image_bytes, image_rgb, results, info, n_colors
    )
    top_l, top_r = st.columns([4, 2])
    with top_l:
        st.caption("Analysis ready — download a full JPEG snapshot of image + colour data.")
    with top_r:
        st.download_button(
            label="Download report",
            data=jpeg_bytes,
            file_name=f"{stem}_chromasense_report.jpg",
            mime="image/jpeg",
            help="Download whole image analysis as a JPEG report",
            use_container_width=True,
        )

    # --- Image information ---
    st.subheader("Image information")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Resolution", f"{info['original_width']}×{info['original_height']}")
    m2.metric("File size", f"{info['file_size_kb']} KB" if info["file_size_kb"] is not None else "—")
    m3.metric("Format", info["format"])
    m4.metric("Pixels", f"{info['pixel_count']:,}")
    m5.metric("Avg luminance", f"{info['mean_luminance']}")

    i1, i2, i3 = st.columns(3)
    i1.write(f"**Filename:** {info['filename']}")
    i2.write(
        f"**Mean RGB:** ({info['mean_r']}, {info['mean_g']}, {info['mean_b']})"
    )
    i3.write(
        f"**Analyzed at:** {info['analysis_width']}×{info['analysis_height']} "
        f"(max 400px for speed)"
    )

    # --- Main panel layout ---
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Uploaded image")
        # Show the original upload bytes (full resolution), not the resized analysis copy
        st.image(image_bytes, use_container_width=True)
        st.caption(
            f"Full resolution · analysis uses a ≤400px copy "
            f"({info['analysis_width']}×{info['analysis_height']})"
        )

    with right:
        st.subheader("Dominant colors")
        fig_bar = build_bar_chart(results)
        st.pyplot(fig_bar, clear_figure=True, use_container_width=True)
        plt.close(fig_bar)

        st.subheader("Color palette")
        fig_swatch = build_palette_swatch(results)
        st.pyplot(fig_swatch, clear_figure=True, use_container_width=True)
        plt.close(fig_swatch)

    # --- Colour histogram (DaVinci-style scopes) ---
    st.subheader("Colour histogram")
    st.caption("RGB scope — tonal distribution of red, green, and blue channels.")
    fig_hist = build_rgb_histogram_figure(image_rgb)
    st.pyplot(fig_hist, clear_figure=True, use_container_width=True)
    plt.close(fig_hist)

    st.subheader("Classified color names")
    # Compact table: CSS nearest name, RF category, RGB, hex, %
    table_rows = []
    for r in results:
        table_rows.append(
            {
                "CSS nearest name": r["name"],
                "RF category": r["category"],
                "RGB": f"({r['rgb'][0]}, {r['rgb'][1]}, {r['rgb'][2]})",
                "Hex": r["hex"],
                "Share %": r["percentage"],
            }
        )
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    # Optional download of a simple text summary
    summary = io.StringIO()
    summary.write("ChromaSense dominant color report\n")
    summary.write(f"Image: {image_name}\n")
    summary.write(f"Requested colors: {n_colors}\n")
    summary.write(
        f"Resolution: {info['original_width']}x{info['original_height']}\n"
    )
    summary.write(
        f"Mean RGB: ({info['mean_r']}, {info['mean_g']}, {info['mean_b']})\n"
    )
    summary.write(f"Avg luminance: {info['mean_luminance']}\n\n")
    for i, r in enumerate(results, start=1):
        summary.write(
            f"{i}. {r['name']} / {r['category']}  "
            f"RGB{r['rgb']}  {r['hex']}  {r['percentage']}%\n"
        )
    st.download_button(
        "Download color report (.txt)",
        data=summary.getvalue(),
        file_name="chromasense_report.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()
