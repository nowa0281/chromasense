# ChromaSense

Lightweight **color detection and analysis** tool. Upload an image, extract its dominant colors with K-means clustering, map them to human-readable names, and classify each into a basic color category with a Random Forest model.

## Features

1. **Dominant color extraction** — OpenCV loads/resizes the image (max 400px), pixels are clustered with scikit-learn K-means, and each cluster’s share of the image is reported as a percentage.
2. **Color naming** — Every RGB center is matched to the nearest CSS3 color name via Euclidean distance in RGB space (`webcolors`).
3. **Color classification** — A Random Forest trained on a small synthetic RGB dataset predicts one of ~12 basic categories (red, blue, green, yellow, orange, purple, pink, brown, black, white, gray, cyan). Accuracy / precision / recall / F1 are shown for a held-out test split.
4. **Visualization** — Matplotlib shows the original image, a horizontal percentage bar chart, a palette swatch strip, and a DaVinci-style RGB histogram.
5. **Image information** — Resolution, file size, format, pixel count, mean RGB, and average luminance.
6. **Recent reports** — Last analyses are saved under `data/report_history.json` and listed in the sidebar.
7. **Streamlit UI** — Sidebar uploader + color-count slider; main panel shows results.

## Project structure

```
chromasense/
  data/                 # Synthetic RGB training CSV (auto-generated on first run)
  src/
    extract.py          # K-means dominant color extraction
    classify.py         # Random Forest train + predict
    utils.py            # RGB → CSS3 name nearest-neighbor helper
  samples/              # Sample images for local testing
  app.py                # Streamlit entry point
  requirements.txt
  README.md
```

## Setup

```bash
cd chromasense
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501), upload a JPG/PNG, and adjust the dominant-color slider (3–8).

## Methodology (short)

| Step | Approach |
|------|----------|
| Preprocess | Decode with OpenCV, convert BGR→RGB, resize longest side to ≤400px for speed |
| Dominant colors | Reshape pixels to `(N, 3)`, fit `KMeans(n_clusters=k)`, rank by cluster size, compute % share |
| Color names | Nearest CSS3 name by Euclidean distance in RGB space |
| Categories | Synthetic RGB samples around 12 prototype centers → `RandomForestClassifier` → predict category |
| Evaluation | Stratified train/test split; report accuracy, weighted precision/recall/F1 |

No deep learning — everything runs on CPU with a small dependency set.

## Dependencies

Only: `opencv-python`, `scikit-learn`, `matplotlib`, `streamlit`, `numpy`, `pandas`, `webcolors`.

## CLI check (optional)

Print classifier metrics without launching the UI:

```bash
python -m src.classify
```
