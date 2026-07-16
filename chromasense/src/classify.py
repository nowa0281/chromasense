"""
classify.py — Supervised color category classification.

Creates a synthetic RGB → basic-color labeled dataset, trains a
Random Forest classifier, evaluates it on a held-out test split, and
exposes a predict() helper for classifying arbitrary RGB values.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Basic color categories used as class labels (~12 categories)
COLOR_CATEGORIES = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "cyan",
]

# Prototype RGB centers for each category (used to synthesize training samples)
_CATEGORY_CENTERS = {
    "red": (220, 30, 30),
    "orange": (240, 140, 30),
    "yellow": (240, 220, 40),
    "green": (40, 180, 60),
    "blue": (40, 80, 220),
    "purple": (140, 50, 180),
    "pink": (240, 120, 170),
    "brown": (140, 90, 50),
    "black": (20, 20, 20),
    "white": (240, 240, 240),
    "gray": (128, 128, 128),
    "cyan": (40, 200, 210),
}

# Default path for the synthetic CSV (relative to this package's parent)
_DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "color_training_data.csv",
)


def generate_synthetic_dataset(
    samples_per_class: int = 80,
    noise_std: float = 28.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a labeled synthetic RGB dataset around category prototypes.

    For each of the ~12 basic color categories, we sample Gaussian noise
    around a prototype RGB center, clip to [0, 255], and label the row.
    This produces a small but realistic training set without needing
    hand-labeled images.
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for label, center in _CATEGORY_CENTERS.items():
        center_arr = np.array(center, dtype=np.float64)
        noise = rng.normal(loc=0.0, scale=noise_std, size=(samples_per_class, 3))
        samples = np.clip(np.round(center_arr + noise), 0, 255).astype(int)
        for r, g, b in samples:
            rows.append({"r": int(r), "g": int(g), "b": int(b), "label": label})

    return pd.DataFrame(rows)


def save_dataset(df: pd.DataFrame, path: Optional[str] = None) -> str:
    """Write the synthetic dataset to CSV and return the path used."""
    path = path or _DEFAULT_DATA_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the training CSV, generating it first if the file is missing.

    Ensures `streamlit run app.py` works with zero manual data prep.
    """
    path = path or _DEFAULT_DATA_PATH
    if not os.path.isfile(path):
        df = generate_synthetic_dataset()
        save_dataset(df, path)
        return df
    return pd.read_csv(path)


def train_classifier(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, dict, str]:
    """
    Train a Random Forest on RGB features and evaluate on a held-out split.

    Returns
    -------
    model : RandomForestClassifier
        Fitted classifier ready for prediction.
    metrics : dict
        accuracy, precision, recall, f1 (weighted averages where applicable).
    report : str
        Full sklearn classification_report text for logging / display.
    """
    if df is None:
        df = load_dataset()

    # Use plain NumPy arrays so sklearn indexing works with pandas 2+/3+ backends
    X = np.asarray(df[["r", "g", "b"]], dtype=np.float64)
    y = np.asarray(df["label"], dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Random Forest is a strong, CPU-friendly baseline for tabular RGB data
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    report = classification_report(y_test, y_pred, zero_division=0)

    return model, metrics, report


def predict_color_category(model: RandomForestClassifier, rgb: Tuple[int, int, int]) -> str:
    """
    Classify a single RGB value into one of the basic color categories.

    Input is reshaped to a (1, 3) feature row expected by sklearn.
    """
    features = np.array([[int(rgb[0]), int(rgb[1]), int(rgb[2])]], dtype=np.float64)
    return str(model.predict(features)[0])


def predict_many(
    model: RandomForestClassifier, colors: List[Tuple[int, int, int]]
) -> List[str]:
    """Classify a list of RGB triples; returns one category label per color."""
    if not colors:
        return []
    features = np.array([[int(c[0]), int(c[1]), int(c[2])] for c in colors], dtype=np.float64)
    return [str(label) for label in model.predict(features)]


def ensure_dataset_and_print_metrics(path: Optional[str] = None) -> Tuple[RandomForestClassifier, dict, str]:
    """
    Convenience entry point: ensure CSV exists, train, print metrics, return model.

    Useful for a quick CLI sanity check of the classification pipeline.
    """
    df = load_dataset(path)
    model, metrics, report = train_classifier(df)
    print("=== Color Classification Metrics (held-out test split) ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print()
    print(report)
    return model, metrics, report


if __name__ == "__main__":
    # Generate CSV (if needed) and print evaluation metrics when run directly
    ensure_dataset_and_print_metrics()
