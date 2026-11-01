"""Shared constants and data loading for the cohort analysis layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Protocol order of all annotated sessions (Park et al., 2025).
ACTIVITY_ORDER = [
    "adaptation", "baseline", "tsst-prep", "tsst-speech", "meditation-1", "screen-reading",
    "ssst-prep", "ssst-sing", "meditation-2", "keyboard-typing", "rest-1", "mobile-typing",
    "rest-2", "standing", "rest-3", "walking", "rest-4", "jogging", "rest-5", "running",
]
ACTIVITY_LABELS = {
    "adaptation": "Adaptation", "baseline": "Baseline", "tsst-prep": "TSST prep",
    "tsst-speech": "TSST speech", "meditation-1": "Meditation 1", "screen-reading": "Reading",
    "ssst-prep": "SSST prep", "ssst-sing": "SSST sing", "meditation-2": "Meditation 2",
    "keyboard-typing": "Keyboard typing", "rest-1": "Rest 1", "mobile-typing": "Mobile typing",
    "rest-2": "Rest 2", "standing": "Standing", "rest-3": "Rest 3", "walking": "Walking",
    "rest-4": "Rest 4 (post-walk)", "jogging": "Jogging", "rest-5": "Rest 5 (post-jog)",
    "running": "Running",
}

# Reporting conditions (groups of sessions with comparable motion/physiology).
# "rest" = seated, hands still; "seated_task" = seated with speech/mental load
# (unconscious hand movement, speech-modulated breathing); "recovery" = seated
# immediately after exercise (non-stationary HR).
CONDITIONS = {
    "rest": ["baseline", "meditation-1", "screen-reading", "meditation-2", "rest-1", "rest-2", "rest-3"],
    "seated_task": ["tsst-prep", "tsst-speech", "ssst-prep", "ssst-sing"],
    "hand_activity": ["keyboard-typing", "mobile-typing"],
    "standing": ["standing"],
    "recovery": ["rest-4", "rest-5"],
    "locomotion": ["walking", "jogging", "running"],
    "adaptation": ["adaptation"],
}
CONDITION_ORDER = ["rest", "seated_task", "hand_activity", "standing", "recovery", "locomotion", "adaptation"]
CONDITION_LABELS = {
    "rest": "Seated rest", "seated_task": "Seated task", "hand_activity": "Hand activity",
    "standing": "Standing", "recovery": "Post-exercise", "locomotion": "Locomotion",
    "adaptation": "Adaptation",
}
ACTIVITY_CONDITION = {a: c for c, acts in CONDITIONS.items() for a in acts}

DEVICE_ORDER = ["PolarH10", "E4", "GalaxyWatch"]
PPG_DEVICES = ["E4", "GalaxyWatch"]
DEVICE_LABELS = {
    "PolarH10": "Polar H10 (ECG)",
    "E4": "Empatica E4",
    "GalaxyWatch": "Galaxy Watch 5",
}

COLORS = {
    "PolarH10": "#1f4e79",
    "E4": "#c0392b",
    "GalaxyWatch": "#2e8b57",
}

MOTION_COLORS = {"still": "#4c9f70", "moderate": "#e8a33d", "severe": "#c0392b"}

QC_MIN_SEGMENTS = 1


def load_combined(csv_path: str | Path) -> pd.DataFrame:
    """Load the cohort results table and normalise categorical ordering.

    Args:
        csv_path: Path to ``combined_results.csv``.

    Returns:
        DataFrame with ordered ``activity``/``device`` categorical columns and
        a ``qc_pass`` flag marking recordings that yielded a valid NN segment.
    """
    df = pd.read_csv(csv_path)
    df["activity"] = pd.Categorical(df["activity"], categories=ACTIVITY_ORDER, ordered=True)
    df["device"] = pd.Categorical(df["device"], categories=DEVICE_ORDER, ordered=True)
    df["condition"] = pd.Categorical(df["activity"].astype(str).map(ACTIVITY_CONDITION),
                                     categories=CONDITION_ORDER, ordered=True)
    df["qc_pass"] = df["n_segments"].fillna(0).astype(int) >= QC_MIN_SEGMENTS
    if "split" not in df.columns:
        df["split"] = "all"
    return df


def ppg_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return only wrist-PPG device rows."""
    return df[df["device"].isin(PPG_DEVICES)].copy()


def qc_passed(df: pd.DataFrame) -> pd.DataFrame:
    """Return only recordings that produced at least one valid NN segment."""
    return df[df["qc_pass"]].copy()
