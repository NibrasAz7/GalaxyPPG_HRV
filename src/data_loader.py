"""GalaxyPPG dataset loading and timestamp conversion.

Handles loading of PPG, ECG, accelerometer, HR, and IBI CSV files for the
three devices used in the GalaxyPPG dataset:

- **Empatica E4** — BVP at 64 Hz, ACC in 1/64 g units
- **Samsung Galaxy Watch 5** — PPG at 25 Hz, ACC in m/s²
- **Polar H10** — ECG at 130 Hz, ACC in mg (milli-g)

Each device stores data in a slightly different CSV layout with different
timestamp formats and timezones.  This module normalises everything to
pandas ``DatetimeIndex``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import timedelta
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def convert_timestamp(ts, device: str) -> pd.Timestamp:
    """Convert a raw device timestamp to a timezone-naive pandas Timestamp.

    Each device uses a different epoch and unit:

    - E4: microseconds since Unix epoch (UTC)
    - GalaxyWatch: milliseconds since Unix epoch (UTC)
    - PolarH10: milliseconds since Unix epoch, offset by +9 hours (KST)

    Args:
        ts: Raw timestamp value (int or float).
        device: One of ``"E4"``, ``"GalaxyWatch"``, ``"PolarH10"``.

    Returns:
        Timezone-naive ``pd.Timestamp``.
    """
    if device == "E4":
        return pd.to_datetime(ts / 1_000_000, unit="s")
    elif device == "GalaxyWatch":
        return pd.to_datetime(ts, unit="ms")
    elif device == "PolarH10":
        return pd.to_datetime(ts, unit="ms") - timedelta(hours=9)
    else:
        return pd.to_datetime(ts, unit="ms")


def load_device_data(user_folder: str, device: str, config: dict) -> Dict:
    """Load all available data for one device from a participant folder.

    Reads signal (PPG/ECG), HR, IBI, and accelerometer CSVs, converting
    timestamps and normalising accelerometer units to g-force.

    Args:
        user_folder: Path to the participant directory (e.g. ``Dataset/P02``).
        device: Device identifier (``"E4"``, ``"GalaxyWatch"``, ``"PolarH10"``).
        config: Parsed configuration dictionary (used for IBI range filtering).

    Returns:
        Dictionary with keys ``"signal"``, ``"hr"``, ``"hr_full"``, ``"ibi"``,
        ``"acc"`` (each a ``pd.DataFrame`` or empty if the file is missing).
    """
    data: Dict = {"device": device}
    ibi_range = config["processing"]["ibi_range"]

    # --- Signal (PPG / ECG) ---
    if device == "E4":
        signal_path = os.path.join(user_folder, device, "BVP.csv")
    elif device == "PolarH10":
        signal_path = os.path.join(user_folder, device, "ECG.csv")
    else:
        signal_path = os.path.join(user_folder, device, "PPG.csv")

    if os.path.exists(signal_path):
        signal_df = pd.read_csv(signal_path)
        ts_col = "phoneTimestamp" if device == "PolarH10" and "phoneTimestamp" in signal_df.columns else "timestamp"

        if ts_col not in signal_df.columns:
            logger.warning("%s: no timestamp column in signal file (columns: %s)", device, signal_df.columns.tolist())
        else:
            signal_df["datetime"] = signal_df[ts_col].apply(lambda x: convert_timestamp(x, device))
            signal_df.set_index("datetime", inplace=True)
            data["signal"] = signal_df
            logger.debug("%s: signal loaded — %d samples", device, len(signal_df))

    # --- Heart rate ---
    hr_path = os.path.join(user_folder, device, "HR.csv")
    if os.path.exists(hr_path):
        hr_df = pd.read_csv(hr_path)
        ts_col = "phoneTimestamp" if device == "PolarH10" and "phoneTimestamp" in hr_df.columns else "timestamp"
        hr_col = "hr" if device != "E4" else "value"

        if ts_col in hr_df.columns:
            hr_df["datetime"] = hr_df[ts_col].apply(lambda x: convert_timestamp(x, device))
            hr_df.set_index("datetime", inplace=True)
            hr_series = hr_df[hr_col][(hr_df[hr_col] > 0) & (hr_df[hr_col] < 300)]
            data["hr"] = hr_series.to_frame("hr")
            data["hr_full"] = hr_df
            logger.debug("%s: HR loaded — %d valid readings", device, len(hr_series))

    # --- IBI ---
    if device == "GalaxyWatch":
        # Galaxy Watch stores IBI arrays inside HR.csv
        if "hr_full" in data and not data["hr_full"].empty:
            ibi_sample = data["hr_full"]["ibi"].dropna().head(10)
            has_ibi = any(
                isinstance(v, str) and v != "[]" and len(v) > 2
                for v in ibi_sample
            )
            data["ibi"] = data["hr_full"] if has_ibi else pd.DataFrame()
        else:
            data["ibi"] = pd.DataFrame()
    else:
        # E4 and PolarH10 use a separate IBI.csv
        ibi_path = os.path.join(user_folder, device, "IBI.csv")
        if os.path.exists(ibi_path):
            ibi_df = pd.read_csv(ibi_path)
            ts_col = "phoneTimestamp" if device == "PolarH10" and "phoneTimestamp" in ibi_df.columns else "timestamp"
            if ts_col in ibi_df.columns:
                ibi_df["datetime"] = ibi_df[ts_col].apply(lambda x: convert_timestamp(x, device))
                ibi_df.set_index("datetime", inplace=True)
                data["ibi"] = ibi_df
            else:
                data["ibi"] = pd.DataFrame()
        else:
            data["ibi"] = pd.DataFrame()

    # --- Accelerometer ---
    acc_path = os.path.join(user_folder, device, "ACC.csv")
    if os.path.exists(acc_path):
        acc_df = pd.read_csv(acc_path)

        # Detect x/y/z columns (case-insensitive)
        x_col = y_col = z_col = None
        for col in acc_df.columns:
            cl = col.lower()
            if cl == "x":
                x_col = col
            elif cl == "y":
                y_col = col
            elif cl == "z":
                z_col = col

        if x_col and y_col and z_col:
            ts_col = "timestamp"
            if device == "PolarH10":
                ts_col = "phoneTimestamp" if "phoneTimestamp" in acc_df.columns else "sensorTimestamp"

            if ts_col in acc_df.columns:
                acc_df["datetime"] = acc_df[ts_col].apply(lambda x: convert_timestamp(x, device))
                acc_df.set_index("datetime", inplace=True)

                # Convert raw accelerometer values to g-force
                if device == "E4":
                    scale = 64.0
                elif device == "GalaxyWatch":
                    scale = 9.81
                else:  # PolarH10
                    scale = 1000.0

                acc_df["acc_x_g"] = acc_df[x_col] / scale
                acc_df["acc_y_g"] = acc_df[y_col] / scale
                acc_df["acc_z_g"] = acc_df[z_col] / scale
                data["acc"] = acc_df
                logger.debug("%s: ACC loaded — %d samples", device, len(acc_df))

    # --- Galaxy Watch quality gating ---
    if device == "GalaxyWatch" and "hr_full" in data:
        if "hrStatus" in data["hr_full"].columns:
            ok = data["hr_full"]["hrStatus"] == 1
            data["hr_full"] = data["hr_full"][ok]
            if "hr" in data:
                data["hr"] = data["hr"][data["hr"].index.isin(data["hr_full"].index)]

    return data


def extract_device_ibis(
    ibi_df: pd.DataFrame,
    device: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract inter-beat intervals from a device's IBI data within a time window.

    Handles three different IBI formats:

    - **E4**: ``duration`` column in microseconds
    - **GalaxyWatch**: ``ibi`` column containing JSON-like string arrays,
      with a parallel ``ibiStatus`` column for quality filtering
    - **PolarH10**: ``duration`` column in milliseconds

    Args:
        ibi_df: DataFrame indexed by datetime with IBI data.
        device: Device identifier.
        start_time: Window start (inclusive).
        end_time: Window end (inclusive).
        config: Parsed configuration dictionary.

    Returns:
        Tuple of ``(ibi_values_ms, ibi_times)`` as numpy arrays.
    """
    if ibi_df.empty:
        return np.array([]), np.array([])

    ibi_range = config["processing"]["ibi_range"]
    segment = ibi_df.loc[(ibi_df.index >= start_time) & (ibi_df.index <= end_time)]

    if segment.empty:
        logger.debug("%s IBI: no data in time segment", device)
        return np.array([]), np.array([])

    ibi_values: list[float] = []
    ibi_times: list = []

    if device == "E4":
        if "duration" in segment.columns:
            durations = segment["duration"].values
            valid = durations[(durations > 400_000) & (durations < 2_500_000)]
            ibi_values = (valid / 1000.0).tolist()

            if len(valid) > 0:
                dur_us = segment["duration"].astype(float).values
                mask = (dur_us > 400_000) & (dur_us < 2_500_000)
                ts_ns = segment.index.view("int64")[mask]
                beat_ns = ts_ns + (dur_us[mask] * 1000.0).astype(np.int64)
                ibi_times = pd.to_datetime(beat_ns)
            else:
                ibi_times = segment.index.values[: len(ibi_values)]

    elif device == "GalaxyWatch":
        if "ibi" in segment.columns:
            for timestamp, row in segment.iterrows():
                ibi_array = row["ibi"]
                ibi_status = row.get("ibiStatus")

                # Parse ibiStatus for quality filtering
                parsed_statuses = _parse_json_array(ibi_status)

                parsed_ibis = _parse_json_array(ibi_array, cast=float)
                if parsed_ibis is None:
                    continue

                for i_val, v in enumerate(parsed_ibis):
                    # Keep only status == 0 (valid) when statuses are available
                    if parsed_statuses is not None and i_val < len(parsed_statuses):
                        if parsed_statuses[i_val] != 0:
                            continue
                    if ibi_range[0] <= v <= ibi_range[1]:
                        ibi_values.append(v)
                        ibi_times.append(timestamp)

    elif device == "PolarH10":
        if "duration" in segment.columns:
            durations = segment["duration"].values
            valid = durations[(durations >= ibi_range[0]) & (durations <= ibi_range[1])]
            ibi_values = valid.tolist()
            ibi_times = segment.index.values[: len(ibi_values)]

    final_ibis = np.array(ibi_values, dtype=float)
    final_times = np.array(ibi_times)

    # Looser validation for Galaxy Watch to retain noisy data for metrics
    if device == "GalaxyWatch" and len(final_ibis) > 0:
        valid = (final_ibis > 250) & (final_ibis < 2500)
        final_ibis = final_ibis[valid]
        final_times = final_times[valid]

    logger.debug("%s IBI: %d intervals extracted", device, len(final_ibis))
    return final_ibis, final_times


def _parse_json_array(value, cast=int) -> list | None:
    """Parse a JSON-like string array from a Galaxy Watch CSV cell.

    Handles both proper JSON (``"[1, 2, 3]"``) and bare comma-separated
    values (``"1, 2, 3"``).  Returns ``None`` if the cell is empty or
    cannot be parsed.

    Args:
        value: Cell value from a pandas DataFrame (may be NaN, str, or list).
        cast: Type to cast each element to (``int`` or ``float``).

    Returns:
        List of parsed values, or ``None``.
    """
    if pd.isna(value) or not isinstance(value, str):
        return None

    s = value.strip().strip('"')
    if not s or s == "[]":
        return None

    # Try JSON first
    try:
        parsed = json.loads(s)
        if isinstance(parsed, (list, tuple)):
            return [cast(v) for v in parsed]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: manual split
    result = []
    for token in s.strip("[]").split(","):
        try:
            result.append(cast(token.strip()))
        except (ValueError, TypeError):
            continue
    return result if result else None
