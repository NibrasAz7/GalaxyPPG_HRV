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

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Value written by the Galaxy Watch 5 PPG stream at optical-gain resets (2**21).
GALAXY_PPG_SENTINEL = 2097152


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


def convert_timestamps(values: pd.Series, device: str) -> pd.DatetimeIndex:
    """Vectorised :func:`convert_timestamp` for a whole column."""
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    if device == "E4":
        out = pd.to_datetime(np.round(numeric).astype("int64"), unit="us")
    else:
        out = pd.to_datetime(np.round(numeric).astype("int64"), unit="ms")
        if device == "PolarH10":
            out = out - pd.Timedelta(hours=9)
    return pd.DatetimeIndex(out)


def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a time-indexed frame and drop duplicate timestamps.

    Real device exports occasionally interleave rows or repeat timestamps,
    which breaks label-based slicing (``df[start:end]``) and interpolation
    helpers.  Sorting on load keeps every downstream stage monotonic.

    Args:
        df: DataFrame indexed by datetime.

    Returns:
        Chronologically sorted DataFrame without duplicated timestamps.
    """
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()


def load_device_data(user_folder: str, device: str, config: dict) -> dict:
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
    data: dict = {"device": device}

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
            signal_df["datetime"] = convert_timestamps(signal_df[ts_col], device)
            signal_df.set_index("datetime", inplace=True)
            if device == "GalaxyWatch" and "ppg" in signal_df.columns:
                # Samples flagged by the watch (status != 0, e.g. -999) and the
                # 2**21 sentinel written at every optical-gain reset are not
                # physiological; they are set to NaN and repaired downstream.
                bad = signal_df["ppg"] == GALAXY_PPG_SENTINEL
                if "status" in signal_df.columns:
                    bad |= signal_df["status"] != 0
                signal_df["ppg"] = signal_df["ppg"].astype(float).mask(bad)
            data["signal"] = _sort_by_time(signal_df)
            logger.debug("%s: signal loaded — %d samples", device, len(signal_df))

    # --- Heart rate ---
    hr_path = os.path.join(user_folder, device, "HR.csv")
    if os.path.exists(hr_path):
        hr_df = pd.read_csv(hr_path)
        ts_col = "phoneTimestamp" if device == "PolarH10" and "phoneTimestamp" in hr_df.columns else "timestamp"
        hr_col = "hr" if device != "E4" else "value"

        if ts_col in hr_df.columns:
            hr_df["datetime"] = convert_timestamps(hr_df[ts_col], device)
            hr_df.set_index("datetime", inplace=True)
            hr_df = _sort_by_time(hr_df)
            hr_series = hr_df[hr_col][(hr_df[hr_col] > 0) & (hr_df[hr_col] < 300)]
            data["hr"] = hr_series.to_frame("hr")
            data["hr_full"] = hr_df
            logger.debug("%s: HR loaded — %d valid readings", device, len(hr_series))

    # --- IBI ---
    if device == "GalaxyWatch":
        # Galaxy Watch stores IBI arrays inside HR.csv
        if "hr_full" in data and not data["hr_full"].empty:
            ibi_column = data["hr_full"]["ibi"].dropna().astype(str)
            # Leading rows are typically empty ("[]") while the sensor settles,
            # so the whole column must be inspected.
            has_ibi = bool((ibi_column.str.len() > 2).any())
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
                ibi_df["datetime"] = convert_timestamps(ibi_df[ts_col], device)
                ibi_df.set_index("datetime", inplace=True)
                data["ibi"] = _sort_by_time(ibi_df)
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
                acc_df["datetime"] = convert_timestamps(acc_df[ts_col], device)
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
                data["acc"] = _sort_by_time(acc_df)
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
    if device == "GalaxyWatch":
        # A batch row is stamped at its first beat and can span many minutes,
        # so rows starting well before the window are expanded and the
        # individual beat times filtered afterwards.
        lookback = pd.Timedelta(minutes=20)
        segment = ibi_df.loc[(ibi_df.index >= start_time - lookback) & (ibi_df.index <= end_time)]
    else:
        segment = ibi_df.loc[(ibi_df.index >= start_time) & (ibi_df.index <= end_time)]

    if segment.empty:
        logger.debug("%s IBI: no data in time segment", device)
        return np.array([]), np.array([])

    ibi_values: list[float] = []
    ibi_times: list = []

    if device == "E4":
        if "duration" in segment.columns:
            durations_us = segment["duration"].astype(float).values
            mask = (durations_us >= ibi_range[0] * 1000.0) & (
                durations_us <= ibi_range[1] * 1000.0
            )
            ibi_values = (durations_us[mask] / 1000.0).tolist()

            if len(ibi_values) > 0:
                # E4 IBI.csv stamps the END of each interval (consecutive rows
                # differ by the duration of the later row), so the timestamp
                # itself is the beat time.
                ibi_times = segment.index[mask]

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

                # IBIs are delivered in batches: the row timestamp marks the
                # beat that opens the first interval and the list runs forward
                # (dataReceived - timestamp ~ sum of the batch).  Interval i
                # therefore ends at timestamp + cumsum(ibis)[i].
                ends = timestamp + pd.to_timedelta(np.cumsum(np.asarray(parsed_ibis, dtype=float)), unit="ms")
                for i_val, v in enumerate(parsed_ibis):
                    # Keep only status == 0 (valid) when statuses are available
                    if parsed_statuses is not None and i_val < len(parsed_statuses):
                        if parsed_statuses[i_val] != 0:
                            continue
                    if ibi_range[0] <= v <= ibi_range[1] and start_time <= ends[i_val] <= end_time:
                        ibi_values.append(v)
                        ibi_times.append(ends[i_val])

    elif device == "PolarH10":
        if "duration" in segment.columns:
            durations = segment["duration"].astype(float).values
            mask = (durations >= ibi_range[0]) & (durations <= ibi_range[1])
            ibi_values = durations[mask].tolist()
            # Polar IBI.csv timestamps are synthetic: consecutive rows
            # accumulate the durations and are periodically re-anchored to the
            # BLE arrival time (jumps of hundreds of ms, sometimes backwards).
            # They are only approximate beat times; the durations are exact and
            # are compared with the ECG by value alignment.
            ibi_times = segment.index[mask]

    final_ibis = np.array(ibi_values, dtype=float)
    final_times = np.array(ibi_times)

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
