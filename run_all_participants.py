#!/usr/bin/env python
"""Batch runner for the GalaxyPPG HRV pipeline.

Processes all participants (P01–P24) through the pipeline and aggregates
results into a single ``combined_results.csv``.

Features:

- Skip already-processed participants (unless ``--force``)
- Process specific participants with repeated ``--p`` flags
- Auto-detect participants from the dataset directory

Usage::

    python run_all_participants.py                          # all participants
    python run_all_participants.py --p P02 --p P03          # specific participants
    python run_all_participants.py --force                  # re-run all
    python run_all_participants.py --data-root /path/to/Dataset
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

from src.utils import load_config
from run_pipeline import run_pipeline

logger = logging.getLogger(__name__)


def discover_participants(data_root: str) -> list[str]:
    """Auto-detect participant folders (P01, P02, ...) in the dataset root.

    Args:
        data_root: Path to the dataset directory.

    Returns:
        Sorted list of participant IDs (e.g. ``["P01", "P02", ...]``).
    """
    pattern = re.compile(r"^P\d{2}$")
    participants = [
        d.name for d in os.scandir(data_root)
        if d.is_dir() and pattern.match(d.name)
    ]
    return sorted(participants)


def main():
    parser = argparse.ArgumentParser(description="GalaxyPPG HRV Pipeline — batch runner")
    parser.add_argument("--p", action="append", dest="participants", metavar="ID",
                        help="Participant ID to process (repeatable). Default: all.")
    parser.add_argument("--data-root", default=None, help="Dataset root directory")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results CSV already exists")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config_path = args.config or os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

    data_root = args.data_root or os.environ.get("PPG_DATA_ROOT") or config["dataset"]["root"]
    if not data_root:
        logger.error("No data root specified. Use --data-root, PPG_DATA_ROOT env var, or config.yaml.")
        sys.exit(1)

    # Determine participants to process
    if args.participants:
        participants = args.participants
    else:
        participants = discover_participants(data_root)

    if not participants:
        logger.error("No participants found in %s", data_root)
        sys.exit(1)

    logger.info("Processing %d participants: %s", len(participants), ", ".join(participants))

    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []

    for pid in participants:
        csv_path = os.path.join(results_dir, f"{pid}_results.csv")

        if os.path.exists(csv_path) and not args.force:
            logger.info("Skipping %s (already processed, use --force to re-run)", pid)
            all_dfs.append(pd.read_csv(csv_path))
            continue

        user_folder = os.path.join(data_root, pid)
        if not os.path.exists(user_folder):
            logger.warning("Folder not found: %s — skipping", user_folder)
            continue

        logger.info("Processing %s ...", pid)
        results_df, _ = run_pipeline(user_folder, config)

        if results_df.empty:
            logger.warning("No results for %s", pid)
            continue

        results_df["participant"] = pid
        results_df.to_csv(csv_path, index=False)
        logger.info("%s: %d rows saved to %s", pid, len(results_df), csv_path)
        all_dfs.append(results_df)

    # Aggregate
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(results_dir, "combined_results.csv")
        combined.to_csv(combined_path, index=False)
        logger.info("Combined results: %d rows → %s", len(combined), combined_path)
    else:
        logger.warning("No results generated for any participant")


if __name__ == "__main__":
    main()
