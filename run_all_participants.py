#!/usr/bin/env python
"""Batch runner for the GalaxyPPG HRV pipeline.

Processes all participants (P01–P24) through the pipeline and aggregates
results into a single ``combined_results.csv``.

Features:

- Skip already-processed participants (unless ``--force``)
- Process specific participants with repeated ``--p`` flags
- Auto-detect participants from the dataset directory
- Optional parallel processing across participants (``--jobs``)

Usage::

    python run_all_participants.py                          # all participants
    python run_all_participants.py --p P02 --p P03          # specific participants
    python run_all_participants.py --force                  # re-run all
    python run_all_participants.py --jobs 4                 # process four at a time
    python run_all_participants.py --data-root /path/to/Dataset
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TypeVar

import pandas as pd

from run_pipeline import run_pipeline
from src.utils import load_config

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is an optional convenience
    tqdm = None

logger = logging.getLogger(__name__)

T = TypeVar("T")


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


def _process_participant(task: tuple[str, str, dict]) -> tuple[str, pd.DataFrame]:
    """Process one participant (worker function for parallel execution).

    Args:
        task: Tuple of ``(participant_id, user_folder, config)``.

    Returns:
        Tuple of ``(participant_id, results_dataframe)``.
    """
    participant, user_folder, config = task
    results_df, _ = run_pipeline(user_folder, config)
    if not results_df.empty:
        results_df.insert(0, "split", participant_split(participant, config))
        results_df.insert(1, "evaluation", config.get("_evaluation_mode", "frozen"))
        results_df.insert(0, "participant", participant)
    return participant, results_df


def loso_config(config: dict, method_selection_csv: str, participant: str) -> dict:
    """Configuration whose detector priority excludes *participant*.

    The priority table is re-derived (same rule as ``run_method_selection``:
    rank by mean beat F1, ties by beat-IBI MAE) from the detector scores of
    every other participant, so the participant's own data never influences
    the detector chosen for it.
    """
    import copy

    from run_method_selection import derive_priority

    scores = pd.read_csv(method_selection_csv)
    others = scores[scores["participant"] != participant]
    summary = others.groupby(["device", "motion_level", "method"]).agg(
        f1=("f1", "mean"), beat_ibi_mae_ms=("beat_ibi_mae_ms", "median")).reset_index()
    fold = copy.deepcopy(config)
    fold.setdefault("detection", {})["priority"] = derive_priority(
        summary, config.get("detection", {}).get("priority", {}))
    fold["_evaluation_mode"] = "loso"
    return fold


def participant_split(participant: str, config: dict) -> str:
    """``"dev"`` / ``"test"`` membership from ``evaluation`` in the config."""
    evaluation = config.get("evaluation", {})
    if participant in evaluation.get("test_participants", []):
        return "test"
    if participant in evaluation.get("dev_participants", []):
        return "dev"
    return "unassigned"


def _save_participant_results(
    participant: str,
    results_df: pd.DataFrame,
    results_dir: str,
) -> None:
    """Write one participant's results to ``<results_dir>/<id>_results.csv``."""
    csv_path = os.path.join(results_dir, f"{participant}_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info("%s: %d rows saved to %s", participant, len(results_df), csv_path)


def _progress(iterable: Iterable[T], total: int, description: str) -> Iterator[T]:
    """Wrap an iterable with a progress bar (or periodic logging fallback).

    Args:
        iterable: Items to iterate over.
        total: Total number of items.
        description: Label shown next to the progress bar.

    Yields:
        The items of *iterable*, in order.
    """
    if tqdm is not None:
        yield from tqdm(iterable, total=total, desc=description, unit="participant")
        return

    for index, item in enumerate(iterable, start=1):
        yield item
        logger.info("%s: %d/%d complete (%.0f%%)", description, index, total, 100 * index / total)


def _run_tasks(
    tasks: list[tuple[str, str, dict]],
    jobs: int,
    results_dir: str,
) -> tuple[list[pd.DataFrame], list[str]]:
    """Execute participant tasks, tolerating individual failures.

    Args:
        tasks: ``(participant, user_folder, config)`` tuples to process.
        jobs: Number of parallel worker processes (1 = serial).
        results_dir: Directory for per-participant result CSVs.

    Returns:
        Tuple of ``(dataframes, failed_participants)``.
    """
    dataframes: list[pd.DataFrame] = []
    failed: list[str] = []

    def _handle(participant: str, results) -> None:
        if results is None or results.empty:
            logger.warning("No results for %s", participant)
            failed.append(participant)
            return
        _save_participant_results(participant, results, results_dir)
        dataframes.append(results)

    if jobs > 1 and len(tasks) > 1:
        logger.info("Running %d participants with %d workers", len(tasks), jobs)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_process_participant, task): task[0] for task in tasks}
            for future in _progress(as_completed(futures), total=len(futures), description="Participants"):
                participant = futures[future]
                try:
                    pid, results_df = future.result()
                except Exception as exc:
                    logger.error("%s: processing failed — %s", participant, exc, exc_info=True)
                    failed.append(participant)
                    continue
                _handle(pid, results_df)
    else:
        for task in _progress(tasks, total=len(tasks), description="Participants"):
            participant = task[0]
            try:
                pid, results_df = _process_participant(task)
            except Exception as exc:
                logger.error("%s: processing failed — %s", participant, exc, exc_info=True)
                failed.append(participant)
                continue
            _handle(pid, results_df)

    return dataframes, failed


def main():
    parser = argparse.ArgumentParser(description="GalaxyPPG HRV Pipeline — batch runner")
    parser.add_argument("--p", action="append", dest="participants", metavar="ID",
                        help="Participant ID to process (repeatable). Default: all.")
    parser.add_argument("--data-root", default=None, help="Dataset root directory")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results CSV already exists")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of participant processes to run in parallel (default: 1)")
    parser.add_argument("--loso", default=None, metavar="METHOD_SELECTION_CSV",
                        help="Leave-one-subject-out: each participant is processed with the "
                             "detector priority derived from the per-window detector scores "
                             "(run_method_selection.py --split all) of all OTHER participants")
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

    tasks: list[tuple[str, str, dict]] = []
    all_dfs: list[pd.DataFrame] = []
    reused = 0

    for participant in participants:
        csv_path = os.path.join(results_dir, f"{participant}_results.csv")
        if os.path.exists(csv_path) and not args.force:
            logger.info("Skipping %s (already processed, use --force to re-run)", participant)
            all_dfs.append(pd.read_csv(csv_path))
            reused += 1
            continue

        user_folder = os.path.join(data_root, participant)
        if not os.path.exists(user_folder):
            logger.warning("Folder not found: %s — skipping", user_folder)
            continue
        tasks.append((participant, user_folder,
                      loso_config(config, args.loso, participant) if args.loso else config))

    processed_dfs, failed = _run_tasks(tasks, args.jobs, results_dir)
    all_dfs.extend(processed_dfs)

    logger.info(
        "Summary: %d processed, %d reused, %d failed%s",
        len(processed_dfs), reused, len(failed),
        f" ({', '.join(failed)})" if failed else "",
    )

    # Aggregate
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        # Deterministic row order: workers finish in arbitrary order and reused
        # CSVs are appended first, so without an explicit sort a parallel or
        # partial re-run would not reproduce the byte layout of a serial run.
        sort_columns = [c for c in ("participant", "device", "activity") if c in combined.columns]
        if sort_columns:
            combined = combined.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        combined_path = os.path.join(results_dir, "combined_results.csv")
        combined.to_csv(combined_path, index=False)
        logger.info("Combined results: %d rows → %s", len(combined), combined_path)
    else:
        logger.warning("No results generated for any participant")


if __name__ == "__main__":
    main()
