#!/usr/bin/env python
"""QC-threshold sensitivity analysis.

The HRV quality gates (maximum interpolated fraction, residual jitter and
device-HR deviation) determine how much data the pipeline accepts.  This
script re-runs the cohort under strict / baseline / permissive gate settings
and tabulates the resulting yield and agreement, so the manuscript can show
that its conclusions are not an artefact of threshold choice.

Usage::

    uv run python run_sensitivity.py --data-root Dataset --jobs 5
    uv run python run_sensitivity.py --p P02 --p P03 --jobs 2      # quick subset
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import subprocess
import sys

import pandas as pd
import yaml

from analysis.common import ACTIVITY_ORDER, PPG_DEVICES

logger = logging.getLogger(__name__)

VARIANTS = {
    "strict": {"max_interpolated_fraction": 0.15, "max_jitter_percent": 10.0},
    "base": {"max_interpolated_fraction": 0.25, "max_jitter_percent": 15.0},
    "permissive": {"max_interpolated_fraction": 0.40, "max_jitter_percent": 25.0},
}


def write_variant_config(base_config: dict, overrides: dict, results_dir: str, path: str) -> str:
    """Write a configuration file with modified HRV quality gates.

    Args:
        base_config: Parsed base configuration.
        overrides: Keys to override inside the ``hrv`` section.
        results_dir: Output directory for this variant's results.
        path: Destination config path.

    Returns:
        The destination path.
    """
    config = copy.deepcopy(base_config)
    config["hrv"].update(overrides)
    config.setdefault("output", {})["results_dir"] = results_dir
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def run_variant(
    config_path: str,
    data_root: str,
    participants: list[str],
    jobs: int,
) -> None:
    """Run the cohort pipeline for one sensitivity variant."""
    command = [
        sys.executable, "run_all_participants.py",
        "--data-root", data_root,
        "--config", config_path,
        "--force",
        "--jobs", str(jobs),
    ]
    for participant in participants:
        command.extend(["--p", participant])

    logger.info("Running variant: %s", config_path)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        logger.warning("Variant %s exited with code %d", config_path, result.returncode)


def summarise_variants(
    variant_dirs: dict[str, str],
    output_path: str,
) -> pd.DataFrame:
    """Tabulate yield and agreement across sensitivity variants.

    Args:
        variant_dirs: Mapping of variant name to its results directory.
        output_path: Destination CSV for the comparison table.

    Returns:
        The comparison DataFrame.
    """
    rows = []
    for variant, results_dir in variant_dirs.items():
        combined = os.path.join(results_dir, "combined_results.csv")
        if not os.path.exists(combined):
            logger.warning("Missing results for variant %s: %s", variant, combined)
            continue
        df = pd.read_csv(combined)
        ppg = df[df["device"].isin(PPG_DEVICES)]

        for (device, activity), group in ppg.groupby(["device", "activity"]):
            passed = group[group["n_segments"].fillna(0) > 0]
            rows.append({
                "variant": variant,
                "device": device,
                "activity": activity,
                "yield_pct": len(passed) / len(group) * 100.0 if len(group) else float("nan"),
                "median_interpolated_pct": passed["pct_interpolated"].median()
                if len(passed) else float("nan"),
                "hr_mae_bpm": group["hr_error_vs_ecg"].mean(),
                "rmssd_ratio": (passed["rmssd"].mean() / passed["ecg_rmssd"].mean())
                if len(passed) and passed["ecg_rmssd"].mean() > 0 else float("nan"),
            })

    table = pd.DataFrame(rows)
    if not table.empty:
        table["activity"] = pd.Categorical(table["activity"], categories=ACTIVITY_ORDER, ordered=True)
        table = table.sort_values(["device", "activity", "variant"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    table.to_csv(output_path, index=False)
    logger.info("Wrote sensitivity comparison → %s", output_path)
    return table


def main():
    parser = argparse.ArgumentParser(description="GalaxyPPG HRV — QC threshold sensitivity")
    parser.add_argument("--data-root", default="Dataset", help="Dataset root directory")
    parser.add_argument("--config", default="config.yaml", help="Base configuration")
    parser.add_argument("--output-dir", default="paper_results/sensitivity",
                        help="Directory for variant configs, results and the comparison table")
    parser.add_argument("--p", action="append", dest="participants", metavar="ID",
                        help="Restrict to specific participants (repeatable)")
    parser.add_argument("--jobs", type=int, default=5, help="Parallel workers per variant")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    with open(args.config, encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    participants = args.participants or sorted(
        entry.name for entry in os.scandir(args.data_root)
        if entry.is_dir() and entry.name.startswith("P") and entry.name[1:].isdigit()
    )

    variant_dirs = {}
    for variant, overrides in VARIANTS.items():
        results_dir = os.path.join(args.output_dir, variant, "results")
        config_path = write_variant_config(
            base_config, overrides, results_dir,
            os.path.join(args.output_dir, variant, "config.yaml"),
        )
        run_variant(config_path, args.data_root, participants, args.jobs)
        variant_dirs[variant] = results_dir

    summarise_variants(variant_dirs, os.path.join(args.output_dir, "comparison.csv"))


if __name__ == "__main__":
    main()
