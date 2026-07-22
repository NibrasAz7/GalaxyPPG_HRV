#!/usr/bin/env python
"""Paper figure generation entry point for the GalaxyPPG HRV pipeline.

Generates all 10 publication-quality figures (EPS + PNG, 300 DPI) from
the combined results CSV produced by ``run_all_participants.py``.

The figure rendering logic lives in ``paper_figures_windowing.py`` (kept
in the original GalaxyPPG directory).  This script provides a clean CLI
wrapper with logging instead of print statements.

Usage::

    python generate_figures.py
    python generate_figures.py --csv paper_results/combined_results.csv
    python generate_figures.py --user /path/to/Dataset/P02 --activity baseline
    python generate_figures.py --output-dir fig/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="GalaxyPPG HRV Pipeline — paper figure generator (10 figures)"
    )
    parser.add_argument("--csv", metavar="CSV", default=None,
                        help="Path to combined results CSV")
    parser.add_argument("--user", metavar="FOLDER", default=None,
                        help="Participant data folder for Fig 1 (e.g. /path/to/Dataset/P02)")
    parser.add_argument("--activity", default="baseline",
                        help="Activity for Fig 1 (default: baseline)")
    parser.add_argument("--device", default="GalaxyWatch",
                        help="Device for Fig 1 (default: GalaxyWatch)")
    parser.add_argument("--output-dir", metavar="DIR", default=None,
                        help="Directory to save figures (default: fig/)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Import figure functions from the original script
    galaxyppg_dir = Path(__file__).resolve().parent.parent / "14635823" / "GalaxyPPG"
    sys.path.insert(0, str(galaxyppg_dir))

    import paper_figures_windowing as pfw

    # Set output directory
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "fig")
    os.makedirs(output_dir, exist_ok=True)
    pfw.OUT_DIR = os.path.abspath(output_dir)

    logger.info("Output directory: %s", pfw.OUT_DIR)

    # Fig 1: representative signal (requires raw data access)
    if args.user:
        logger.info("Generating Fig 1: representative signal")
        pfw.fig1_representative_signal(args.user, args.activity, args.device)
    else:
        logger.info("Skipping Fig 1 (--user not provided)")

    # Figs 2–10: from results CSV
    csv_path = args.csv
    if csv_path is None:
        results_dir = os.path.join(os.path.dirname(__file__), "paper_results")
        for candidate in [
            os.path.join(results_dir, "combined_results.csv"),
            os.path.join(results_dir, "combined_results_windowed.csv"),
        ]:
            if os.path.exists(candidate):
                csv_path = candidate
                break

    if csv_path and os.path.exists(csv_path):
        logger.info("Loading results from %s", csv_path)
        df = pd.read_csv(csv_path)
        df = pfw._clean_df(df)
        n_parts = df["participant"].nunique() if "participant" in df.columns else 1
        logger.info("  %d rows | %d participants | devices: %s",
                     len(df), n_parts, df["device"].unique().tolist())

        logger.info("Generating Figs 2–10")
        pfw.fig2_snr_heatmap(df)
        pfw.fig3_motion_stacked(df)
        pfw.fig4_hr_error_violin(df)
        pfw.fig5_bland_altman_grid(df)
        pfw.fig6_icc_bar(df)
        pfw.fig7_correlation_scatter(df)
        pfw.fig8_hrv_grouped_bar(df)
        pfw.fig9_participant_variability(df)
    else:
        logger.warning("No CSV found. Run run_all_participants.py first, then: python generate_figures.py")

    logger.info("All figures saved to %s/", output_dir)


if __name__ == "__main__":
    main()
