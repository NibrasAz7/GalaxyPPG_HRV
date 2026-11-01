#!/usr/bin/env python
"""Cohort statistics entry point.

Builds every manuscript summary table from the combined cohort results.

Usage::

    uv run python run_stats.py
    uv run python run_stats.py --csv paper_results/combined_results.csv
    uv run python run_stats.py --output-dir paper_results/tables
"""

from __future__ import annotations

from analysis.stats import main

if __name__ == "__main__":
    main()
