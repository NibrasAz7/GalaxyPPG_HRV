#!/usr/bin/env python
"""Paper figure generation entry point.

Generates the manuscript figures (300 dpi PNG + vector PDF) from the combined
cohort results produced by ``run_all_participants.py``.

Usage::

    uv run python generate_figures.py
    uv run python generate_figures.py --csv paper_results/combined_results.csv
    uv run python generate_figures.py --user Dataset/P02 --activity baseline
    uv run python generate_figures.py --output-dir ../Assessment-of-PPG-Pipline-Paper/fig
"""

from __future__ import annotations

from analysis.figures import main

if __name__ == "__main__":
    main()
