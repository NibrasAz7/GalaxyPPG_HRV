"""Chance-level control for the vendor-IBI registration.

The vendor interval streams are registered to the ECG by value
(``src.validation.register_vendor_runs``).  To show that a registration is
not obtainable by chance, the *same* vendor intervals are randomly permuted
(destroying their temporal order) and registered again; the coverage obtained
from permuted sequences is the false-registration rate.

Usage::

    .venv/bin/python -m analysis.vendor_control --jobs 20
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from analysis.common import ACTIVITY_CONDITION
from run_all_participants import discover_participants
from run_pipeline import process_participant
from src.utils import load_config
from src.validation import register_vendor_runs


def _run(task: tuple[str, str, dict]) -> list[dict]:
    data_root, participant, config = task
    out = process_participant(os.path.join(data_root, participant), config)
    rng = np.random.default_rng(0)
    rows = []
    for (device, activity), (_, extras) in out.get("ppg", {}).items():
        reference = out["reference"].get(activity)
        vendor_s, vendor_ibis = extras.get("vendor_s"), extras.get("vendor_ibis")
        if reference is None or vendor_s is None or len(vendor_ibis) < 20:
            continue
        real = register_vendor_runs(reference["beats_s"], vendor_s, vendor_ibis)
        shuffled = [register_vendor_runs(reference["beats_s"], vendor_s, rng.permutation(vendor_ibis))
                    ["coverage_pct"] for _ in range(5)]
        rows.append({"participant": participant, "device": device, "activity": activity,
                     "condition": ACTIVITY_CONDITION.get(activity), "coverage_pct": real["coverage_pct"],
                     "control_coverage_pct": float(np.mean(shuffled)), "n_vendor": int(len(vendor_ibis))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor registration chance control")
    parser.add_argument("--data-root", default="Dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="paper_results/vendor_control.csv")
    parser.add_argument("--jobs", type=int, default=20)
    args = parser.parse_args()
    config = load_config(args.config)
    participants = [p for p in discover_participants(args.data_root)
                    if os.path.exists(os.path.join(args.data_root, p, "Event.csv"))]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        chunks = list(pool.map(_run, [(args.data_root, p, config) for p in participants]))
    frame = pd.DataFrame(r for chunk in chunks for r in chunk)
    frame.to_csv(args.output, index=False)
    print(frame.groupby(["device", "condition"])[["coverage_pct", "control_coverage_pct"]].mean().round(2))


if __name__ == "__main__":
    main()
