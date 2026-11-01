# GalaxyPPG HRV Pipeline

Open, reproducible pipeline that extracts heart rate and heart-rate (pulse-rate)
variability from wrist PPG — Empatica E4 (BVP, 64 Hz) and Samsung Galaxy
Watch 5 (PPG, 25 Hz) — and validates it **beat by beat** against the Polar H10
ECG (130 Hz) on the public [GalaxyPPG dataset](https://doi.org/10.1038/s41597-025-05152-z)
(Park et al., 2025; 23 analysable participants - P01 has no event log - and
all 20 annotated protocol sessions: adaptation, baseline, TSST, SSST,
meditation, screen reading, keyboard/mobile typing, standing, walking,
jogging, running and the rests between them).  Results are reported by
condition: seated rest, seated task, hand activity, standing, post-exercise,
locomotion.

Design principles:

* **Standalone** — no stage depends on the vendor's own HR/IBI output or on
  protocol activity labels; processing adapts only to the accelerometer.
* **Beat-level validation on a common clock** — device clocks are synchronised
  from the accelerometers, the PPG–ECG delay is estimated per recording, and
  beats are matched one-to-one within ±100 ms, with an explicit chance level.
* **Unbiased evaluation** — design decisions were taken on the development
  participants (P02–P16) and are reported on untouched test participants
  (P17–P24); the primary cohort run additionally uses leave-one-subject-out
  detector priorities, so no participant's data influences the detector used
  for it.

## Installation

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt -r requirements-dev.txt
```

## Usage

```bash
# Everything (≈5 min on 20 cores): selection, LOSO cohort, stats, studies, baselines, figures
./run_all.sh

# Whole cohort (parallel, ~10 s on 20 cores)  -> paper_results/combined_results.csv
uv run python run_all_participants.py --data-root Dataset --jobs 20 --force

# Detector priority study (development split; prints the derived priority)
uv run python run_method_selection.py --jobs 16

# Cohort statistics -> paper_results/tables/*.csv  (split = test / dev / all)
uv run python run_stats.py

# Ablation / sensitivity study (every design choice as a config variant)
uv run python run_ablation.py --jobs 20

# HRV agreement vs analysis window length
uv run python run_window_analysis.py --jobs 20

# PPG fiducial-point study (development split)
uv run python run_fiducial_study.py

# External baselines (HeartPy, NeuroKit2-Elgendi, MSPTDfast, vendor IBI) on identical windows
uv run python run_baselines.py

# Manuscript figures -> fig/*.png (preview) + fig/*.eps (manuscript; needs Ghostscript)
uv run python generate_figures.py --rebuild-cache

# Manuscript numbers (LaTeX macros), tables and figure copies -> <paper>/generated, <paper>/fig
uv run python -m analysis.paper_assets --paper-dir ../Assessment-of-PPG-Pipline-Paper

# Unit tests (synthetic data, dataset not required)
uv run pytest
```

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Load | `src/data_loader.py` | Vectorised timestamp conversion; Galaxy Watch status/sentinel samples masked; vendor IBI streams reconstructed with each device's beat-time convention |
| Clock sync | `src/sync.py` | Session offset of each watch vs Polar from accelerometer activity envelopes (cross-correlation, ±120 s), refined around every activity window (±10 s) to follow drift and mid-session re-syncs |
| Time base | `src/timebase.py` | Piecewise-linear per-sample clock fitted to the recorded timestamps (true rates: Polar ≈130.3 Hz, GW ≈24.8 Hz); packet-loss gaps preserved |
| Repair | `src/signal_processing.py` | NaN bridging and removal of optical-gain level shifts; repaired samples marked unreliable |
| Motion | `src/motion_detection.py` | Accelerometer motion level and periodicity (step / arm-swing frequency) |
| Filter | `src/signal_processing.py` | Zero-phase Butterworth in SOS form, band chosen by motion state; E4 accelerometer-referenced ANC under motion |
| Polarity | `config.yaml` | Fixed per device (E4 BVP upright, Galaxy Watch raw inverted — Park et al., 2025); verified by pulse rise/fall asymmetry |
| Detect | `src/peak_detection.py` | NeuroKit2, SciPy, slope-sum, template; consensus ensemble (≥2 detectors, one fiducial); detector chosen by device × motion priority derived on the dev split; HR ceilings ≥200 bpm |
| Timing | `src/peak_detection.py` | Beats detected as pulse onsets (detectors on the inverted pulse) and re-localised on a wide-band (≤10/8 Hz) signal at the intersecting-tangents foot |
| Beat QC | `src/artifacts.py` | Template-matching SQI per beat; Lipponen–Tarvainen (Kubios) classification with time-conserving correction; missed/ectopic beats are first recovered from the signal (template-matched pulse) before interpolation; interpolated and recovered beats are excluded from successive-difference indices |
| Segment | `src/hrv_metrics.py` | Gap-aware segments; gates on correction burden (≤5 %), residual jitter, segment SQI and cadence lock (HR at k × accelerometer cadence under periodic motion) |
| HRV | `src/hrv_metrics.py` | Time domain, Poincaré, SampEn/ApEn; spectral HRV from a cubic-spline tachogram, fixed 64-s Welch window, ΣPSD·Δf integration, ratios from aggregated powers |
| Validate | `src/validation.py` | Beat cross-correlogram delay (anchored across windows), one-to-one matching, chance-level surrogate, beat-to-beat IBI agreement, ECG HRV over the **identical** accepted spans, vendor IBI baseline (time-constrained run registration) |

The ECG reference is itself validated against the Polar H10 firmware R-R
stream (median MAE < 1 ms in every activity; `tables/reference_validity.csv`).

## Statistics (`analysis/agreement.py`, `analysis/stats.py`)

ICC(A,1) with exact confidence intervals; Bland–Altman with
repeated-measures limits (Bland & Altman 2007) and log-ratio limits for
positive HRV indices; TOST equivalence against a-priori bounds
(HR ±5 bpm, ln-amplitude ±0.10, ln-power ±0.20); participant-clustered
bootstrap confidence intervals; Friedman tests across activities;
Benjamini–Hochberg within outcome families. Every table is reported for the
test, development and full cohorts.

## Outputs

```
paper_results/
  combined_results.csv          one row per participant x device x activity
  P??_results.csv
  tables/                       cohort_flow, hr_agreement, beat_detection, hrv_agreement,
                                beat_ibi_agreement, pipeline_vs_vendor, motion_effect,
                                device_comparison, reference_validity, clock_sync,
                                timing_jitter, method_selection_summary, fiducial_summary
  ablation/ + ablation_summary.csv
  window_analysis/
  fiducial_study.csv, method_selection.csv, method_selection_all.csv, baselines.csv
fig/                            fig1-fig11 (main), figS1-figS5 (supplement), PNG + EPS
```

## Repository layout

```
run_pipeline.py            single participant (orchestration + validation)
run_all_participants.py    cohort runner (dev/test split column)
run_method_selection.py    detector priority study (dev split)
run_ablation.py            design-choice ablation / sensitivity
run_window_analysis.py     window-length study
run_fiducial_study.py      fiducial-point study
run_baselines.py           external baselines on identical windows
analysis/vendor_control.py chance control of the vendor-interval registration
analysis/paper_assets.py   manuscript numbers, tables and figures
run_all.sh                 full reproducible chain
run_stats.py               statistics tables
generate_figures.py        figures
src/                       pipeline modules
analysis/                  statistics and figures
tests/                     unit and regression tests
learning/                  learned beat detector (separate study)
```
