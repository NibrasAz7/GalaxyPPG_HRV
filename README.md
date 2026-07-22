# GalaxyPPG HRV Pipeline

A modular Python pipeline for extracting heart rate variability (HRV) from
photoplethysmography (PPG) signals, validated against ECG ground truth.

Designed for use with the **GalaxyPPG** dataset
([DOI: 10.1038/s41597-025-05152-z](https://doi.org/10.1038/s41597-025-05152-z)),
which contains synchronized PPG and ECG recordings from 24 participants
across 5 activities (baseline, rest, typing, walking, jogging) using three
devices:

- **Empatica E4** — PPG at 64 Hz
- **Samsung Galaxy Watch 5** — PPG at 25 Hz
- **Polar H10** — ECG at 130 Hz (ground truth)

## Prerequisites

- Python 3.10+
- GalaxyPPG dataset downloaded and extracted

## Installation

```bash
cd code/
pip install -r requirements.txt
```

## Configuration

All pipeline parameters are in `config.yaml`. Key settings:

- `dataset.root` — path to the GalaxyPPG `Dataset/` folder (or use
  `--data-root` CLI flag / `PPG_DATA_ROOT` environment variable)
- `devices` — sampling rates, filter bands, motion thresholds per device
- `processing` — IBI range, PMR tolerance, peak validation bounds
- `activities` — which activity sessions to process (from Event.csv)

## Usage

### Process a single participant

```bash
python run_pipeline.py --participant P02 --data-root /path/to/GalaxyPPG/Dataset
```

Output: `paper_results/P02_results.csv`

### Process all participants

```bash
python run_all_participants.py --data-root /path/to/GalaxyPPG/Dataset
```

Output: `paper_results/combined_results.csv` (one row per device × activity)

Options:

- `--p P02 --p P03` — process specific participants
- `--force` — re-run even if results CSV already exists
- `--verbose` — enable debug logging

### Generate paper figures

```bash
python generate_figures.py --csv paper_results/combined_results.csv
```

Output: 10 figures (EPS + PNG, 300 DPI) in `fig/`

To include the representative signal figure (Fig 1):

```bash
python generate_figures.py --user /path/to/GalaxyPPG/Dataset/P02 --activity baseline
```

## Pipeline Architecture

The pipeline consists of 6 stages, each in its own module:

| Stage | Module | Description |
|-------|--------|-------------|
| 1. Data loading | `src/data_loader.py` | Load PPG/ECG/ACC/IBI CSVs, normalize timestamps |
| 2. Motion detection | `src/motion_detection.py` | Accelerometer-based motion classification (low/moderate/severe) |
| 3. Signal processing | `src/signal_processing.py` | Adaptive bandpass filter, auto-polarity, SNR estimation |
| 4. Peak detection | `src/peak_detection.py` | Multi-method (NeuroKit2, SciPy, SSF, template) + ensemble voting |
| 5. HRV metrics | `src/hrv_metrics.py` | IBI artifact correction (LOF, Hampel), rolling HR, SDNN/RMSSD |
| 6. Validation | `src/validation.py` | PMR, Bland-Altman analysis, HR error vs ECG |

Entry-point scripts wire these modules together:

- `run_pipeline.py` — single participant (ECG ground truth → PPG comparison)
- `run_all_participants.py` — batch runner with skip-if-done logic
- `generate_figures.py` — 10 publication-quality figures

## Design Principles

This codebase follows the UNIX philosophy:

- **Modularity** — each module does one thing; importable independently
- **Separation** — parameters in `config.yaml`, logic in `src/`, orchestration in entry scripts
- **Silence** — `logging` instead of `print()`; no output unless `--verbose`
- **Composition** — `from src.peak_detection import detect_peaks_neurokit2` works standalone

## Dataset Structure

The GalaxyPPG dataset should be organized as:

```
Dataset/
├── P01/
│   ├── Event.csv              # Activity session timestamps
│   ├── E4/
│   │   ├── BVP.csv            # PPG signal
│   │   ├── HR.csv             # Heart rate
│   │   ├── IBI.csv            # Inter-beat intervals
│   │   └── ACC.csv            # Accelerometer
│   ├── GalaxyWatch/
│   │   ├── PPG.csv
│   │   ├── HR.csv             # Contains IBI arrays
│   │   └── ACC.csv
│   └── PolarH10/
│       ├── ECG.csv
│       ├── HR.csv
│       ├── IBI.csv
│       └── ACC.csv
├── P02/
│   └── ...
└── P24/
    └── ...
```

## Citation

If you use this pipeline or the GalaxyPPG dataset, please cite:

> GalaxyPPG: A Multi-Device PPG Dataset for HRV Analysis. *Scientific Data*.
> DOI: 10.1038/s41597-025-05152-z
