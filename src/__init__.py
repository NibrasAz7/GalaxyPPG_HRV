"""GalaxyPPG HRV Pipeline — Source Package.

This package contains the modular implementation of the PPG/ECG heart rate
variability extraction pipeline described in the accompanying MDPI Sensors
paper.  Each module handles one stage of the processing chain:

- :mod:`data_loader`      — dataset I/O and timestamp conversion
- :mod:`signal_processing`— bandpass filtering and signal quality metrics
- :mod:`motion_detection` — accelerometer-based motion classification
- :mod:`peak_detection`   — multi-method systolic peak detection and ensemble
- :mod:`hrv_metrics`      — artifact correction, gap-aware segmentation, HRV indices
- :mod:`validation`       — ECG ground-truth comparison (peak matching, Bland-Altman)
- :mod:`utils`            — shared helpers (configuration loading)
"""

__version__ = "2.5.0"
