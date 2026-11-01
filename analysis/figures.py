"""Manuscript figures for the GalaxyPPG HRV validation study.

Main figures
    fig01_pipeline            Pipeline overview and evaluation design
    fig02_signal_chain        Beat detection at rest vs walking, aligned to ECG
    fig03_clock_sync          Accelerometer clock sync, offsets, beat delay
    fig04_matching_tolerance  Why +-500 ms peak matching is uninformative
    fig05_beat_detection      Beat F1 vs condition, chance level, vendor coverage
    fig06_cohort_flow         Windows -> valid reference -> accepted HRV
    fig07_hr_agreement        Window HR Bland-Altman (rest / other conditions)
    fig08_beat_ibi_agreement  Beat-to-beat IBI agreement at rest
    fig09_hrv_forest          HRV log-ratio agreement with equivalence bounds
    fig10_rmssd_mechanism     Beat-timing error structure behind RMSSD inflation
    fig11_baselines           Pipeline vs HeartPy / NeuroKit2 / MSPTDfast
    fig12_ablation            Design choices: development vs test effect

Appendix
    figA1_reference_validity  ECG R-peaks vs Polar firmware R-R
    figA2_fiducial_study      Fiducial point comparison
    figA3_method_selection    Detector F1 per device x motion (dev split)
    figA4_cadence_lock        Accelerometer periodicity and motion-locked HR
    figA5_participant_heatmap HR error per participant x session

Beat-level figures use a cache of per-window beat times built by running the
pipeline with ``return_extras`` (``paper_results/cache/beats.pkl``).
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import pickle
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.agreement import bland_altman
from analysis.common import (
    ACTIVITY_CONDITION,
    ACTIVITY_LABELS,
    ACTIVITY_ORDER,
    CONDITION_LABELS,
    CONDITIONS,
    DEVICE_LABELS,
    PPG_DEVICES,
    load_combined,
)
from analysis.timing_error import cache_timing_errors

logger = logging.getLogger(__name__)

# Okabe-Ito colour-blind safe palette
C = {"E4": "#D55E00", "GalaxyWatch": "#0072B2", "PolarH10": "#000000",
     "grey": "#7F7F7F", "green": "#009E73", "yellow": "#E69F00", "purple": "#CC79A7",
     "sky": "#56B4E9"}
COND_COLORS = {"rest": "#0072B2", "seated_task": "#56B4E9", "hand_activity": "#009E73",
               "standing": "#CC79A7", "recovery": "#F0E442", "locomotion": "#D55E00", "adaptation": "#999999"}
# Reported conditions in display order (adaptation is shown only in the flow).
COND = ["rest", "seated_task", "hand_activity", "standing", "recovery", "locomotion"]
COND_SHORT = {"rest": "Seated\nrest", "seated_task": "Seated\ntask", "hand_activity": "Hand\nactivity",
              "standing": "Standing", "recovery": "Post-\nexercise", "locomotion": "Loco-\nmotion",
              "adaptation": "Adapt-\nation"}


def _cond_ticks(ax, positions, keys, fontsize: float = 6.3) -> None:
    """Single-line, rotated condition labels (they do not collide in narrow panels)."""
    labels = [COND_SHORT[k].replace("Loco-\nmotion", "Locomotion").replace("-\n", "-").replace("\n", " ")
              for k in keys]
    ax.set_xticks(list(positions), labels, fontsize=fontsize, rotation=35, ha="right", rotation_mode="anchor")


ACT_COLORS = {"baseline": "#0072B2", "rest-1": "#56B4E9", "keyboard-typing": "#009E73",
              "walking": "#E69F00", "jogging": "#D55E00"}
REST = tuple(CONDITIONS["rest"])
MOTION = tuple(CONDITIONS["locomotion"])
SPLIT_MARKERS = {"dev": "o", "test": "D"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 7.0, "axes.titlesize": 7.5,
        "axes.labelsize": 7.0, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.3, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False,
        "font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def _content_bbox(pdf_path: str) -> tuple[float, float, float, float] | None:
    """Tight bounding box of the drawn content of a PDF (Ghostscript ``bbox`` device)."""
    gs = shutil.which("gs")
    if gs is None:
        return None
    out = subprocess.run([gs, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=bbox", pdf_path],
                         capture_output=True, text=True, check=False)
    match = re.search(r"%%HiResBoundingBox:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", out.stderr + out.stdout)
    return tuple(float(v) for v in match.groups()) if match else None  # type: ignore[return-value]


def _fix_eps_bbox(eps_path: str, bbox: tuple[float, float, float, float], margin: float = 1.0) -> bool:
    """Rewrite the EPS header bounding box when it disagrees with the real content.

    ``eps2write`` occasionally writes a wildly wrong box (observed: a 10900 x 10000 pt box
    for a 790 x 385 pt figure) although the drawing itself is placed correctly; LaTeX then
    scales the figure from the wrong box.

    Returns:
        ``True`` if the header was rewritten.
    """
    with open(eps_path, "rb") as handle:
        data = handle.read()
    head = data[:4096].decode("latin-1")
    match = re.search(r"%%BoundingBox:\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", head)
    if match is None:
        return False
    x0, y0, x1, y1 = (float(v) for v in match.groups())
    cx0, cy0, cx1, cy1 = bbox
    area, content_area = max((x1 - x0) * (y1 - y0), 1e-9), max((cx1 - cx0) * (cy1 - cy0), 1e-9)
    if 0.7 <= area / content_area <= 1.6 and x0 <= cx0 + 1 and y0 <= cy0 + 1:
        return False
    nx0, ny0, nx1, ny1 = max(cx0 - margin, 0.0), max(cy0 - margin, 0.0), cx1 + margin, cy1 + margin
    bbox = f"%%BoundingBox: {int(nx0)} {int(ny0)} {int(nx1 + 0.999)} {int(ny1 + 0.999)}"
    hires = f"%%HiResBoundingBox: {nx0:.2f} {ny0:.2f} {nx1:.2f} {ny1:.2f}"
    fixed = re.sub(r"%%BoundingBox:[^\n]*", lambda _m: bbox, head, count=1)
    fixed = re.sub(r"%%HiResBoundingBox:[^\n]*", lambda _m: hires, fixed, count=1)
    with open(eps_path, "wb") as handle:
        handle.write(fixed.encode("latin-1") + data[4096:])
    return True


def _pdf_to_eps(pdf_path: str, eps_path: str) -> bool:
    """Convert a PDF to EPS with Ghostscript (``eps2write``), preserving transparency.

    Matplotlib's own EPS backend ignores alpha, which would change every plot that uses
    semi-transparent markers or bands; rendering through PDF and Ghostscript flattens
    transparency correctly.  The EPS bounding box is verified against the real content.

    Returns:
        ``True`` on success.
    """
    gs = shutil.which("gs")
    if gs is None:
        return False
    result = subprocess.run(
        [gs, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=eps2write", "-dEPSCrop",
         f"-sOutputFile={eps_path}", pdf_path],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or os.path.getsize(eps_path) == 0:
        return False
    bbox = _content_bbox(pdf_path)
    if bbox is not None and _fix_eps_bbox(eps_path, bbox):
        logger.info("Repaired EPS bounding box of %s", os.path.basename(eps_path))
    return True


def save_figure(fig, output_dir: str, name: str) -> None:
    """Save *fig* as a 300-dpi PNG preview and as EPS (the manuscript format)."""
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{name}.png"), bbox_inches="tight")
    eps_path = os.path.join(output_dir, f"{name}.eps")
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, f"{name}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        if not _pdf_to_eps(pdf_path, eps_path):
            logger.warning("Ghostscript unavailable or failed; using matplotlib EPS (alpha is dropped) for %s", name)
            fig.savefig(eps_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", name)


def _panel(ax, letter: str) -> None:
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


# ---------------------------------------------------------------------------
# Beat cache
# ---------------------------------------------------------------------------

def _collect_participant(task):
    data_root, participant, config = task
    from run_all_participants import participant_split
    from run_pipeline import run_pipeline
    frame, reference, extras = run_pipeline(os.path.join(data_root, participant), config, return_extras=True)
    if frame.empty:
        return participant, None
    ppg = frame[frame["device"] != "PolarH10"].set_index(["device", "activity"])
    windows = {}
    for (device, activity), ext in extras.items():
        if activity not in reference or (device, activity) not in ppg.index:
            continue
        row = ppg.loc[(device, activity)]
        windows[(device, activity)] = {
            "detected_s": ext["detected_s"], "beats_s": ext["beats_s"],
            "segments": [(float(pd.Timestamp(s["start"]).value / 1e9), float(pd.Timestamp(s["end"]).value / 1e9))
                         for s in ext.get("segments", [])],
            "ecg_s": reference[activity]["beats_s"], "lag_s": float(row["lag_ms"]) / 1000.0,
            "hrv_ibi": ext.get("hrv_ibi_pairs"), "vendor_s": ext.get("vendor_s"),
            "motion_level": row["motion_level"],
        }
    return participant, {"split": participant_split(participant, config), "windows": windows}


def load_beat_cache(data_root: str, config_path: str, cache_path: str, jobs: int = 20,
                    rebuild: bool = False) -> dict:
    import hashlib
    import json

    from run_all_participants import discover_participants
    from src.utils import load_config
    config = load_config(config_path)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    if os.path.exists(cache_path) and not rebuild:
        with open(cache_path, "rb") as handle:  # local cache written by this function
            cached = pickle.load(handle)
        if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
            return cached["participants"]
        logger.info("Beat cache is stale (configuration changed); rebuilding")
    participants = [p for p in discover_participants(data_root)
                    if os.path.exists(os.path.join(data_root, p, "Event.csv"))]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        cache = {p: d for p, d in pool.map(_collect_participant, [(data_root, p, config) for p in participants]) if d}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump({"fingerprint": fingerprint, "participants": cache}, handle)
    return cache


# ---------------------------------------------------------------------------
# Figure 1 - pipeline overview
# ---------------------------------------------------------------------------

def figure_pipeline(output_dir: str) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(5.46, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 88)
    ax.axis("off")
    ax.grid(False)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    def rbox(x, y, w, h, fc, ec, lw=1.2, **kw):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.0", fc=fc, ec=ec,
                                    lw=lw, **kw))

    def arrow(x0, y0, x1, y1, lw=1.1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=lw,
                                     color="#444444", shrinkA=0, shrinkB=0))

    # --- sensors -------------------------------------------------------------------
    sensors = [("Polar H10 (ground truth)", "ECG 130 Hz | ACC 200 Hz", "#000000"),
               ("Empatica E4", "BVP 64 Hz | ACC 32 Hz", C["E4"]),
               ("Galaxy Watch 5", "PPG 25 Hz | ACC 25 Hz", C["GalaxyWatch"])]
    sw = 31.3
    sx = [1.0 + i * 33.35 for i in range(3)]
    for x, (name, rate, color) in zip(sx, sensors, strict=False):
        rbox(x, 79.5, sw, 8.0, color, color)
        ax.text(x + sw / 2, 84.5, name, ha="center", va="center", fontsize=7.8, fontweight="bold", color="white")
        ax.text(x + sw / 2, 81.6, rate, ha="center", va="center", fontsize=6.8, color="white")
        ax.plot([x + sw / 2, x + sw / 2], [76.8, 79.5], color="#444444", lw=1.1)
    ax.plot([sx[0] + sw / 2, sx[2] + sw / 2], [76.8, 76.8], color="#444444", lw=1.1)
    arrow(50, 76.8, 50, 74.0)

    # --- six stages (two rows of three) ------------------------------------------------
    stages = [
        ("1  Time base", ["fitted sample clock", "(true rate, gaps kept)", "clock sync from accelerometer",
                          "activity: session, then", "per window"], "#355C7D"),
        ("2  Preparation", ["fixed device polarity", "missing-sample / gain repair", "motion class from ACC",
                            "adaptive noise cancel. (E4)", "zero-phase band-pass filter"], "#6C5B7B"),
        ("3  Detection", ["NeuroKit2, SciPy,", "slope-sum, template", "consensus ensemble",
                          "priority: device x motion", "(leave-one-subject-out)"], "#C06C84"),
        ("4  Beat timing + QC", ["onset anchor, tangent foot", "on a wide-band signal",
                                 "template quality per beat", "Lipponen-Tarvainen fix",
                                 "signal-based beat recovery"], "#E0626E"),
        ("5  HRV", ["gap-aware segments", "burden, jitter, quality", "and cadence-lock gates",
                    "time, Poincare, SampEn,", "spectral (64 s Welch)"], "#D9822B"),
        ("6  Validation", ["per-window delay estimate", "1:1 matching +-100 ms",
                           "chance-level surrogate", "beat-to-beat IBI error",
                           "ECG HRV over same spans"], "#2A9D8F"),
    ]
    w, h = 31.3, 22.0
    ys = [51.0, 25.0]
    pos = []
    for i in range(6):
        pos.append((sx[i % 3], ys[i // 3]))
    for (x, y0), (title, lines, color) in zip(pos, stages, strict=False):
        rbox(x, y0, w, h, "white", color, lw=1.5)
        rbox(x, y0 + h - 5.0, w, 5.0, color, color, lw=1.5)
        ax.text(x + w / 2, y0 + h - 2.5, title, ha="center", va="center", fontsize=7.6, fontweight="bold",
                color="white")
        for j, line in enumerate(lines):
            ax.text(x + w / 2, y0 + h - 8.2 - j * 3.1, line, ha="center", va="center", fontsize=6.8)
    for i in (0, 1):
        arrow(pos[i][0] + w + 0.1, ys[0] + h / 2 - 2.5, pos[i + 1][0] - 0.1, ys[0] + h / 2 - 2.5)
    for i in (3, 4):
        arrow(pos[i][0] + w + 0.1, ys[1] + h / 2 - 2.5, pos[i + 1][0] - 0.1, ys[1] + h / 2 - 2.5)
    # elbow from stage 3 (end of row 1) back to stage 4 (start of row 2)
    c3, c4 = pos[2][0] + w / 2, pos[3][0] + w / 2
    ax.plot([c3, c3], [ys[0], 48.5], color="#444444", lw=1.1)
    ax.plot([c3, c4], [48.5, 48.5], color="#444444", lw=1.1)
    arrow(c4, 48.5, c4, ys[1] + h + 0.1)

    # --- outputs / design ------------------------------------------------------------
    arrow(pos[5][0] + w / 2, ys[1] - 0.1, pos[5][0] + w / 2, 18.2)
    rbox(1.0, 0.8, 58.0, 17.0, "#F4F4F4", "#999999", lw=1.0)
    ax.text(2.4, 14.6, "Outputs per activity window", fontsize=7.8, fontweight="bold")
    for j, line in enumerate(["HR; beat F1, timing SD, beat-to-beat IBI error;",
                              "accepted HRV segments (yield); RMSSD, SDNN, SD1/SD2,",
                              "SampEn, LF, HF, with ECG values over identical spans"]):
        ax.text(2.4, 10.4 - j * 3.4, line, fontsize=6.8)
    rbox(61.0, 0.8, 38.0, 17.0, "#F4F4F4", "#999999", lw=1.0)
    ax.text(62.4, 14.6, "Evaluation design", fontsize=7.8, fontweight="bold")
    for j, line in enumerate(["decisions on dev P02-P16;", "untouched test P17-P24;",
                              "primary run: leave-one-subject-out"]):
        ax.text(62.4, 10.4 - j * 3.4, line, fontsize=6.8)
    save_figure(fig, output_dir, "fig01_pipeline")


# ---------------------------------------------------------------------------
# Figure 2 - signal chain
# ---------------------------------------------------------------------------

def figure_signal_chain(data_root: str, config_path: str, output_dir: str,
                        participant: str = "P02", zoom_s: float = 12.0) -> None:
    from run_pipeline import _window, load_participant, process_participant
    from src.utils import load_config
    from src.validation import pair_beats

    config = load_config(config_path)
    folder = os.path.join(data_root, participant)
    out = process_participant(folder, config)
    acc_data, _ = load_participant(folder, config)
    columns = [("E4", "baseline"), ("GalaxyWatch", "baseline"), ("E4", "walking")]
    fig, axes = plt.subplots(3, 3, figsize=(5.46, 6.6), gridspec_kw={"height_ratios": [2.2, 0.9, 1.6]})
    tolerance = config["validation"]["match_tolerance_s"]
    for col, (device, activity) in enumerate(columns):
        res, ext = out["ppg"][(device, activity)]
        lag = ext["lag_s"]
        start, end = out["windows"][activity]
        filtered = ext["signal_filtered"]  # the exact signal the beats were detected on
        t = ext["clock_t0_s"] + ext["clock_seconds"]
        t_rel0 = t[0] + 60.0
        reference = out["reference"][activity]
        ecg = reference["beats_s"] + lag
        det = ext["detected_s"]
        pairs = pair_beats(ecg, det, tolerance)
        matched = np.zeros(len(det), bool)
        matched[pairs[:, 1]] = True
        zoom = (t >= t_rel0) & (t <= t_rel0 + zoom_s)
        scale = np.std(filtered[zoom]) + 1e-9

        ax = axes[0, col]
        sig = filtered[zoom] / scale
        ax.plot(t[zoom] - t_rel0, sig, color=C[device], lw=0.9)
        sel = (det >= t_rel0) & (det <= t_rel0 + zoom_s)
        y = np.interp(det[sel], t, filtered / scale)
        ax.scatter(det[sel][matched[sel]] - t_rel0, y[matched[sel]], s=16, color=C["green"], zorder=3,
                   label="beat, matched")
        ax.scatter(det[sel][~matched[sel]] - t_rel0, y[~matched[sel]], s=22, marker="x", color="k", zorder=3,
                   label="beat, unmatched")
        # systolic peak of every detected pulse (orientation cue)
        fs_dev = 1.0 / float(np.median(np.diff(ext["clock_seconds"])))
        foot_idx = np.interp(det[sel], t, np.arange(len(t)))
        peak_t = []
        ibi_s = float(np.median(np.diff(det))) if len(det) > 2 else 0.8
        for fi in foot_idx:
            a, b = int(fi), min(len(filtered) - 1, int(fi + 0.45 * ibi_s * fs_dev))
            if b > a:
                peak_t.append(t[a + int(np.argmax(filtered[a:b + 1]))])
        peak_t = np.asarray(peak_t)
        ax.scatter(peak_t - t_rel0, np.interp(peak_t, t, filtered / scale), s=14, marker="v",
                   facecolor="white", edgecolor="k", lw=0.6, zorder=3, label="systolic peak")
        # inset: median pulse (foot-aligned), shows upright orientation
        feet = np.round(np.interp(det, t, np.arange(len(t)))).astype(int)
        beats = []
        for a, b in itertools.pairwise(feet):
            if 0.3 * fs_dev < b - a < 1.6 * fs_dev:
                seg = filtered[a:b + 1]
                beats.append(np.interp(np.linspace(0, 1, 60), np.linspace(0, 1, len(seg)), seg))
        if len(beats) > 10:
            avg = np.median(beats, axis=0)
            avg = (avg - avg.min()) / (np.ptp(avg) + 1e-9)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.45 * (hi - lo))  # headroom so the inset never hides the trace
            inset = ax.inset_axes([0.03, 0.72, 0.20, 0.25])
            inset.plot(np.linspace(0, 1, 60), avg, color=C[device], lw=1.2)
            inset.scatter([0], [avg[0]], s=10, color=C["green"], zorder=3)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_linewidth(0.4)
        esel = (ecg >= t_rel0) & (ecg <= t_rel0 + zoom_s)
        ymin = sig.min() - 0.6
        ax.vlines(ecg[esel] - t_rel0, ymin, ymin + 0.5, color=C["PolarH10"], lw=1.0,
                  label="ECG R-peak + delay")
        ax.set_ylim(bottom=ymin - 0.15)  # keep the ECG tick row visible
        ax.text(0.99, 0.97, f"delay {lag*1000:.0f} ms", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.3)
        motion = "rest" if activity in REST else "walking"
        ax.set_title(f"{DEVICE_LABELS[device]} - {motion}")
        ax.set_ylabel("Filtered PPG (z)" if col == 0 else "")
        ax.set_xlim(0, zoom_s)
        _panel(ax, "abc"[col])

        ax = axes[1, col]
        acc = _window(acc_data[device]["acc"], start, end)
        ta = (acc.index - pd.Timestamp("1970-01-01")).total_seconds().to_numpy()
        ta = ta - (res.get("clock_offset_s", 0.0) - res.get("clock_offset_session_s", 0.0))
        mag = np.sqrt(acc.acc_x_g ** 2 + acc.acc_y_g ** 2 + acc.acc_z_g ** 2).to_numpy()
        az = (ta >= t_rel0) & (ta <= t_rel0 + zoom_s)
        ax.plot(ta[az] - t_rel0, mag[az], color=C["grey"], lw=0.7)
        ax.set_ylabel("|acc| (g)" if col == 0 else "")
        ax.set_xlim(0, zoom_s)

        ax = axes[2, col]
        ecg_ibi = np.diff(reference["beats_s"]) * 1000
        ax.plot(reference["beats_s"][1:] + lag - t[0], ecg_ibi, color="k", lw=1.0)
        ax.plot(det[1:] - t[0], np.diff(det) * 1000, color=C[device], lw=0.8, alpha=0.85)
        for seg in ext.get("segments", []):
            a, b = pd.Timestamp(seg["start"]).value / 1e9, pd.Timestamp(seg["end"]).value / 1e9
            ax.axvspan(a - t[0], b - t[0], color=C["green"], alpha=0.12, lw=0)
        ax.set_ylim(250, 1500)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("IBI (ms)" if col == 0 else "")
        ax.set_xlim(0, t[-1] - t[0])
        ax.text(0.03, 0.97, f"recall {res.get('recall_vs_ecg', np.nan):.0f}%\n"
                            f"precision {res.get('precision_vs_ecg', np.nan):.0f}%",
                transform=ax.transAxes, fontsize=6.3, va="top",
                bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1.0})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], color="k", label="ECG IBI"), Line2D([], [], color="grey", label="PPG IBI"),
                Patch(color=C["green"], alpha=0.25, label="accepted HRV segment")]
    labels += ["ECG IBI", "PPG IBI (colour = device)", "accepted HRV segment"]
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.4)
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=6.0, columnspacing=0.8,
               handletextpad=0.3, bbox_to_anchor=(0.5, 1.0))
    save_figure(fig, output_dir, "fig02_signal_chain")


# ---------------------------------------------------------------------------
# Figure 2 - clock synchronisation
# ---------------------------------------------------------------------------

def figure_clock_sync(df: pd.DataFrame, cache: dict, data_root: str, config_path: str,
                      output_dir: str, participant: str = "P15", mislock: tuple[str, str] = ("P04", "E4")) -> None:
    from run_pipeline import extract_activity_windows
    from src.data_loader import load_device_data
    from src.sync import activity_envelope
    from src.utils import load_config

    config = load_config(config_path)
    folder = os.path.join(data_root, participant)
    pol = load_device_data(folder, "PolarH10", config)["acc"]
    gw = load_device_data(folder, "GalaxyWatch", config)["acc"]
    rows = df[(df["participant"] == participant) & (df["device"] == "GalaxyWatch")]
    events = pd.read_csv(os.path.join(folder, "Event.csv"))
    events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")
    windows = extract_activity_windows(events, config["activities"])

    fig = plt.figure(figsize=(5.46, 7.6))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.5, wspace=0.34)

    # (a) excerpt around walking onset, offset applied at that window
    ax = fig.add_subplot(grid[0, 0])
    start = windows["walking"][0]
    applied = float(rows[rows["activity"] == "walking"]["clock_offset_s"].iloc[0])
    t0, t1 = start - pd.Timedelta(seconds=40), start + pd.Timedelta(seconds=110)
    grid_t = np.arange(0, (t1 - t0).total_seconds(), 0.1)
    shifted = gw.copy()
    shifted.index = shifted.index - pd.Timedelta(seconds=applied)
    ax.plot(grid_t, activity_envelope(pol, t0, t1), color="k", lw=1.0, label="Polar H10 (reference)")
    ax.plot(grid_t, activity_envelope(gw, t0, t1) + 2.6, color=C["GalaxyWatch"], lw=0.9, alpha=0.55,
            label="Watch, raw clock")
    ax.plot(grid_t, activity_envelope(shifted, t0, t1) + 5.2, color=C["GalaxyWatch"], lw=1.0,
            label=f"Watch, shifted by {applied:+.1f} s")
    ax.axvline(40.0, color=C["grey"], lw=0.8, ls=":")
    ax.text(41, 0.03, "walking starts", fontsize=6.5, va="bottom", color=C["grey"],
            transform=ax.get_xaxis_transform())
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Accelerometer envelopes ({participant})")
    ax.set_ylim(top=ax.get_ylim()[1] + 2.4)
    ax.legend(loc="upper left", fontsize=6.0, ncol=1, handlelength=1.4)
    _panel(ax, "a")

    # (b) per-window offsets: P15 watch (a real step) and an isolated mis-lock that is rejected
    ax = fig.add_subplot(grid[0, 1])
    series = [(participant, "GalaxyWatch", C["GalaxyWatch"], "P15 watch: real clock step"),
              (mislock[0], mislock[1], C["E4"], f"{mislock[0]} E4: isolated mis-lock")]
    for pid, dev, color, label in series:
        r = df[(df["participant"] == pid) & (df["device"] == dev)].copy()
        ev = pd.read_csv(os.path.join(data_root, pid, "Event.csv"))
        ev["datetime"] = pd.to_datetime(ev["timestamp"], unit="ms")
        win = extract_activity_windows(ev, config["activities"])
        r["order"] = r["activity"].map({a: i for i, a in enumerate(sorted(win, key=lambda k: win[k][0]))})
        r = r.sort_values("order")
        ax.plot(r["order"], r["clock_offset_s"], color=color, lw=1.6, label=f"{label} (applied)")
        rejected = r["clock_offset_source"] != "local"
        ax.scatter(r["order"][~rejected], r["clock_local_estimate_s"][~rejected], s=16, color=color, zorder=3,
                   label="accepted local estimate" if dev == "GalaxyWatch" else None)
        ax.scatter(r["order"][rejected], r["clock_local_estimate_s"][rejected], s=34, marker="x", color="k",
                   zorder=4, label="rejected estimate" if dev == "GalaxyWatch" else None)
    ax.set_xlabel("Protocol window (chronological)")
    ax.set_ylabel("Clock offset vs Polar (s)")
    ax.set_title("Per-window clock offsets")
    ax.set_ylim(top=ax.get_ylim()[1] + 1.5)
    ax.legend(fontsize=5.8, loc="upper right", handlelength=1.4, borderaxespad=0.1)
    _panel(ax, "b")

    # (c) session offsets
    ax = fig.add_subplot(grid[1, 0])
    per = df[df["device"].isin(PPG_DEVICES)].groupby(["participant", "device"])["clock_offset_session_s"].first().unstack()
    for i, device in enumerate(PPG_DEVICES):
        v = per[device].dropna()
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v,
                   color=C[device], s=14, alpha=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks([0, 1], [DEVICE_LABELS[d] for d in PPG_DEVICES], rotation=20, ha="right")
    ax.set_ylabel("Session offset vs Polar (s)")
    ax.set_title("Session offsets")
    _panel(ax, "c")

    # (d) beat correlogram
    ax = fig.add_subplot(grid[1, 1])
    p = cache.get("P02", {}).get("windows", {}).get(("E4", "baseline"))
    if p is not None:
        det, ecg = np.sort(p["detected_s"]), np.sort(p["ecg_s"])
        diffs = np.concatenate([d - ecg[(ecg > d - 1.0) & (ecg < d + 1.0)] for d in det])
        ax.hist(diffs * 1000, bins=np.arange(-1000, 1005, 10), color=C["E4"], alpha=0.85)
        ax.axvline(p["lag_s"] * 1000, color="k", ls="--", lw=0.8)
        ax.text(p["lag_s"] * 1000 - 40, ax.get_ylim()[1] * 0.85, f"delay\n{p['lag_s']*1000:.0f} ms", fontsize=7,
                ha="right")
    ax.set_xlabel("PPG beat - ECG R-peak (ms)")
    ax.set_ylabel("Beat pairs")
    ax.set_title("Beat cross-correlogram")
    ax.text(0.02, 0.97, "E4, P02, baseline", transform=ax.transAxes, fontsize=6.5, va="top")
    _panel(ax, "d")

    # (e) delay by condition
    ax = fig.add_subplot(grid[2, :])
    rest = df[df["device"].isin(PPG_DEVICES)]
    for i, device in enumerate(PPG_DEVICES):
        for j, condition in enumerate(COND):
            v = rest[(rest["device"] == device) & (rest["condition"] == condition)]["lag_ms"].dropna()
            ax.boxplot(v, positions=[j + (i - 0.5) * 0.35], widths=0.3, patch_artist=True, showfliers=False,
                       boxprops={"facecolor": C[device], "alpha": 0.5}, medianprops={"color": "k"})
    _cond_ticks(ax, range(len(COND)), COND)
    ax.set_ylabel("Residual delay (ms)")
    ax.set_title("Residual device-ECG delay by condition")
    ax.legend(handles=[Patch(color=C[d], alpha=0.5, label=DEVICE_LABELS[d]) for d in PPG_DEVICES], fontsize=6.3, loc="center right")
    _panel(ax, "e")
    save_figure(fig, output_dir, "fig03_clock_sync")


# ---------------------------------------------------------------------------
# Figure 3 - matching tolerance
# ---------------------------------------------------------------------------

def figure_matching_tolerance(cache: dict, output_dir: str) -> None:
    from src.validation import match_peaks
    tolerances = np.array([10, 25, 50, 75, 100, 150, 200, 300, 400, 500]) / 1000.0
    curves = {}
    for data in cache.values():
        for (device, activity), w in data["windows"].items():
            group = "rest" if activity in REST else ("locomotion" if activity in MOTION else None)
            if group is None or len(w["detected_s"]) < 10:
                continue
            det, ecg, lag = w["detected_s"], w["ecg_s"], w["lag_s"]
            half = 0.5 * np.median(np.diff(det))
            for kind, d, lg in (("aligned", det, lag), ("unaligned", det, 0.0), ("chance", det + half, lag)):
                curves.setdefault((device, group, kind), []).append(
                    [match_peaks(ecg, d, tol, lg)["recall"] for tol in tolerances])
    fig, axes = plt.subplots(1, 2, figsize=(5.46, 3.6), sharey=True)
    styles = {"aligned": ("-", "delay-corrected"), "unaligned": (":", "no delay correction"),
              "chance": ("--", "chance (half-beat shifted)")}
    for ax, group, letter in zip(axes, ("rest", "locomotion"), "ab", strict=False):
        for device in PPG_DEVICES:
            for kind, (ls, label) in styles.items():
                values = np.asarray(curves.get((device, group, kind), []))
                if values.size == 0:
                    continue
                ax.plot(tolerances * 1000, values.mean(axis=0), ls=ls, color=C[device], lw=1.4,
                        label=f"{DEVICE_LABELS[device].split()[-2] if device == 'GalaxyWatch' else 'E4'}: {label}")
        ax.axvline(100, color="k", lw=0.7)
        ax.axvline(500, color=C["grey"], lw=0.7, ls="-.")
        ax.set_ylim(-3, 118)
        ax.text(105, 103, "\u00b1100 ms", fontsize=6.3, ha="left", va="bottom")
        ax.text(494, 103, "\u00b1500 ms", fontsize=6.3, color=C["grey"], ha="right", va="bottom")
        ax.set_xlabel("Matching tolerance (ms)")
        ax.set_title("Seated rest" if group == "rest" else "Locomotion")
        ax.set_xlim(0, 560)
        _panel(ax, letter)
    axes[0].set_ylabel("Beat recall vs ECG (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.0, columnspacing=1.0, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.27)
    save_figure(fig, output_dir, "fig04_matching_tolerance")


# ---------------------------------------------------------------------------
# Figure 4 - cohort flow / yield
# ---------------------------------------------------------------------------

def figure_cohort_flow(flow: pd.DataFrame, output_dir: str) -> None:
    flow = flow[flow["split"] == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(5.46, 3.3), sharey=True)
    for ax, device, letter in zip(axes, PPG_DEVICES, "ab", strict=False):
        order = [*COND, "adaptation"]
        g = flow[flow["device"] == device].set_index("condition").reindex(order)
        x = np.arange(len(order))
        total = g["n_windows"].to_numpy(float)
        accepted = g["n_hrv_paired_with_ecg"].to_numpy(float)
        ax.bar(x, total, color="#DDDDDD", label="activity windows")
        ax.bar(x, g["n_reference_valid"], color="#BBBBBB", width=0.6, label="valid ECG reference")
        ax.bar(x, accepted, color=C[device], width=0.6, label="HRV accepted & paired")
        for xi, a, t in zip(x, accepted, total, strict=False):
            ax.text(xi, max(a, 0) + 0.4, f"{100*a/t:.0f}%", ha="center", fontsize=7)
        _cond_ticks(ax, x, order)
        ax.set_title(DEVICE_LABELS[device])
        _panel(ax, letter)
    axes[0].set_ylabel("Activity windows (23 participants)")
    handles, _ = axes[0].get_legend_handles_labels()
    handles[-1] = Patch(color="#555555", label="HRV accepted & paired with ECG")
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save_figure(fig, output_dir, "fig06_cohort_flow")


# ---------------------------------------------------------------------------
# Figure 5 - beat detection vs motion
# ---------------------------------------------------------------------------

def figure_beat_detection(df: pd.DataFrame, output_dir: str) -> None:
    ppg = df[df["device"].isin(PPG_DEVICES)]
    fig, axes = plt.subplots(1, 2, figsize=(5.46, 3.2), sharey=True)
    for ax, device, letter in zip(axes, PPG_DEVICES, "ab", strict=False):
        g = ppg[ppg["device"] == device]
        x = np.arange(len(COND))
        for split, offset in (("dev", -0.12), ("test", 0.12)):
            s = g[g["split"] == split]
            for xi, condition in zip(x, COND, strict=False):
                v = s[s["condition"] == condition].groupby("participant")["f1_vs_ecg"].mean().dropna()
                ax.scatter(np.full(len(v), xi + offset) + np.random.default_rng(xi).uniform(-0.05, 0.05, len(v)),
                           v, s=9, color=C[device], alpha=0.35, marker=SPLIT_MARKERS[split], lw=0)
        med = g.groupby("condition", observed=True)["f1_vs_ecg"].median().reindex(COND)
        chance = g.groupby("condition", observed=True)["recall_chance_vs_ecg"].median().reindex(COND)
        vendor = g.groupby("condition", observed=True)["vendor_ibi_coverage_pct"].apply(
            lambda s: s.fillna(0).median()).reindex(COND)
        ax.plot(x, med, color=C[device], lw=2, marker="o", label="pipeline beat F1 (median)")
        ax.fill_between(x, 0, chance, color=C["grey"], alpha=0.25, label="chance level")
        ax.plot(x, vendor, color="k", lw=1.2, ls="--", marker="s", ms=3, label="vendor IBI coverage")
        _cond_ticks(ax, x, COND)
        ax.set_ylim(-2, 104)
        ax.set_title(DEVICE_LABELS[device])
        _panel(ax, letter)
    axes[0].set_ylabel("Beat-level F1 vs ECG (%, +-100 ms)")
    handles = axes[1].get_legend_handles_labels()[0] + [
        Line2D([], [], marker="o", ls="", color="grey", label="dev participant"),
        Line2D([], [], marker="D", ls="", color="grey", label="test participant")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save_figure(fig, output_dir, "fig05_beat_detection")


# ---------------------------------------------------------------------------
# Figure 6 - HR agreement
# ---------------------------------------------------------------------------

def figure_hr_agreement(df: pd.DataFrame, output_dir: str) -> None:
    ppg = df[df["device"].isin(PPG_DEVICES)].dropna(subset=["hr_median", "ecg_hr"])
    fig, axes = plt.subplots(2, 2, figsize=(5.46, 5.2), sharex="row", sharey="row")
    for col, device in enumerate(PPG_DEVICES):
        for row, (group, conditions) in enumerate((("Seated rest", ("rest",)),
                                                   ("All other conditions",
                                                    ("seated_task", "hand_activity", "standing", "recovery",
                                                     "locomotion")))):
            ax = axes[row, col]
            g = ppg[(ppg["device"] == device) & ppg["condition"].isin(conditions)]
            mean = (g["hr_median"] + g["ecg_hr"]) / 2
            diff = g["hr_median"] - g["ecg_hr"]
            for condition in conditions:
                for split in ("dev", "test"):
                    m = (g["condition"] == condition) & (g["split"] == split)
                    ax.scatter(mean[m], diff[m], s=12, color=COND_COLORS[condition], marker=SPLIT_MARKERS[split],
                               alpha=0.75, lw=0.3, edgecolor="white")
            ba = bland_altman(g["hr_median"], g["ecg_hr"], g["participant"])
            for value, ls in ((ba["bias"], "-"), (ba["loa_low"], "--"), (ba["loa_high"], "--")):
                ax.axhline(value, color="k", lw=0.8, ls=ls)
            ax.text(0.01, 0.98, f"bias {ba['bias']:.1f} bpm\nLoA [{ba['loa_low']:.1f}, {ba['loa_high']:.1f}] bpm",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.3,
                    bbox={"facecolor": "white", "alpha": 0.85, "lw": 0, "pad": 1.5})
            if row == 1:
                # Half-rate detection: PPG = ECG/2 -> difference = -(2/3) x mean
                xs = np.linspace(50, 120, 50)
                ax.plot(xs, -2 * xs / 3, color=C["grey"], lw=0.8, ls=":")
                ax.text(xs[-1], -2 * xs[-1] / 3 - 4, "half-rate locus", fontsize=6.5, color=C["grey"], ha="right",
                        va="top")
            ax.set_title(f"{DEVICE_LABELS[device]}: {group.replace('All other conditions', 'other conditions')}", fontsize=7.5)
            ax.set_xlabel("Mean of PPG and ECG HR (bpm)")
            ax.set_ylabel("PPG - ECG HR (bpm)")
            _panel(ax, "abcd"[row * 2 + col])
    handles = [Line2D([], [], marker="o", ls="", color=COND_COLORS[c], label=CONDITION_LABELS[c]) for c in COND]
    handles += [Line2D([], [], marker=m, ls="", color="grey", label=s) for s, m in SPLIT_MARKERS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, output_dir, "fig07_hr_agreement")


# ---------------------------------------------------------------------------
# Figure 7 - beat-to-beat IBI agreement
# ---------------------------------------------------------------------------

def figure_beat_ibi(cache: dict, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.46, 3.5), sharey=True)
    for ax, device, letter in zip(axes, PPG_DEVICES, "ab", strict=False):
        ref, det, subj = [], [], []
        for participant, data in cache.items():
            for activity in REST:
                w = data["windows"].get((device, activity))
                if w is None or not w.get("hrv_ibi"):
                    continue
                ref.append(w["hrv_ibi"]["ibi_reference"])
                det.append(w["hrv_ibi"]["ibi_detected"])
                subj.append(np.full(len(w["hrv_ibi"]["ibi_reference"]), participant))
        if not ref:
            continue
        ref, det, subj = np.concatenate(ref), np.concatenate(det), np.concatenate(subj)
        mean, diff = (ref + det) / 2, det - ref
        hb = ax.hexbin(mean, diff, gridsize=55, extent=(450, 1350, -120, 120), mincnt=1, cmap="viridis", bins="log")
        ba = bland_altman(det, ref, subj)
        for value, ls in ((ba["bias"], "-"), (ba["loa_low"], "--"), (ba["loa_high"], "--")):
            ax.axhline(value, color="w", lw=1.1, ls=ls)
            ax.axhline(value, color="k", lw=0.5, ls=ls)
        ax.text(0.02, 0.97, f"{len(diff):,} beats, {len(set(subj))} part.\n"
                            f"bias {ba['bias']:.1f} ms, LoA [{ba['loa_low']:.0f}, {ba['loa_high']:.0f}] ms\n"
                            f"MAE {np.mean(np.abs(diff)):.1f} ms",
                transform=ax.transAxes, va="top", fontsize=5.8, bbox={"facecolor": "white", "alpha": 0.8, "lw": 0})
        ax.set_title(DEVICE_LABELS[device])
        ax.set_xlabel("Mean of PPG and ECG IBI (ms)")
        _panel(ax, letter)
    axes[0].set_ylabel("PPG - ECG IBI (ms)")
    fig.colorbar(hb, ax=axes, label="beats (log)", shrink=0.85)
    save_figure(fig, output_dir, "fig08_beat_ibi_agreement")


# ---------------------------------------------------------------------------
# Figure 8 - HRV forest plot
# ---------------------------------------------------------------------------

def figure_hrv_forest(hrv: pd.DataFrame, output_dir: str) -> None:
    metrics = ["sdnn", "rmssd", "sd1", "sd2", "hf_power", "lf_power", "lf_hf_ratio"]
    labels = {"sdnn": "SDNN", "rmssd": "RMSSD", "sd1": "SD1", "sd2": "SD2", "hf_power": "HF power",
              "lf_power": "LF power", "lf_hf_ratio": "LF/HF"}
    bounds = {"sdnn": 0.10, "rmssd": 0.10, "sd1": 0.10, "sd2": 0.10, "hf_power": 0.20, "lf_power": 0.20,
              "lf_hf_ratio": 0.20}
    g = hrv[(hrv["condition"] == "rest") & hrv["metric"].isin(metrics)]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    y = 0
    ticks, tick_labels = [], []
    for metric in metrics:
        b = bounds[metric]
        ax.fill_between([np.exp(-b), np.exp(b)], y - 0.45, y + 2.6, color=C["green"], alpha=0.10, lw=0)
        for device in PPG_DEVICES:
            for split, dy in (("dev", 0.0), ("test", 0.35)):
                r = g[(g["metric"] == metric) & (g["device"] == device) & (g["split"] == split)]
                if r.empty or not np.isfinite(r["ratio"].iloc[0]) or r["n"].iloc[0] < 5:
                    continue
                r = r.iloc[0]
                yy = y + dy
                ax.plot([r["ratio_loa_low"], r["ratio_loa_high"]], [yy, yy], color=C[device], lw=1.2,
                        alpha=0.9 if split == "test" else 0.5)
                ax.scatter(r["ratio"], yy, color=C[device], marker=SPLIT_MARKERS[split], s=26, zorder=3,
                           edgecolor="k" if r["equivalent"] else "none", lw=0.8)
                if split == "test":
                    ax.text(4.3, yy - 0.17, f"n={int(r['n'])}, ICC {r['icc_a1']:.2f}", fontsize=6.4, va="center")
            y += 1
        ticks.append(y - 1.5)
        tick_labels.append(labels[metric])
        y += 0.8
    ax.axvline(1, color="k", lw=0.7)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3], ["0.5", "0.75", "1", "1.5", "2", "3"])
    ax.set_xlim(0.4, 6.2)
    ax.set_yticks(ticks, tick_labels)
    ax.invert_yaxis()
    ax.set_xlabel("PPG / ECG ratio (geometric mean, 95% limits of agreement)")
    ax.set_title("HRV agreement in seated rest over identical spans", pad=16)
    ax.text(0.0, 1.01, "shaded: a-priori equivalence bounds; rows with n < 5 omitted; labels: test split",
            transform=ax.transAxes, fontsize=6.5, color=C["grey"])
    handles = [Line2D([], [], color=C[d], lw=2, label=DEVICE_LABELS[d]) for d in PPG_DEVICES]
    handles += [Line2D([], [], marker=m, ls="", color="grey", label=f"{s} split") for s, m in SPLIT_MARKERS.items()]
    handles += [Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="equivalent (TOST p<0.05)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=6.8)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig09_hrv_forest")


# ---------------------------------------------------------------------------
# Figure 9 - mechanism of RMSSD inflation
# ---------------------------------------------------------------------------

def figure_rmssd_mechanism(timing: pd.DataFrame, cache: dict, windows: pd.DataFrame, output_dir: str) -> None:
    """Why PPG RMSSD exceeds ECG RMSSD: structure of the beat-timing error."""
    from analysis.timing_error import _runs
    from src.validation import pair_beats

    rest = timing[(timing["condition"] == "rest") & (timing["n_beats"] >= 30)]
    fig = plt.figure(figsize=(5.46, 5.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.5, wspace=0.42)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    ax = axes[0]
    for device in PPG_DEVICES:
        g = rest[(rest["device"] == device) & rest["sigma_ms"].notna()]
        target = g.loc[(g["sigma_ms"] - g["sigma_ms"].median()).abs().idxmin()]
        w = cache[target["participant"]]["windows"][(device, target["activity"])]
        det, ecg, lag = np.sort(w["detected_s"]), np.sort(w["ecg_s"]), w["lag_s"]
        pairs = pair_beats(ecg, det - lag, 0.10)
        err = (det[pairs[:, 1]] - lag - ecg[pairs[:, 0]]) * 1000.0
        run = max(_runs(pairs[:, 0], pairs[:, 1]), key=len)
        e = err[run][:60]
        ax.plot(np.arange(len(e)), e - np.mean(e), color=C[device], lw=1.4, marker="o", ms=2.8,
                label=f"{DEVICE_LABELS[device]} ({target['participant']}, {ACTIVITY_LABELS[target['activity']]})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Consecutive beats")
    ax.set_ylabel("PPG - ECG beat time (ms)")
    ax.set_title("Timing error across consecutive beats (mean removed)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
    ax.legend(fontsize=6.0, loc="upper center", ncol=2, columnspacing=1.0)
    _panel(ax, "a")

    ax = axes[1]
    positions, data, colors = [], [], []
    for k, device in enumerate(PPG_DEVICES):
        g = rest[(rest["device"] == device)]
        for j, col in enumerate(("rho1", "whiteness")):
            positions.append(k * 3.6 + j * 1.5)
            data.append(g[col].dropna())
            colors.append(C[device])
    bp = ax.boxplot(data, positions=positions, widths=1.0, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for median in bp["medians"]:
        median.set_color("k")
    ax.axhline(1.0, color=C["grey"], lw=0.7, ls=":")
    ax.text(-0.8, 1.02, "white errors", fontsize=6.0, color=C["grey"], ha="left")
    ax.set_xticks(positions, ["lag-1\nr", "white-\nness"] * 2, fontsize=6.3)
    ax.set_ylim(0, 1.2)
    ax.set_xlim(-0.9, 5.9)
    ax.set_title("Autocorrelation of errors")
    for k, device in enumerate(PPG_DEVICES):
        ax.text(k * 3.6 + 0.75, 1.1, DEVICE_LABELS[device].replace("Empatica ", "").replace(" 5", ""),
                ha="center", fontsize=6.8, color=C[device], fontweight="bold")
    _panel(ax, "b")

    ax = axes[2]
    w = windows[(windows["split"] == "all") & windows["metric"].isin(["sdnn", "rmssd"])]
    for device in PPG_DEVICES:
        for metric, ls in (("sdnn", "--"), ("rmssd", "-")):
            s = w[(w["device"] == device) & (w["metric"] == metric)].sort_values("window_s")
            ax.errorbar(s["window_s"], s["error_median"], yerr=[s["error_median"] - s["error_q25"],
                        s["error_q75"] - s["error_median"]], color=C[device], ls=ls, marker="o", ms=3,
                        capsize=2, lw=1.2, label=f"{'E4' if device == 'E4' else 'Watch'} {metric.upper()}")
    ax.axhline(np.log(1.1), color=C["grey"], lw=0.7, ls=":")
    ax.text(121, np.log(1.1) - 0.012, "10 %", fontsize=6.5, color=C["grey"], ha="right")
    ax.set_xticks([30, 60, 120])
    ax.set_xlabel("Analysis window (s)")
    ax.set_ylabel("|ln(PPG / ECG)|, median and IQR")
    ax.set_title("Error vs window")
    ax.set_ylim(top=ax.get_ylim()[1] + 0.17)
    ax.legend(fontsize=5.8, loc="upper right", ncol=2, columnspacing=0.6, handlelength=1.4, handletextpad=0.3)
    _panel(ax, "c")
    save_figure(fig, output_dir, "fig10_rmssd_mechanism")


# ---------------------------------------------------------------------------
# Figure 10 - ablation
# ---------------------------------------------------------------------------

ABLATION_LABELS = {
    "anc_off": "ANC off (E4)",
    "anc_both_devices": "ANC on both devices",
    "anc_all_motion": "ANC in all motion states (E4)",
    "no_beat_recovery": "no beat recovery",
    "recovered_in_differences": "recovered beats in differences",
    "no_timing_refinement": "no wide-band timing",
    "no_sqi": "no beat SQI screen",
    "sqi_0.70": "SQI threshold 0.70 (both)",
    "sqi_0.90": "SQI threshold 0.90",
    "rhythm_gate": "spectral rhythm gate",
    "no_cadence_gate": "no cadence-lock gate",
    "device_hr_gate": "device-HR gate (circular)",
    "strict": "strict gates (2 % burden)",
    "burden_10pct": "burden limit 10 %",
    "permissive": "permissive gates (15 %)",
    "min_segment_60s": "min segment 60 s",
    "neurokit2_only": "NeuroKit2 only",
}


def figure_ablation(summary: pd.DataFrame, output_dir: str) -> None:
    variants = [v for v in ABLATION_LABELS if v in set(summary["variant"])]
    panels = [("rest", "hrv_yield_pct", "Seated rest\nHRV yield (pp)"),
              ("rest", "rmssd_abs_log_error", "Seated rest\nRMSSD |ln ratio|"),
              ("locomotion", "f1_pct", "Locomotion\nbeat F1 (pp)"),
              ("seated_task", "hrv_yield_pct", "Seated task\nHRV yield (pp)")]
    fig, axes = plt.subplots(2, 2, figsize=(5.46, 6.2), sharey=True)
    y = np.arange(len(variants))
    for ax, (condition, metric, title), letter in zip(axes.ravel(), panels, "abcd", strict=False):
        for device, dy in (("E4", -0.18), ("GalaxyWatch", 0.18)):
            for split, alpha in (("dev", 0.45), ("test", 1.0)):
                s = summary[(summary["condition"] == condition) & (summary["device"] == device)
                            & (summary["split"] == split)].set_index("variant")
                if "base" not in s.index:
                    continue
                delta = (s[metric] - s.loc["base", metric]).reindex(variants)
                ax.scatter(delta, y + dy, color=C[device], marker=SPLIT_MARKERS[split], s=22, alpha=alpha, lw=0)
        ax.axvline(0, color="k", lw=0.7)
        ax.set_title(title)
        ax.set_xlabel("change vs base")
        _panel(ax, letter)
    for ax in axes[:, 0]:
        ax.set_yticks(y, [ABLATION_LABELS[v] for v in variants], fontsize=6.5)
    axes[0, 0].invert_yaxis()
    handles = [Line2D([], [], marker="o", ls="", color=C[d], label=DEVICE_LABELS[d]) for d in PPG_DEVICES]
    handles += [Line2D([], [], marker=m, ls="", color="grey", alpha=a, label=f"{s} participants")
                for (s, m), a in zip(SPLIT_MARKERS.items(), (0.45, 1.0), strict=False)]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.55, -0.03))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save_figure(fig, output_dir, "fig12_ablation")


# ---------------------------------------------------------------------------
# Figure 11 - external baselines
# ---------------------------------------------------------------------------

BASELINE_LABELS = {"pipeline": "This pipeline", "heartpy": "HeartPy", "nk_elgendi": "NeuroKit2 (Elgendi)",
                   "nk_msptdfast": "MSPTDfast", "vendor": "Vendor IBI"}
BASELINE_COLORS = {"pipeline": "#000000", "heartpy": "#E69F00", "nk_elgendi": "#56B4E9",
                   "nk_msptdfast": "#CC79A7", "vendor": "#999999"}


def figure_baselines(frame: pd.DataFrame, output_dir: str) -> None:
    methods = ["pipeline", "heartpy", "nk_elgendi", "nk_msptdfast"]
    conds = ["rest", "seated_task", "hand_activity", "standing", "recovery", "locomotion"]
    fig, axes = plt.subplots(2, 2, figsize=(5.46, 5.2))
    for col, device in enumerate(PPG_DEVICES):
        d = frame[frame["device"] == device]
        ax = axes[0, col]
        width = 0.8 / len(methods)
        for k, method in enumerate(methods):
            g = d[d["method"] == method]
            per = g.groupby(["condition", "participant"])["f1"].mean().groupby("condition").median().reindex(conds)
            ax.bar(np.arange(len(conds)) + (k - 1.5) * width, per, width, color=BASELINE_COLORS[method],
                   label=BASELINE_LABELS[method])
        _cond_ticks(ax, range(len(conds)), conds)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Beat F1 vs ECG (%, median)")
        ax.set_title(f"{DEVICE_LABELS[device]} - beat detection")
        _panel(ax, "ab"[col])

        ax = axes[1, col]
        r = d[d["condition"] == "rest"].dropna(subset=["ecg_rmssd_spans"])
        positions, data, colors = [], [], []
        for k, method in enumerate(methods):
            g = r[r["method"] == method]
            column = "rmssd" if method == "pipeline" else "rmssd_corrected"
            v = np.log(g[column] / g["ecg_rmssd_spans"]).replace([np.inf, -np.inf], np.nan).dropna()
            positions.append(k)
            data.append(v)
            colors.append(BASELINE_COLORS[method])
        parts = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
        for patch, color in zip(parts["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for median in parts["medians"]:
            median.set_color("k")
        ax.axhspan(-np.log(1.1), np.log(1.1), color=C["green"], alpha=0.12, lw=0)
        ax.axhline(0, color="k", lw=0.6)
        ticks = [0.5, 1, 2, 4, 8]
        ax.set_yticks(np.log(ticks), [f"x{t:g}" for t in ticks])
        ax.set_xticks(positions, [BASELINE_LABELS[m].replace(" (", "\n(") for m in methods], fontsize=6.8)
        ax.set_ylabel("RMSSD PPG / ECG (same spans)")
        ax.set_title(f"{DEVICE_LABELS[device]} - RMSSD in seated rest")
        _panel(ax, "cd"[col])
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.text(0.5, -0.005, "Baselines after the same Lipponen-Tarvainen correction; shaded band: +-10 %.",
             ha="center", fontsize=7, color=C["grey"])
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=6.8, bbox_to_anchor=(0.5, 1.0))
    save_figure(fig, output_dir, "fig11_baselines")


# ---------------------------------------------------------------------------
# Supplementary
# ---------------------------------------------------------------------------

def figure_reference_validity(df: pd.DataFrame, output_dir: str) -> None:
    ecg = df[df["device"] == "PolarH10"].dropna(subset=["reference_ibi_mae_ms"])
    fig, axes = plt.subplots(1, 2, figsize=(5.46, 2.6))
    for i, condition in enumerate(COND):
        v = ecg[ecg["condition"] == condition]
        axes[0].scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)),
                        v["reference_ibi_mae_ms"], s=8, color=COND_COLORS[condition], alpha=0.7)
        axes[1].scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)),
                        v["reference_within_5ms_pct"], s=8, color=COND_COLORS[condition], alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("R-R MAE vs Polar firmware (ms)")
    axes[1].set_ylabel("Intervals within 5 ms (%)")
    for ax, letter in zip(axes, "ab", strict=False):
        _cond_ticks(ax, range(len(COND)), COND)
        _panel(ax, letter)
    axes[0].set_title("R-R accuracy vs Polar firmware")
    fig.tight_layout()
    save_figure(fig, output_dir, "figA1_reference_validity")


def figure_fiducial(fiducial: pd.DataFrame, output_dir: str) -> None:
    order = ["peak", "max_slope", "tangent", "onset", "d2_max"]
    labels = {"peak": "systolic\npeak", "max_slope": "max\nslope", "tangent": "tangent\nfoot",
              "onset": "onset\n(min.)", "d2_max": "max 2nd\nderiv."}
    anchors = [("systolic", "detect systolic peak", 0.35), ("onset", "detect onset (used)", 0.85)]
    fig, axes = plt.subplots(2, 2, figsize=(5.46, 4.8), sharex=True)
    for row, device in enumerate(PPG_DEVICES):
        for col, (metric, ylabel) in enumerate((("timing_sd_ms", "Timing jitter SD vs ECG (ms)"),
                                                ("rmssd_ratio", "RMSSD ratio (matched beats)"))):
            ax = axes[row, col]
            for k, (anchor, _label, alpha) in enumerate(anchors):
                g = fiducial[(fiducial["device"] == device) & (fiducial.get("anchor", "systolic") == anchor)] \
                    if "anchor" in fiducial else fiducial[fiducial["device"] == device]
                data = [g[g["fiducial"] == f][metric].dropna() for f in order]
                pos = np.arange(len(order)) + (k - 0.5) * 0.36
                ax.boxplot(data, positions=pos, widths=0.32, patch_artist=True, showfliers=False,
                           boxprops={"facecolor": C[device], "alpha": alpha}, medianprops={"color": "k"})
            if metric == "rmssd_ratio":
                ax.axhline(1, color="k", lw=0.6)
            ax.set_xticks(range(len(order)), [labels[f] for f in order], fontsize=6.8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{DEVICE_LABELS[device]}", fontsize=8.5)
            _panel(ax, "abcd"[row * 2 + col])
    handles = [Patch(color="grey", alpha=a, label=lab) for _, lab, a in anchors]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Fiducial point study (development participants, seated rest + standing)", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output_dir, "figA2_fiducial_study")


def figure_method_selection(summary: pd.DataFrame, output_dir: str) -> None:
    methods = ["neurokit2", "scipy", "ensemble", "slope", "template"]
    motions = ["low", "moderate", "severe"]
    fig, axes = plt.subplots(1, 2, figsize=(5.46, 2.5), gridspec_kw={"wspace": 0.08})
    for ax, device, letter in zip(axes, PPG_DEVICES, "ab", strict=False):
        s = summary[summary["device"] == device].pivot_table(index="method", columns="motion_level", values="f1")
        s = s.reindex(index=methods, columns=motions)
        im = ax.imshow(s.to_numpy(), cmap="magma", vmin=0, vmax=100, aspect="auto")
        for i in range(len(methods)):
            for j in range(len(motions)):
                v = s.iloc[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="w" if v < 60 else "k", fontsize=7)
        ax.set_xticks(range(3), motions)
        ax.set_yticks(range(len(methods)), methods if letter == "a" else [""] * len(methods))
        ax.set_title(f"{DEVICE_LABELS[device]}")
        ax.grid(False)
        _panel(ax, letter)
    fig.colorbar(im, ax=axes, label="beat F1 (%)", shrink=0.9)
    save_figure(fig, output_dir, "figA3_method_selection")


def figure_cadence_lock(df: pd.DataFrame, ablation_dir: str, output_dir: str) -> None:
    path = os.path.join(ablation_dir, "no_cadence_gate.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.46, 2.9))
    ppg = df[df["device"].isin(PPG_DEVICES)]
    for i, condition in enumerate(COND):
        v = ppg[ppg["condition"] == condition]["acc_periodicity"].dropna()
        axes[0].scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=6,
                        color=COND_COLORS[condition], alpha=0.6)
    axes[0].axhline(0.30, color="k", lw=0.8, ls="--")
    _cond_ticks(axes[0], range(len(COND)), COND)
    axes[0].set_ylabel("Accelerometer spectral concentration")
    axes[0].set_title("Periodic motion detection")
    _panel(axes[0], "a")
    if os.path.exists(path):
        nc = pd.read_csv(path)
        nc = nc[nc["device"].isin(PPG_DEVICES) & (nc["n_segments"].fillna(0) > 0) & (nc["acc_periodicity"] >= 0.30)].copy()
        nc["cond"] = nc["activity"].map(ACTIVITY_CONDITION)
        cadence = nc["acc_peak_hz"] * 60.0
        nc["r_ppg"], nc["r_ecg"] = nc["mean_hr"] / cadence, nc["ecg_hr"] / cadence
        axes[1].plot([0.3, 2.2], [0.3, 2.2], color=C["grey"], lw=0.7)
        for level in (0.5, 1.0):
            axes[1].axhline(level, color="k", lw=0.5, ls=":")
        for cond, marker in (("locomotion", "o"), ("recovery", "s"), ("rest", "^"), ("adaptation", "^")):
            g = nc[nc["cond"] == cond]
            axes[1].scatter(g["r_ecg"], g["r_ppg"], s=34, marker=marker, color=COND_COLORS[cond],
                            edgecolor="k", lw=0.5, label=CONDITION_LABELS[cond], zorder=3)
        axes[1].text(2.15, 0.52, "0.5 x cadence", fontsize=6.4, ha="right")
        axes[1].text(2.15, 1.02, "1 x cadence", fontsize=6.4, ha="right")
        axes[1].set_xlim(0.3, 2.2)
        axes[1].set_ylim(0.3, 2.2)
        axes[1].set_xlabel("ECG heart rate / accelerometer cadence")
        axes[1].set_ylabel("PPG heart rate / accelerometer cadence")
        axes[1].set_title("Windows passing without the gate")
        axes[1].legend(fontsize=6.0, loc="upper left")
    _panel(axes[1], "b")
    fig.tight_layout()
    save_figure(fig, output_dir, "figA4_cadence_lock")


def figure_participant_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    ppg = df[df["device"].isin(PPG_DEVICES)]
    fig, axes = plt.subplots(1, 2, figsize=(5.46, 5.8), sharey=True)
    for ax, device, letter in zip(axes, PPG_DEVICES, "ab", strict=False):
        p = ppg[ppg["device"] == device].pivot_table(index="participant", columns="activity",
                                                      values="hr_error_vs_ecg", observed=True)
        p = p.reindex(columns=ACTIVITY_ORDER)
        im = ax.imshow(np.log10(p.to_numpy() + 0.1), cmap="RdYlGn_r", vmin=-1, vmax=2, aspect="auto")
        ax.set_xticks(range(len(ACTIVITY_ORDER)), [ACTIVITY_LABELS[a] for a in ACTIVITY_ORDER],
                      rotation=90, fontsize=6.4)
        splits = ppg.groupby("participant")["split"].first()
        ax.set_yticks(range(len(p)), [f"{i}{'*' if splits.get(i) == 'test' else ''}" for i in p.index], fontsize=6.5)
        ax.set_title(DEVICE_LABELS[device])
        ax.grid(False)
        _panel(ax, letter)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, ticks=[-1, 0, 1, 2])
    cbar.ax.set_yticklabels(["0.1", "1", "10", "100"])
    cbar.set_label("|HR error| (bpm), * = test participant")
    save_figure(fig, output_dir, "figA5_participant_heatmap")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_all(csv_path: str, output_dir: str, data_root: str, config_path: str,
              results_dir: str = "paper_results", rebuild_cache: bool = False, jobs: int = 20) -> None:
    apply_style()
    df = load_combined(csv_path)
    tables = os.path.join(results_dir, "tables")
    read = lambda name: pd.read_csv(os.path.join(tables, f"{name}.csv"))  # noqa: E731
    cache = load_beat_cache(data_root, config_path, os.path.join(results_dir, "cache", "beats.pkl"),
                            jobs=jobs, rebuild=rebuild_cache)
    steps = [
        ("fig01", lambda: figure_pipeline(output_dir)),
        ("fig1", lambda: figure_signal_chain(data_root, config_path, output_dir)),
        ("fig2", lambda: figure_clock_sync(df, cache, data_root, config_path, output_dir)),
        ("fig3", lambda: figure_matching_tolerance(cache, output_dir)),
        ("fig4", lambda: figure_cohort_flow(read("cohort_flow"), output_dir)),
        ("fig5", lambda: figure_beat_detection(df, output_dir)),
        ("fig6", lambda: figure_hr_agreement(df, output_dir)),
        ("fig7", lambda: figure_beat_ibi(cache, output_dir)),
        ("fig8", lambda: figure_hrv_forest(read("hrv_agreement"), output_dir)),
        ("fig10", lambda: figure_rmssd_mechanism(
            cache_timing_errors(cache, df), cache,
            pd.read_csv(os.path.join(results_dir, "window_analysis", "window_summary.csv")), output_dir)),
        ("fig10", lambda: figure_ablation(pd.read_csv(os.path.join(results_dir, "ablation_summary.csv")), output_dir)),
        ("fig11", lambda: figure_baselines(pd.read_csv(os.path.join(results_dir, "baselines.csv")), output_dir)),
        ("figS1", lambda: figure_reference_validity(df, output_dir)),
        ("figS2", lambda: figure_fiducial(pd.read_csv(os.path.join(results_dir, "fiducial_study.csv")), output_dir)),
        ("figS3", lambda: figure_method_selection(read("method_selection_summary"), output_dir)),
        ("figS4", lambda: figure_cadence_lock(df, os.path.join(results_dir, "ablation"), output_dir)),
        ("figS5", lambda: figure_participant_heatmap(df, output_dir)),
    ]
    for name, step in steps:
        try:
            step()
        except Exception as exc:  # keep going so one missing input does not block the rest
            logger.error("%s failed: %s", name, exc, exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="GalaxyPPG HRV - manuscript figures")
    parser.add_argument("--csv", default="paper_results/combined_results.csv")
    parser.add_argument("--output-dir", default="fig")
    parser.add_argument("--data-root", default="Dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default="paper_results")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--jobs", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    build_all(args.csv, args.output_dir, args.data_root, args.config, args.results_dir,
              args.rebuild_cache, args.jobs)


if __name__ == "__main__":
    main()
