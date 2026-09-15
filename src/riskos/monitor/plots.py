"""Phase 6 exhibit: the monitoring blind spot, in one picture.

Two measures on two different scales, so two panels sharing a time axis rather
than a dual-axis chart. Top: observed-over-expected default rate, the lagging
calibration indicator that eventually caught 2008. Bottom: score PSI, the leading
drift indicator that stayed below warning during the crisis. Both PD models, the same
windows, no refitting.

Restricted to 1999-2009 deliberately. The 2015-2019 windows are in the CSVs and
in the report, but their calibration is dominated by COVID forbearance (F-001),
which is a different story from the one this exhibit exists to tell.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes

from riskos.log import get_logger

log = get_logger(__name__)

FIGURES = Path("reports/figures")

# Categorical slots 1 and 2 of the validated reference palette, fixed order.
SERIES = {
    "scorecard": ("#2a78d6", "Scorecard (RISKOS_PD_001)"),
    "challenger": ("#eb6834", "LightGBM challenger (RISKOS_PD_002)"),
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
STRESS_SHADE = "#9a9891"
WARN = "#fab219"
BREACH = "#d03b3b"

FIRST_YEAR, LAST_YEAR = 1999, 2009
LINE = {"linewidth": 2, "solid_joinstyle": "round", "solid_capstyle": "round", "zorder": 3}


def _year_axis(labels: pl.Series) -> np.ndarray:
    """'2007Q3' -> 2007.625, the mid-point of the quarter, for a numeric time axis."""
    years = labels.str.slice(0, 4).cast(pl.Int64).to_numpy()
    quarters = labels.str.slice(5, 1).cast(pl.Int64).to_numpy()
    return years + (quarters - 1) / 4.0 + 0.125


def _series(frame: pl.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    kept = frame.filter(pl.col("window").str.slice(0, 4).cast(pl.Int64) <= LAST_YEAR)
    return _year_axis(kept["window"]), kept[column].to_numpy().astype(float)


def _style(ax: Axes, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=1, zorder=0)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
    ax.set_xlim(FIRST_YEAR, LAST_YEAR + 1)
    ax.axvspan(2007, 2010, color=STRESS_SHADE, alpha=0.10, zorder=0, linewidth=0)


def _reference(ax: Axes, y: float, colour: str, label: str) -> None:
    ax.axhline(y, color=colour, linewidth=1, zorder=1)
    ax.text(FIRST_YEAR + 0.08, y, label, color=TEXT_2, fontsize=8, va="bottom", ha="left")


def _end_dot(ax: Axes, x: np.ndarray, y: np.ndarray, colour: str) -> tuple[float, float]:
    ok = ~np.isnan(y)
    ax.plot(
        x[ok][-1],
        y[ok][-1],
        "o",
        color=colour,
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=4,
    )
    return float(x[ok][-1]), float(y[ok][-1])


def _end_labels(ax: Axes, ends: list[tuple[float, float]], fmt: str, merge_within: float) -> None:
    """Label line ends; when two series finish on the same value, say so once.

    Two identical numbers stacked on one point read as a rendering fault, and an
    identical PSI for both families is itself the point of the lower panel.
    """
    if len(ends) == 2 and abs(ends[0][1] - ends[1][1]) < merge_within:
        x, y = ends[0]
        ax.text(x + 0.15, y, f"{fmt.format(y)} both", color=TEXT, fontsize=9, va="center", zorder=5)
        return
    for x, y in ends:
        ax.text(x + 0.15, y, fmt.format(y), color=TEXT, fontsize=9, va="center", zorder=5)


def blind_spot(
    timelines: dict[str, tuple[pl.DataFrame, pl.DataFrame]], out: Path | None = None
) -> Path:
    """Calibration deterioration above score drift that stays low during the crisis.

    ``timelines`` maps a series label to its (drift, performance) frames as
    written by ``riskos monitor``.
    """
    out = out or FIGURES / "monitoring_blind_spot.png"
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7.6), sharex=True, gridspec_kw={"height_ratios": [1.45, 1]}
    )
    fig.patch.set_facecolor(SURFACE)

    _style(top, "Observed / expected default rate")
    _style(bottom, "Score PSI vs development sample")
    _reference(top, 1.25, WARN, "warn above 1.25")
    _reference(top, 2.0, BREACH, "breach above 2.0")
    _reference(bottom, 0.10, WARN, "warn above 0.10")
    _reference(bottom, 0.25, BREACH, "breach above 0.25")
    top.axhline(1.0, color=GRID, linewidth=1, zorder=1)

    top_ends: list[tuple[float, float]] = []
    bottom_ends: list[tuple[float, float]] = []
    peak_psi = (0.0, 0.0)
    for label, (drift, perf) in timelines.items():
        colour, name = SERIES[label]
        x, oe = _series(perf, "observed_over_expected")
        top.plot(x, oe, color=colour, label=name, **LINE)
        top_ends.append(_end_dot(top, x, oe, colour))
        xp, psi = _series(drift, "score_psi")
        bottom.plot(xp, psi, color=colour, label=name, **LINE)
        bottom_ends.append(_end_dot(bottom, xp, psi, colour))
        i = int(np.nanargmax(psi))
        if psi[i] > peak_psi[1]:
            peak_psi = (float(xp[i]), float(psi[i]))
    _end_labels(top, top_ends, "{:.2f}", merge_within=0.05)
    _end_labels(bottom, bottom_ends, "{:.3f}", merge_within=0.005)

    top.set_ylim(0, 5.2)
    bottom.set_ylim(0, 0.30)
    bottom.set_xticks(range(FIRST_YEAR, LAST_YEAR + 1))
    bottom.set_xticklabels([str(y) for y in range(FIRST_YEAR, LAST_YEAR + 1)])
    top.text(
        2008.5,
        5.05,
        "2007-09: out-of-time, the crisis",
        color=TEXT_2,
        fontsize=8.5,
        ha="center",
        va="top",
    )
    top.text(2003.0, 5.05, "Fitted on 1999-2006", color=TEXT_2, fontsize=8.5, ha="center", va="top")
    top.legend(
        loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False, fontsize=9, labelcolor=TEXT
    )
    # The largest drift reading of the whole run is inside the training window.
    bottom.annotate(
        f"highest PSI shown: {peak_psi[1]:.2f} (training period)",
        xy=peak_psi,
        xytext=(peak_psi[0] + 0.6, 0.235),
        color=TEXT_2,
        fontsize=8.5,
        ha="left",
        arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 1},
    )

    fig.suptitle(
        "Crisis defaults rose. Score drift stayed below warning.",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=TEXT,
        y=0.985,
    )
    fig.text(
        0.06,
        0.935,
        "Both models fitted on 1999-2006 and evaluated without refitting. Shading marks the 2007-09 crisis.\n"
        "Observed / expected defaults rise sharply; score PSI stays below 0.10 during the crisis.\n"
        "PSI measures changes in score distributions; it does not measure prediction accuracy.",
        fontsize=9,
        color=TEXT_2,
        ha="left",
        va="top",
        linespacing=1.5,
    )
    bottom.set_xlabel(
        "Observation quarter (default outcomes become available 12 months later)",
        color=TEXT_2,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.855))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    log.info("wrote_blind_spot_exhibit", path=str(out))
    return out


def from_artefacts(figures: Path = FIGURES) -> Path:
    """Render from the CSVs the monitoring run wrote, without rerunning it."""
    timelines = {
        label: (
            pl.read_csv(figures / f"monitoring_{label}_drift_timeline.csv"),
            pl.read_csv(figures / f"monitoring_{label}_performance_timeline.csv"),
        )
        for label in SERIES
        if (figures / f"monitoring_{label}_drift_timeline.csv").exists()
    }
    return blind_spot(timelines)
