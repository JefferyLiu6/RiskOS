"""Phase 3 exhibits: the reliability curve and the discrimination/calibration split.

The reliability curve is the centrepiece of the calibration section (build plan
§7.5): predicted PD against observed default rate, with binomial confidence
intervals, per split. A model can sit far from the diagonal while ranking
perfectly, and this is the plot that shows it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes

from riskos.log import get_logger

log = get_logger(__name__)

FIGURES = Path("reports/figures")
SPLIT_STYLE = {
    "train": ("#4C72B0", "o", "train"),
    "validation_in_time": ("#8FA8CE", "s", "in-time validation"),
    "oot_stress": ("#C44E52", "^", "OOT stress (2007-09)"),
    "oot_benign": ("#55A868", "D", "OOT benign (2015-19)"),
    "oot_benign_ex_forbearance": ("#2E6B45", "v", "OOT benign, ex-COVID forbearance"),
}


def _diagonal(ax: Axes, lo: float, hi: float) -> None:
    ax.plot([lo, hi], [lo, hi], color="#888", linestyle="--", linewidth=1, zorder=1)
    ax.text(hi * 0.72, hi * 0.98, "perfect calibration", fontsize=8, color="#777", rotation=41)


def reliability(curves: dict[str, pl.DataFrame], out: Path | None = None) -> Path:
    """Predicted vs observed default rate per split, with Wilson intervals."""
    out = out or FIGURES / "reliability_by_split.png"
    # Log-log. Predicted PDs span roughly 0.01% to 11%, so a linear axis
    # compresses every point below 1% into the corner - which is precisely where
    # most of the portfolio sits and where calibration matters for ECL.
    values = [
        v
        for c in curves.values()
        for arr in (c["observed_rate"].to_numpy(), c["mean_predicted"].to_numpy())
        for v in arr
        if v > 0
    ]
    axis_lo = float(min(values)) * 100 * 0.6
    axis_hi = float(max(values)) * 100 * 1.6

    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    _diagonal(ax, axis_lo, axis_hi)
    for name, curve in curves.items():
        colour, marker, label = SPLIT_STYLE[name]
        x = (curve["mean_predicted"] * 100).to_list()
        y = (curve["observed_rate"] * 100).to_list()
        err_lo = ((curve["observed_rate"] - curve["ci_low"]) * 100).to_list()
        err_hi = ((curve["ci_high"] - curve["observed_rate"]) * 100).to_list()
        ax.errorbar(
            x,
            y,
            yerr=[err_lo, err_hi],
            fmt=marker,
            color=colour,
            label=label,
            markersize=5,
            linewidth=1.2,
            capsize=2,
            alpha=0.9,
            zorder=3,
        )
    ax.set_xlim(axis_lo, axis_hi)
    ax.set_ylim(axis_lo, axis_hi)
    ax.set_xlabel("Mean predicted PD (%)")
    ax.set_ylabel("Observed 12-month default rate (%)")
    ax.set_title(
        "Reliability by split, with 95% Wilson intervals\n"
        "Points above the diagonal mean the model under-predicts default",
        fontsize=10.5,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.6, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("wrote_figure", path=str(out))
    return out


def discrimination_vs_calibration(metrics: pl.DataFrame, out: Path | None = None) -> Path:
    """The project's headline: the two degrade at very different rates.

    Left panel is discrimination (Gini), right is calibration (observed over
    expected). A model can keep most of its ranking power while its predicted
    level becomes badly wrong, and that gap is the whole argument for not
    selecting on AUC.
    """
    out = out or FIGURES / "discrimination_vs_calibration.png"
    order = [s for s in SPLIT_STYLE if s in metrics["split"].to_list()]
    frame = metrics.with_columns(pl.col("split").cast(pl.Enum(order)).alias("_o")).sort("_o")
    labels = [SPLIT_STYLE[s][2] for s in frame["split"]]
    colours = [SPLIT_STYLE[s][0] for s in frame["split"]]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    left.bar(labels, frame["gini"].to_list(), color=colours)
    left.set_ylabel("Gini")
    left.set_title("Discrimination holds up", fontsize=10.5, loc="left")
    left.set_ylim(0, 1)

    ratios = frame["observed_over_expected"].to_list()
    right.bar(labels, ratios, color=colours)
    right.axhline(1.0, color="#333", linestyle="--", linewidth=1)
    right.text(
        len(ratios) - 0.4, 1.06, "perfectly calibrated", fontsize=8, ha="right", color="#333"
    )
    right.set_ylabel("Observed / expected default rate")
    right.set_title("Calibration does not", fontsize=10.5, loc="left")
    for i, value in enumerate(ratios):
        right.text(i, value + 0.06, f"{value:.2f}x", ha="center", fontsize=9)

    for ax in (left, right):
        ax.tick_params(axis="x", labelrotation=12, labelsize=8.5)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("wrote_figure", path=str(out))
    return out
