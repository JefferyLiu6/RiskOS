"""Phase 2 exhibit: default rate by observation quarter.

Build plan §9: every plot is saved as PNG *and* its underlying data as CSV, so
the report regenerates without rerunning models. The CSV is written by the
builder; this module only renders it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes

from riskos.log import get_logger
from riskos.panel.config import panel_config

log = get_logger(__name__)

FIGURES = Path("reports/figures")
SPLIT_COLOURS = {
    "train": "#4C72B0",
    "validation_in_time": "#4C72B0",
    "oot_stress": "#C44E52",
    "oot_benign": "#55A868",
}


def _shade_splits(ax: Axes) -> None:
    """Shade each split window so the regime change is visible, not inferred."""
    for name, (lo, hi) in panel_config().splits.windows().items():
        if name == "validation_in_time":
            continue  # same window as train; shading twice just darkens it
        start, end = int(lo.split("-")[0]), int(hi.split("-")[0]) + 1
        ax.axvspan(start, end, color=SPLIT_COLOURS[name], alpha=0.10, zorder=0)
        ax.text(
            (start + end) / 2,
            ax.get_ylim()[1] * 0.95,
            name.replace("_", " "),
            ha="center",
            va="top",
            fontsize=8,
            color=SPLIT_COLOURS[name],
        )


def default_rate_by_quarter(csv: Path | None = None, out: Path | None = None) -> Path:
    """Render the Phase 2 acceptance exhibit."""
    csv = csv or FIGURES / "default_rate_by_quarter.csv"
    out = out or FIGURES / "default_rate_by_quarter.png"
    df = pl.read_csv(csv).sort("observation_date")
    x = (df["year"] + (df["quarter"] - 1) / 4).to_list()
    rate = (df["default_rate"] * 100).to_list()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_ylim(0, max(rate) * 1.18)
    _shade_splits(ax)
    # Break the line wherever consecutive observations are more than one quarter
    # apart. 2010-2014 belongs to no split, and joining across that gap would
    # draw a five-year trend that was never measured.
    xs: list[float] = []
    ys: list[float] = []
    for i, (xi, yi) in enumerate(zip(x, rate, strict=True)):
        if i and xi - x[i - 1] > 0.3:
            xs.append(float("nan"))
            ys.append(float("nan"))
        xs.append(xi)
        ys.append(yi)
    ax.plot(xs, ys, color="#22223B", linewidth=1.8, zorder=3)
    ax.scatter(x, rate, s=9, color="#22223B", zorder=4)

    crisis = df.with_row_index("i").filter(pl.col("year").is_between(2007, 2009))
    peak = int(crisis["i"][int(crisis["default_rate"].arg_max() or 0)])
    ax.annotate(
        f"crisis peak {df['default_rate'][peak]:.2%}\n{df['year'][peak]}Q{df['quarter'][peak]}",
        xy=(x[peak], rate[peak]),
        xytext=(x[peak] - 5.5, rate[peak] * 0.98),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#666"},
    )
    # The 2019 tail is a COVID payment-forbearance artefact, not an economic
    # default wave: 75% of 2020 90+ DPD loan-months carry the forbearance flag,
    # against 3.9% in 2019. Labelling it on the exhibit stops it being read as
    # a genuine spike.
    last = len(x) - 1
    ax.annotate(
        "2019 windows close in 2020:\nCOVID forbearance, not\neconomic default",
        xy=(x[last], rate[last]),
        xytext=(x[last] - 7.0, rate[last] * 0.80),
        fontsize=8.5,
        color="#8C2F39",
        arrowprops={"arrowstyle": "->", "color": "#8C2F39"},
    )
    ax.set_xlabel("Observation quarter")
    ax.set_ylabel("12-month default rate (%)")
    ax.set_title(
        "12-month default rate by observation quarter\n"
        "Freddie Mac SFLLD sample, U.S. residential mortgages (illustrative)",
        fontsize=11,
        loc="left",
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    log.info("wrote_figure", path=str(out))
    return out
