"""Phase 3 — optimal binning, Weight of Evidence, and Information Value.

Credit-risk vocabulary rather than generic ML (build plan §5). A scorecard bins
each feature, replaces the bin with its Weight of Evidence, and fits a logistic
regression on those. That buys three things a raw GBM does not give:

* **Monotonicity where the economics demand it.** Higher LTV must not produce
  lower risk. optbinning enforces this as a constraint on the fit rather than
  leaving it to chance.
* **Missing handled as a bin, not imputed.** 8% of loans have no DTI. That
  absence is informative and gets its own WOE (build plan rule 6).
* **A per-feature contribution for free.** Points per bin sum to a score, so
  every decision decomposes without a post-hoc explainer.

    WOE_i = ln( (good_i / good_total) / (bad_i / bad_total) )
    IV    = sum over bins of (good_i/good_total - bad_i/bad_total) * WOE_i

Sign convention: "good" is a non-default. A **higher WOE means lower risk**, so
a fitted logistic coefficient on WOE features is expected to be negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from optbinning import BinningProcess

from riskos.log import get_logger
from riskos.panel.config import BinningConfig, feature_config

log = get_logger(__name__)

IV_BANDS = ((0.02, "unpredictive"), (0.10, "weak"), (0.30, "medium"), (0.50, "strong"))


@dataclass(frozen=True)
class FeatureIV:
    name: str
    iv: float
    band: str
    n_bins: int
    selected: bool
    reason: str


def classify_iv(iv: float) -> str:
    """Conventional Information Value reading."""
    for threshold, label in IV_BANDS:
        if iv < threshold:
            return label
    return "suspiciously_strong"


def _dtype_split(frame: pl.DataFrame, names: list[str]) -> tuple[list[str], list[str]]:
    numeric = [n for n in names if frame.schema[n].is_numeric()]
    return numeric, [n for n in names if n not in numeric]


def candidate_features(frame: pl.DataFrame, cfg: BinningConfig | None = None) -> list[str]:
    """Allow-listed features, minus the high-cardinality identifiers."""
    cfg = cfg or feature_config().binning
    excluded = set(cfg.exclude_high_cardinality) | set(cfg.duplicate_drops)
    return [f for f in feature_config().allowed if f in frame.columns and f not in excluded]


def _fit_params(names: list[str], cfg: BinningConfig) -> dict[str, dict[str, object]]:
    """Per-feature optbinning overrides: monotone trend and minimum bin size."""
    params: dict[str, dict[str, object]] = {}
    for name, trend in cfg.monotonic_trend.items():
        if name in names:
            params.setdefault(name, {})["monotonic_trend"] = trend
    for name, size in cfg.min_bin_size_overrides.items():
        if name in names:
            params.setdefault(name, {})["min_bin_size"] = size
    return params


def build_process(names: list[str], frame: pl.DataFrame, cfg: BinningConfig) -> BinningProcess:
    """Configure a BinningProcess with the declared monotone constraints."""
    numeric, categorical = _dtype_split(frame, names)
    log.info("binning_features", numeric=len(numeric), categorical=len(categorical))
    return BinningProcess(
        variable_names=names,
        categorical_variables=categorical,
        max_n_bins=cfg.max_n_bins,
        min_n_bins=cfg.min_n_bins,
        min_bin_size=cfg.min_bin_size,
        min_prebin_size=cfg.min_prebin_size,
        binning_fit_params=_fit_params(names, cfg),
    )


def fit(
    frame: pl.DataFrame, target: str = "default_12m", cfg: BinningConfig | None = None
) -> tuple[BinningProcess, list[str]]:
    """Fit binning on the training split only.

    Bins derived from anything but the training split would let the evaluation
    periods inform the feature definitions, which is leakage of a subtler kind
    than a forbidden column.
    """
    cfg = cfg or feature_config().binning
    names = candidate_features(frame, cfg)
    process = build_process(names, frame, cfg)
    x = frame.select(names).to_pandas()
    y = frame[target].to_numpy()
    process.fit(x, y)
    log.info("binning_fitted", features=len(names), rows=len(x))
    return process, names


def information_values(
    process: BinningProcess, names: list[str], cfg: BinningConfig | None = None
) -> list[FeatureIV]:
    """Score every feature by IV and apply the configured floor and ceiling."""
    cfg = cfg or feature_config().binning
    out: list[FeatureIV] = []
    for name in names:
        table = process.get_binned_variable(name).binning_table.build()
        iv = float(table.loc["Totals", "IV"])
        n_bins = int(len(table) - 3)  # drop Special, Missing, Totals rows
        if iv < cfg.iv_floor:
            selected, reason = False, f"IV {iv:.4f} below floor {cfg.iv_floor}"
        elif iv > cfg.iv_leakage_ceiling and name not in cfg.cleared_high_iv:
            selected, reason = (
                False,
                f"IV {iv:.4f} above leakage ceiling {cfg.iv_leakage_ceiling} and not "
                "cleared in conf/features.yaml iv_cleared",
            )
        elif iv > cfg.iv_leakage_ceiling:
            selected, reason = True, f"IV {iv:.4f} above ceiling; cleared on documented evidence"
        else:
            selected, reason = True, "selected"
        out.append(FeatureIV(name, iv, classify_iv(iv), n_bins, selected, reason))
    return sorted(out, key=lambda f: -f.iv)


def selected_names(ivs: list[FeatureIV]) -> list[str]:
    return [f.name for f in ivs if f.selected]


def transform(
    process: BinningProcess, frame: pl.DataFrame, names: list[str]
) -> tuple[pd.DataFrame, np.ndarray[Any, Any]]:
    """WOE-transform a frame using bins fitted on training.

    `names` selects the output columns. The process is handed every variable it
    was fitted on, because optbinning requires its full fitted set to be present
    in the input; subsetting happens after.
    """
    fitted = list(process.variable_names)
    woe = process.transform(frame.select(fitted).to_pandas(), metric="woe")
    return woe[names], frame["default_12m"].to_numpy()


def binning_table(process: BinningProcess, name: str) -> pl.DataFrame:
    """Per-bin counts, event rate, WOE and IV for one feature — the audit trail.

    The optbinning table carries a trailing "Totals" row, so several columns mix
    numbers with that label. Object columns are stringified rather than coerced,
    which would drop the totals row.
    """
    table = process.get_binned_variable(name).binning_table.build().reset_index()
    table.columns = [str(c) for c in table.columns]
    for column in table.columns:
        if table[column].dtype == object:
            table[column] = table[column].astype(str)
    return pl.from_pandas(table)


def monotonicity_report(process: BinningProcess, cfg: BinningConfig | None = None) -> pl.DataFrame:
    """Verify each declared constraint actually holds in the fitted bins.

    optbinning is asked for monotonicity; this checks it was delivered. A
    constraint that silently failed to bind would put a non-monotone feature on
    a scorecard that claims to be monotone.
    """
    cfg = cfg or feature_config().binning
    rows = []
    fitted = set(process.variable_names)
    for name, trend in cfg.monotonic_trend.items():
        if name not in fitted:  # dropped as a duplicate before fitting
            continue
        table = process.get_binned_variable(name).binning_table.build()
        rates = table.iloc[:-3]["Event rate"].to_numpy(dtype=float)
        finite = rates[np.isfinite(rates)]
        diffs = np.diff(finite)
        holds = bool(
            (diffs >= -1e-12).all()
            if trend == "ascending"
            else (diffs <= 1e-12).all()
            if trend == "descending"
            else True
        )
        rows.append(
            {"feature": name, "declared_trend": trend, "n_bins": len(finite), "holds": holds}
        )
    return pl.DataFrame(rows)


def save_binning_tables(process: BinningProcess, names: list[str], out_dir: Path) -> None:
    """Persist every binning table as CSV so the report needs no refit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        binning_table(process, name).write_csv(out_dir / f"bins_{name}.csv")
    log.info("wrote_binning_tables", count=len(names), path=str(out_dir))


def prune_correlated(
    woe: pd.DataFrame, ivs: list[FeatureIV], cfg: BinningConfig | None = None
) -> tuple[list[str], list[dict[str, object]]]:
    """Drop the lower-IV member of any pair correlated above the ceiling.

    Applied to the WOE matrix, which is what the regression actually sees, not
    to the raw columns. Collinearity is far more damaging to a scorecard than to
    a GBM: it inflates coefficient standard errors and destabilises the points,
    destroying the per-feature contribution that is the scorecard's whole
    purpose.
    """
    cfg = cfg or feature_config().binning
    ranked = [f.name for f in sorted(ivs, key=lambda f: -f.iv) if f.name in woe.columns]
    corr = woe[ranked].corr().abs()

    kept: list[str] = []
    dropped: list[dict[str, object]] = []
    for name in ranked:  # highest IV first, so the survivor is the stronger one
        clash = next((k for k in kept if corr.loc[name, k] > cfg.max_correlation), None)
        if clash is None:
            kept.append(name)
        else:
            dropped.append(
                {"dropped": name, "kept": clash, "correlation": round(corr.loc[name, clash], 4)}
            )
            log.info("pruned_collinear", dropped=name, kept=clash)
    return kept, dropped
