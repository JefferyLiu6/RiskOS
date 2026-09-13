"""Phase 4 — the LightGBM challenger (build plan §7.6).

Fitted on **raw** features with native categorical handling, not on the
scorecard's WOE matrix. That is the point of running it: if the challenger were
given the champion's engineered inputs it would only be testing a different
link function, not a different model family.

Both candidates receive the same candidate pool and the same economically
motivated monotone constraints, so the comparison isolates model form rather
than feature preparation. Selection on a small fixed grid, never an unbounded
search: the challenger is not tuned until it wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

from riskos.log import get_logger
from riskos.metrics import auc
from riskos.models.config import LightGBMConfig, models_config

log = get_logger(__name__)


@dataclass
class ChallengerFit:
    features: list[str]
    categorical_features: list[str]
    best_params: dict[str, int]
    best_iteration: int
    n_train: int
    grid_results: list[dict[str, Any]] = field(default_factory=list)
    constraints_applied: dict[str, int] = field(default_factory=dict)
    constraints_ignored: list[str] = field(default_factory=list)
    # Level-to-code mapping captured from the TRAINING frame, so a reloaded
    # bundle can encode a scoring batch without re-deriving levels from it.
    categories: dict[str, list[str]] = field(default_factory=dict)


def to_frame(
    panel: pl.DataFrame,
    features: list[str],
    categories: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Raw features as pandas, with string columns as pandas categoricals.

    LightGBM splits categoricals natively rather than one-hot expanding them,
    which is one of the reasons it was chosen as the challenger. It does that by
    splitting on the integer CODES behind the categorical, so the mapping from
    level to code is part of the model, not a presentation detail.

    ``categories`` supplies that mapping explicitly. Without it pandas derives
    the levels from whatever is in the batch, and a batch missing one level
    shifts the code of every level sorting after it - which looks like the same
    batch-dependence defect as R-003 and R-005.

    It is NOT that defect here, and the distinction is worth stating because the
    obvious reading is wrong. LightGBM records the fitted level NAMES on the
    booster as ``pandas_categorical`` and re-aligns an incoming frame by name
    before predicting, and SHAP's TreeExplainer inherits the same path. Both were
    checked directly on a frame with a level deliberately removed: predictions
    and SHAP values were identical. The in-time validation split really is
    missing a level that training has - 65 Virgin Islands rows, shifting 5.6% of
    its rows by one code - and it changed nothing. Every Phase 4 metric is
    bit-identical before and after this parameter was introduced.

    The mapping is carried anyway, for a reason that is about serving rather
    than about training. A scoring request arriving as JSON has no categorical
    dtype at all, and reconstructing one from a single loan would derive a
    one-level mapping. Relying on the library to rescue that would be relying on
    an implementation detail of a dependency to preserve a property the model
    needs. Making the mapping part of the bundle makes it the model's own.
    """
    frame = panel.select(features).to_pandas()
    for name in frame.columns:
        if categories is not None and name in categories:
            frame[name] = frame[name].astype(
                pd.CategoricalDtype(categories=categories[name], ordered=False)
            )
        elif frame[name].dtype == object:
            frame[name] = frame[name].astype("category")
    return frame


class Challenger:
    """LightGBM binary classifier over a small, fixed, documented grid."""

    def __init__(self, features: list[str], cfg: LightGBMConfig | None = None) -> None:
        self.features = features
        self.cfg = cfg or models_config().lightgbm
        self.model: lgb.LGBMClassifier | None = None
        self._fit: ChallengerFit | None = None
        self.categories: dict[str, list[str]] = {}

    def _constraints(self, frame: pd.DataFrame) -> tuple[list[int], dict[str, int], list[str]]:
        """Constraint vector, plus what was applied and what could not be.

        LightGBM silently ignores a monotone constraint on a categorical
        feature. Rather than trust that never happens, any such request is
        surfaced and reported alongside the fit.
        """
        categorical = {n for n in frame.columns if str(frame[n].dtype) == "category"}
        ignored = [n for n in self.cfg.monotone_constraints if n in categorical]
        applied = {
            n: v
            for n, v in self.cfg.monotone_constraints.items()
            if n in frame.columns and n not in categorical
        }
        if ignored:
            log.warning("monotone_constraint_not_applicable", features=ignored)
        return [applied.get(name, 0) for name in self.features], applied, ignored

    def fit(
        self,
        train: pd.DataFrame,
        y_train: npt.NDArray[Any],
        valid: pd.DataFrame,
        y_valid: npt.NDArray[Any],
    ) -> ChallengerFit:
        """Fit every grid combination, select on in-time validation AUC.

        Selection uses the in-time validation split, never an out-of-time one:
        choosing hyperparameters on the data reserved to judge generalisation
        would make the out-of-time result meaningless.
        """
        self.categories = {
            name: [str(level) for level in train[name].cat.categories]
            for name in train.columns
            if str(train[name].dtype) == "category"
        }
        # The early-stopping set is re-cast onto the training mapping here rather
        # than at the call site, so the stopping metric is computed against the
        # same encoding the model was fitted with regardless of what the caller
        # passed. Verified to be a no-op for LightGBM, which re-aligns by level
        # name; kept because it should not depend on that.
        valid = self._align(valid)
        constraints, applied, ignored = self._constraints(train)
        results: list[dict[str, Any]] = []
        best: tuple[float, lgb.LGBMClassifier, dict[str, int]] | None = None

        for params in self.cfg.combinations():
            # Params come from validated config as dict[str, Any]; LightGBM's
            # stubs type each keyword individually, which **-unpacking cannot
            # satisfy. The values are checked by ModelsConfig, not by mypy here.
            settings: dict[str, Any] = {
                **self.cfg.fixed_params,
                **params,
                "monotone_constraints": constraints,
            }
            model = lgb.LGBMClassifier(**settings)
            model.fit(
                train,
                y_train,
                eval_set=[(valid, y_valid)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False)],
            )
            score = auc(y_valid, np.asarray(model.predict_proba(valid))[:, 1])
            results.append({**params, "valid_auc": score, "best_iteration": model.best_iteration_})
            log.info("grid_candidate", **params, valid_auc=round(score, 5))
            if best is None or score > best[0]:
                best = (score, model, params)

        assert best is not None
        _, self.model, best_params = best
        self._fit = ChallengerFit(
            features=list(self.features),
            categorical_features=[n for n in train.columns if str(train[n].dtype) == "category"],
            best_params=best_params,
            best_iteration=int(self.model.best_iteration_ or 0),
            n_train=len(y_train),
            grid_results=results,
            constraints_applied=applied,
            constraints_ignored=ignored,
            categories=dict(self.categories),
        )
        log.info("challenger_fitted", **best_params, valid_auc=round(best[0], 5))
        return self._fit

    def _align(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Re-cast a pandas frame onto the fitted level-to-code mapping."""
        out = frame.copy()
        for name, levels in self.categories.items():
            if name in out.columns:
                out[name] = out[name].astype(pd.CategoricalDtype(categories=levels, ordered=False))
        return out

    def prepare(self, panel: pl.DataFrame) -> pd.DataFrame:
        """Model-ready frame using the level-to-code mapping fixed at fit time.

        Every caller that scores after fitting must go through here rather than
        calling ``to_frame`` directly, or the encoding becomes a property of the
        batch again.
        """
        assert self.categories or self._fit is not None, "fit or load first"
        return to_frame(panel, self.features, self.categories)

    def predict_proba(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        assert self.model is not None, "fit first"
        proba = np.asarray(self.model.predict_proba(frame[self.features]), dtype=np.float64)
        return np.asarray(proba[:, 1], dtype=np.float64)

    def grid_table(self) -> pl.DataFrame:
        assert self._fit is not None, "fit first"
        return pl.DataFrame(self._fit.grid_results).sort("valid_auc", descending=True)

    def save_bundle(
        self,
        directory: Path,
        *,
        model_id: str = "RISKOS_PD_002",
        version: str = "1.0.0",
        training_window: str = "1999Q1-2006Q4",
        calibrator: Any = None,
        calibration_method: str | None = None,
        shap_background: pd.DataFrame | None = None,
    ) -> Any:
        """Persist everything needed to score and explain a new loan.

        The categorical mapping is a component rather than a note in the
        manifest, because a scoring request arriving as JSON carries no
        categorical dtype and the encoding has to come from somewhere the model
        owns. The SHAP background sample is carried for the same reason the
        scorecard carries its reference points: an attribution relative to a
        baseline is only comparable across batches if the baseline is fixed at
        fit time (R-003).
        """
        from riskos.models.persistence import save_bundle

        assert self._fit is not None, "fit first"
        return save_bundle(
            directory,
            model_id=model_id,
            model_family="lightgbm",
            version=version,
            features=self.features,
            training_window=training_window,
            n_train=self._fit.n_train,
            components={
                "estimator": self.model,
                "categories": self.categories,
                "calibrator": calibrator,
                "shap_background": shap_background,
                "fit": self._fit,
            },
            calibration_method=calibration_method,
        )

    @classmethod
    def from_bundle(cls, directory: Path) -> tuple[Challenger, Any, Any]:
        """Reload a scoring-ready challenger, its calibrator and SHAP baseline."""
        from riskos.models.persistence import load_bundle

        manifest, parts = load_bundle(directory)
        challenger = cls(list(manifest.features))
        challenger.model = parts["estimator"]
        challenger._fit = parts["fit"]
        challenger.categories = parts.get("categories") or {}
        return challenger, parts.get("calibrator"), parts.get("shap_background")
