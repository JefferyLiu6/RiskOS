"""Phase 3 — the WOE scorecard, champion candidate (build plan §7.6).

Logistic regression on WOE-transformed features, expressed as a points-based
scorecard with documented PDO and base-odds scaling.

    log(odds) = beta_0 + sum_j beta_j * WOE_j(x_j)

    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)
    score  = offset + factor * (-log_odds)

**Sign convention, stated once and asserted in tests.** WOE is built with "good"
= non-default, so a higher WOE means lower risk. Every fitted coefficient is
therefore expected to be NEGATIVE. A positive one means the feature's fitted
direction contradicts its own binning, which is a conceptual-soundness failure
worth failing the build over, not a curiosity.

Scores are oriented so that **higher is safer**, the industry convention.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

from riskos.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Scaling:
    """Points scaling. Cosmetic for ranking, but it is what a human reads."""

    pdo: float = 20.0  # points to double the odds
    base_score: float = 600.0
    base_odds: float = 50.0  # 50:1 good:bad at base_score

    @property
    def factor(self) -> float:
        return float(self.pdo / np.log(2.0))

    @property
    def offset(self) -> float:
        return float(self.base_score - self.factor * np.log(self.base_odds))


@dataclass
class ScorecardFit:
    features: list[str]
    coefficients: dict[str, float]
    intercept: float
    scaling: dict[str, float]
    n_train: int
    train_default_rate: float
    wrong_sign: list[str]
    dropped_wrong_sign: list[str]


class Scorecard:
    """Logistic regression on WOE features, with points scaling."""

    def __init__(self, features: list[str], scaling: Scaling | None = None) -> None:
        self.features = features
        self.scaling = scaling or Scaling()
        # No penalty: features are already screened by IV and pruned for
        # collinearity, and an unpenalised fit keeps the coefficients directly
        # interpretable as log-odds per unit of WOE.
        # C=inf is an unpenalised fit. Features are already screened by IV and
        # pruned for collinearity, and no penalty keeps the coefficients
        # directly interpretable as log-odds per unit of WOE.
        self.model = LogisticRegression(C=np.inf, max_iter=1000, solver="lbfgs")
        self._fit: ScorecardFit | None = None
        self.dropped_wrong_sign: list[str] = []
        # Mean points contribution per feature on the TRAINING population.
        # Explanations are centred on this, never on the batch being scored.
        self._reference_points: pd.Series | None = None

    def fit(
        self, woe: pd.DataFrame, y: npt.NDArray[Any], drop_wrong_sign: bool = True
    ) -> ScorecardFit:
        """Fit, optionally eliminating wrong-sign features iteratively.

        A positive coefficient means the multivariate fit reverses the direction
        the feature's own binning established univariately. On a scorecard that
        is not a curiosity to note and move past: it produces a bin that awards
        *more* points for *worse* credit, which cannot be defended to an
        adjudicator or a validator.

        Removing them one at a time, weakest first, is standard scorecard
        practice - the reversal is usually residual collinearity, and dropping
        the worst offender often restores the others.
        """
        dropped: list[str] = []
        while True:
            self.model.fit(woe[self.features], y)
            coefs = dict(zip(self.features, self.model.coef_[0], strict=True))
            wrong = [name for name, beta in coefs.items() if beta > 0]
            if not wrong or not drop_wrong_sign:
                break
            worst = max(wrong, key=lambda name: coefs[name])
            log.warning(
                "dropped_wrong_sign_feature", feature=worst, coefficient=round(coefs[worst], 6)
            )
            dropped.append(worst)
            self.features = [f for f in self.features if f != worst]
        self.dropped_wrong_sign = dropped
        self._fit = ScorecardFit(
            features=list(self.features),
            coefficients={k: float(v) for k, v in coefs.items()},
            intercept=float(self.model.intercept_[0]),
            scaling=asdict(self.scaling),
            n_train=len(y),
            train_default_rate=float(np.mean(y)),
            wrong_sign=wrong,
            dropped_wrong_sign=dropped,
        )
        self._reference_points = self.points_table(woe).mean(axis=0)
        log.info(
            "scorecard_fitted", features=len(self.features), rows=len(y), wrong_sign=len(wrong)
        )
        return self._fit

    def predict_proba(self, woe: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Predicted probability of default."""
        return np.asarray(self.model.predict_proba(woe[self.features])[:, 1], dtype=np.float64)

    def log_odds(self, woe: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray(self.model.decision_function(woe[self.features]), dtype=np.float64)

    def score(self, woe: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Points score. Higher is safer."""
        s = self.scaling
        return np.asarray(s.offset + s.factor * (-self.log_odds(woe)), dtype=np.float64)

    def points_table(self, woe: pd.DataFrame) -> pd.DataFrame:
        """Per-feature points contribution for each row.

        This is the scorecard's explanation mechanism: contributions sum to the
        score, exactly, with no post-hoc approximation. Compare with SHAP in
        Phase 4.
        """
        assert self._fit is not None, "fit first"
        factor = self.scaling.factor
        return pd.DataFrame(
            {
                name: -factor * self._fit.coefficients[name] * woe[name].to_numpy()
                for name in self.features
            },
            index=woe.index,
        )

    def principal_drivers(self, woe: pd.DataFrame, top: int = 5) -> list[list[dict[str, float]]]:
        """The `top` features moving each row furthest from the training average.

        Centred on the training-population mean captured at fit time, NOT on the
        batch being scored. Centring on the batch makes an explanation a
        function of who else happened to be scored alongside: a borrower scored
        alone would report every driver as exactly 0.0, and the same borrower
        would get a different explanation in a different batch. An explanation
        shown to an adjudicator has to be a property of the borrower.
        """
        assert self._reference_points is not None, "fit first"
        points = self.points_table(woe)
        centred = points - self._reference_points
        out: list[list[dict[str, float]]] = []
        for _, row in centred.iterrows():
            ranked = row.reindex(row.abs().sort_values(ascending=False).index)[:top]
            out.append([{"feature": k, "points": float(v)} for k, v in ranked.items()])
        return out

    def coefficient_table(self) -> pl.DataFrame:
        """Fitted coefficients with their expected-sign check."""
        assert self._fit is not None, "fit first"
        return pl.DataFrame(
            [
                {
                    "feature": name,
                    "coefficient": beta,
                    "expected_sign": "negative",
                    "sign_ok": beta < 0,
                    "points_per_woe_unit": -self.scaling.factor * beta,
                }
                for name, beta in self._fit.coefficients.items()
            ]
        ).sort("coefficient")

    def save(self, path: Path) -> None:
        """Write the human-readable fit record.

        Metadata only: coefficients, scaling and the sign check. This is what a
        reviewer reads, NOT what a scoring service loads. Use `save_bundle` for
        a runnable artefact - a scorecard cannot be reconstructed from
        coefficients alone, because the WOE values they multiply are defined by
        bins learnt on the training split.
        """
        assert self._fit is not None, "fit first"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self._fit), indent=2), encoding="utf-8")
        log.info("wrote_scorecard_metadata", path=str(path))

    def save_bundle(
        self,
        directory: Path,
        binning_process: Any,
        *,
        model_id: str = "RISKOS_PD_001",
        version: str = "1.0.0",
        training_window: str = "1999Q1-2006Q4",
        calibrator: Any = None,
        calibration_method: str | None = None,
    ) -> Any:
        """Persist everything needed to score a new loan."""
        from riskos.models.persistence import save_bundle

        assert self._fit is not None, "fit first"
        return save_bundle(
            directory,
            model_id=model_id,
            model_family="woe_scorecard",
            version=version,
            features=self.features,
            training_window=training_window,
            n_train=self._fit.n_train,
            components={
                "binning": binning_process,
                "estimator": self.model,
                "calibrator": calibrator,
                "scaling": self.scaling,
                "reference_points": self._reference_points,
                "fit": self._fit,
            },
            calibration_method=calibration_method,
        )

    @classmethod
    def from_bundle(cls, directory: Path) -> tuple[Scorecard, Any, Any]:
        """Reload a scoring-ready scorecard, its binning process and calibrator.

        The explanation baseline is restored with the model, so principal
        drivers remain a property of the borrower rather than of the batch
        (defect R-003).
        """
        from riskos.models.persistence import load_bundle

        manifest, parts = load_bundle(directory)
        card = cls(list(manifest.features), scaling=parts["scaling"])
        card.model = parts["estimator"]
        card._fit = parts["fit"]
        card._reference_points = parts.get("reference_points")
        return card, parts.get("binning"), parts.get("calibrator")
