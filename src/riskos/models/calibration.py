"""Phase 4 — probability calibration (build plan §7.6).

Two methods, both fitted on the **in-time validation** split:

* **Platt** — a one-parameter logistic recalibration of the log-odds. Rigid,
  monotone, and cannot invent structure it did not see; it can only shift and
  stretch the odds scale.
* **Isotonic** — a free monotone step function. Strictly more flexible, and
  therefore strictly more able to overfit the split it was fitted on.

Why in-time validation and nowhere else. Training is wrong because the model is
already optimal there, so the fitted mapping collapses toward the identity and
buys nothing. An out-of-time split is worse: it is the data reserved to judge
generalisation, and fitting a correction on it converts an honest test into a
fitted result.

**What calibration can and cannot do.** Both methods are monotone, so neither
can *reverse* an ordering: a pair ranked correctly before is ranked correctly
after. They act on the level, not the ranking.

The invariance is not quite exact for both, and the difference is worth stating
precisely. Platt is strictly monotone, so AUC is preserved to machine precision.
Isotonic is only weakly monotone: its flat segments collapse distinct raw scores
onto one calibrated value, creating ties where none existed, and because ties
count a half in AUC the metric shifts slightly. Small, but not zero, and not the
sign of a bug.

Either way, calibration is the right instrument for a model that ranks well and
predicts the wrong magnitude — and useless against a level shift the fitting
period never contained, which is why Phase 5's macro overlay exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from riskos.log import get_logger

log = get_logger(__name__)

Method = Literal["isotonic", "platt", "uncalibrated"]
_EPS = 1e-9  # keeps log-odds finite at p = 0 or 1


@dataclass(frozen=True)
class CalibrationFit:
    method: Method
    n_fit: int
    fit_on: str


class Calibrator:
    """Monotone probability recalibration. Preserves ranking by construction."""

    def __init__(self, method: Method, fit_on: str = "validation_in_time") -> None:
        if method not in ("isotonic", "platt", "uncalibrated"):
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self.fit_on = fit_on
        self._model: Any = None
        self._fit: CalibrationFit | None = None

    @staticmethod
    def _log_odds(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        clipped = np.clip(p, _EPS, 1.0 - _EPS)
        return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)

    def fit(self, p: npt.NDArray[np.float64], y: npt.NDArray[Any]) -> CalibrationFit:
        if self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._model.fit(p, y)
        elif self.method == "platt":
            # Fitted on the log-odds, not on the probability: Platt scaling is a
            # logistic regression in the score space, which is what makes it a
            # one-parameter shift-and-stretch of the odds rather than an
            # arbitrary curve.
            self._model = LogisticRegression(C=1e10, max_iter=1000)
            self._model.fit(self._log_odds(p).reshape(-1, 1), y)
        self._fit = CalibrationFit(self.method, len(y), self.fit_on)
        log.info("calibrator_fitted", method=self.method, rows=len(y), fit_on=self.fit_on)
        return self._fit

    def transform(self, p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.method == "uncalibrated":
            return p
        assert self._model is not None, "fit first"
        if self.method == "isotonic":
            return np.asarray(self._model.predict(p), dtype=np.float64)
        return np.asarray(
            self._model.predict_proba(self._log_odds(p).reshape(-1, 1))[:, 1], dtype=np.float64
        )


def fit_all(
    p_valid: npt.NDArray[np.float64], y_valid: npt.NDArray[Any], methods: tuple[str, ...]
) -> dict[str, Calibrator]:
    """Fit every configured method plus the uncalibrated pass-through."""
    out: dict[str, Calibrator] = {"uncalibrated": Calibrator("uncalibrated")}
    for method in methods:
        cal = Calibrator(method)  # type: ignore[arg-type]
        cal.fit(p_valid, y_valid)
        out[method] = cal
    return out
