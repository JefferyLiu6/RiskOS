"""Phase 3: scorecard mechanics, sign convention, and points scaling.

The scaling tests use analytically known values. The behavioural tests use a
small synthetic WOE matrix — permitted as a test fixture and named accordingly,
and in any case not loan data: these check the arithmetic of the scorecard, not
anything about mortgages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from riskos.models.scorecard import Scaling, Scorecard


@pytest.fixture
def fixture_woe() -> tuple[pd.DataFrame, np.ndarray]:
    """Two informative WOE columns and a label they separate cleanly."""
    rng = np.random.default_rng(20250901)
    n = 4_000
    y = rng.binomial(1, 0.08, size=n)
    # Higher WOE means lower risk, so defaulters get lower WOE.
    good = rng.normal(0.6, 0.5, size=n)
    weak = rng.normal(0.2, 0.5, size=n)
    woe = pd.DataFrame(
        {
            "strong_feature": np.where(y == 1, good - 1.6, good),
            "weak_feature": np.where(y == 1, weak - 0.4, weak),
        }
    )
    return woe, y


# --------------------------------------------------------------------------
# Points scaling — hand-computable.
# --------------------------------------------------------------------------


def test_factor_and_offset_match_the_published_formulas() -> None:
    # factor = PDO / ln(2) = 20 / 0.6931471805599453 = 28.85390081777927
    # offset = 600 - factor * ln(50) = 600 - 28.8539 * 3.912023 = 487.1229...
    s = Scaling(pdo=20.0, base_score=600.0, base_odds=50.0)

    assert s.factor == pytest.approx(28.85390081777927)
    assert s.offset == pytest.approx(600 - 28.85390081777927 * np.log(50.0))


def test_pdo_means_what_it_says_doubling_the_odds_moves_the_score_by_pdo() -> None:
    """The defining property of the scaling: PDO points per doubling of odds."""
    s = Scaling(pdo=20.0)
    # score = offset + factor * (-log_odds); halving the odds of default (i.e.
    # doubling good:bad) subtracts ln(2) from log_odds, adding factor*ln(2)=PDO.
    score_at = lambda log_odds: s.offset + s.factor * (-log_odds)  # noqa: E731

    assert score_at(0.0) - score_at(np.log(2.0)) == pytest.approx(20.0)


def test_a_different_pdo_rescales_proportionally() -> None:
    assert Scaling(pdo=40.0).factor == pytest.approx(2 * Scaling(pdo=20.0).factor)


# --------------------------------------------------------------------------
# Sign convention — the conceptual-soundness check.
# --------------------------------------------------------------------------


def test_coefficients_on_woe_features_are_negative(
    fixture_woe: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Higher WOE means lower risk, so every coefficient must be negative.

    A positive one means the multivariate fit reverses the direction the
    feature's own binning established, which produces a bin awarding more points
    for worse credit.
    """
    woe, y = fixture_woe
    card = Scorecard(list(woe.columns))

    fit = card.fit(woe, y)

    assert fit.wrong_sign == []
    assert all(beta < 0 for beta in fit.coefficients.values())


def test_a_reversed_feature_is_dropped_rather_than_kept(
    fixture_woe: tuple[pd.DataFrame, np.ndarray],
) -> None:
    woe, y = fixture_woe
    woe = woe.assign(reversed_feature=-woe["strong_feature"] * 0.9)
    card = Scorecard(list(woe.columns))

    fit = card.fit(woe, y, drop_wrong_sign=True)

    assert "reversed_feature" in fit.dropped_wrong_sign
    assert "reversed_feature" not in card.features
    assert fit.wrong_sign == []


def test_wrong_sign_elimination_can_be_turned_off_for_diagnosis(
    fixture_woe: tuple[pd.DataFrame, np.ndarray],
) -> None:
    woe, y = fixture_woe
    woe = woe.assign(reversed_feature=-woe["strong_feature"] * 0.9)

    fit = Scorecard(list(woe.columns)).fit(woe, y, drop_wrong_sign=False)

    assert fit.wrong_sign
    assert fit.dropped_wrong_sign == []


# --------------------------------------------------------------------------
# Scores and explanations.
# --------------------------------------------------------------------------


def test_higher_score_means_lower_predicted_default(
    fixture_woe: tuple[pd.DataFrame, np.ndarray],
) -> None:
    woe, y = fixture_woe
    card = Scorecard(list(woe.columns))
    card.fit(woe, y)

    scores = card.score(woe)
    probs = card.predict_proba(woe)

    # The relationship is exactly rank-reversing but not linear: score is affine
    # in the log-odds while probability is a sigmoid of them. Spearman is the
    # property that must hold, and it must hold exactly.
    assert spearmanr(scores, probs).statistic == pytest.approx(-1.0)


def test_points_contributions_reconstruct_the_score_exactly(
    fixture_woe: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """The scorecard's explanation is exact, not an approximation.

    This is the property SHAP cannot offer for a GBM, and the reason a scorecard
    can be shown to an adjudicator. Contributions plus the constant reproduce
    the score to floating-point precision.
    """
    woe, y = fixture_woe
    card = Scorecard(list(woe.columns))
    fit = card.fit(woe, y)

    points = card.points_table(woe)
    constant = card.scaling.offset - card.scaling.factor * fit.intercept
    reconstructed = points.sum(axis=1).to_numpy() + constant

    np.testing.assert_allclose(reconstructed, card.score(woe), rtol=1e-9)


def test_principal_drivers_are_ranked_by_absolute_points(
    fixture_woe: tuple[pd.DataFrame, np.ndarray],
) -> None:
    woe, y = fixture_woe
    card = Scorecard(list(woe.columns))
    card.fit(woe, y)

    drivers = card.principal_drivers(woe.head(20), top=2)

    assert len(drivers) == 20
    for row in drivers:
        magnitudes = [abs(d["points"]) for d in row]
        assert magnitudes == sorted(magnitudes, reverse=True)


def test_coefficient_table_flags_the_sign_check(
    fixture_woe: tuple[pd.DataFrame, np.ndarray],
) -> None:
    woe, y = fixture_woe
    card = Scorecard(list(woe.columns))
    card.fit(woe, y)

    table = card.coefficient_table()

    assert table["sign_ok"].all()
    assert (table["points_per_woe_unit"] > 0).all()  # positive points for higher WOE
