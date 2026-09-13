"""AUC, Gini, KS, lift — each against a value worked out by hand."""

from __future__ import annotations

import pytest

from riskos.metrics._common import DegenerateInputError
from riskos.metrics.discrimination import auc, gini, ks_statistic, lift_table


def test_auc_matches_hand_counted_concordant_pairs() -> None:
    # positives score {0.35, 0.8}, negatives {0.1, 0.4}. Of the four pairs,
    # three are concordant (0.35>0.1, 0.8>0.1, 0.8>0.4) and one is not
    # (0.35 < 0.4), so AUC = 3/4.
    assert auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_gini_is_twice_auc_minus_one() -> None:
    assert gini([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.5)


def test_perfect_and_inverted_separation() -> None:
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert gini([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    # Reversing the score reverses the ordering entirely.
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_all_scores_tied_gives_no_discrimination() -> None:
    # Every pair is a tie, each counting a half: AUC = 0.5, Gini = 0.
    assert auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert gini([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_ks_on_alternating_labels() -> None:
    # Sorted ascending the labels alternate n,p,n,p,n,p with 3 of each.
    # Cumulative negative share runs 1/3, 1/3, 2/3, 2/3, 1, 1 and positive
    # 0, 1/3, 1/3, 2/3, 2/3, 1, so the gap is 1/3 at every odd position.
    result = ks_statistic([0, 1, 0, 1, 0, 1], [1, 2, 3, 4, 5, 6])

    assert result.statistic == pytest.approx(1 / 3)
    assert result.n_positive == 3
    assert result.n_negative == 3


def test_ks_is_one_under_perfect_separation_and_reports_its_location() -> None:
    result = ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    assert result.statistic == pytest.approx(1.0)
    assert result.threshold == pytest.approx(0.2)
    # The threshold sits at the 2nd of 4 observations ranked ascending, i.e.
    # 75% of the way down from the riskiest, which is risk decile 8.
    assert result.decile == 8


def test_ks_does_not_split_a_tie_group() -> None:
    # A threshold inside a run of equal scores is not attainable. Counting it
    # would report KS = 0.5 here; the correct answer is 0.
    result = ks_statistic([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])

    assert result.statistic == pytest.approx(0.0)


def test_single_class_is_rejected_rather_than_scored() -> None:
    with pytest.raises(DegenerateInputError, match="undefined"):
        auc([1, 1, 1], [0.2, 0.5, 0.9])


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        auc([0, 1], [0.1, 0.2, 0.3])


def test_lift_table_band_one_is_riskiest_and_capture_reaches_one() -> None:
    # 10 loans, 2 defaults, both with the highest scores.
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    scores = [0.99, 0.98, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01]

    table = lift_table(y, scores, n_bands=5).sort("band")

    assert table["n"].to_list() == [2, 2, 2, 2, 2]
    assert table["n_default"].to_list() == [2, 0, 0, 0, 0]
    # Band 1 holds a 100% default rate against a 20% portfolio rate: lift 5x.
    assert table["lift"][0] == pytest.approx(5.0)
    assert table["cum_capture_rate"][0] == pytest.approx(1.0)
    assert table["cum_capture_rate"][-1] == pytest.approx(1.0)


def test_lift_is_flat_when_the_score_is_uninformative() -> None:
    y = [1, 0, 1, 0, 1, 0, 1, 0]
    scores = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    table = lift_table(y, scores, n_bands=4)

    assert table["lift"].to_list() == pytest.approx([1.0] * 4)
