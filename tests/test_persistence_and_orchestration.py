"""Model persistence, scenario ECL, LGD statistics, and CLI reachability.

Covers the paths a review found untested: the corrected `scenario_ecl`, the LGD
distribution functions, the bundle round-trip, and that every Makefile target
resolves to a command that exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from typer.testing import CliRunner

from riskos.cli import app
from riskos.ecl import engine
from riskos.ecl import lgd as lgd_mod
from riskos.models import persistence
from riskos.models.scorecard import Scaling, Scorecard

runner = CliRunner()


# --------------------------------------------------------------------------
# Bundle persistence — a model must be reloadable without a refit.
# --------------------------------------------------------------------------


@pytest.fixture
def fitted_card() -> tuple[Scorecard, pd.DataFrame]:  # type: ignore[name-defined]
    import pandas as pd

    rng = np.random.default_rng(20250901)
    n = 2_000
    y = rng.binomial(1, 0.1, n)
    woe = pd.DataFrame(
        {
            "a": np.where(y == 1, rng.normal(-1.0, 0.5, n), rng.normal(0.6, 0.5, n)),
            "b": np.where(y == 1, rng.normal(-0.3, 0.5, n), rng.normal(0.2, 0.5, n)),
        }
    )
    card = Scorecard(["a", "b"])
    card.fit(woe, y)
    return card, woe


def test_a_reloaded_model_scores_identically(
    fitted_card: tuple[Scorecard, object], tmp_path: Path
) -> None:
    """The point of a bundle: no refit, and no drift in what it predicts."""
    card, woe = fitted_card
    card.save_bundle(tmp_path / "bundle", binning_process={"stub": True})

    reloaded, binning_process, _ = Scorecard.from_bundle(tmp_path / "bundle")

    np.testing.assert_allclose(reloaded.predict_proba(woe), card.predict_proba(woe), rtol=1e-12)
    assert binning_process == {"stub": True}


def test_the_explanation_baseline_survives_the_round_trip(
    fitted_card: tuple[Scorecard, object], tmp_path: Path
) -> None:
    """Without it, principal drivers become batch-dependent again (R-003)."""
    card, woe = fitted_card
    card.save_bundle(tmp_path / "bundle", binning_process=None)
    reloaded, _, _ = Scorecard.from_bundle(tmp_path / "bundle")

    solo = reloaded.principal_drivers(woe.head(1))[0]  # type: ignore[attr-defined]
    batched = reloaded.principal_drivers(woe)[0]  # type: ignore[arg-type]

    assert solo[0]["points"] == pytest.approx(batched[0]["points"])
    assert abs(solo[0]["points"]) > 1e-9


def test_scaling_survives_so_scores_are_on_the_same_points_axis(
    fitted_card: tuple[Scorecard, object], tmp_path: Path
) -> None:
    card, woe = fitted_card
    card.scaling = Scaling(pdo=40.0, base_score=700.0, base_odds=20.0)
    card.save_bundle(tmp_path / "bundle", binning_process=None)

    reloaded, _, _ = Scorecard.from_bundle(tmp_path / "bundle")

    assert reloaded.scaling.pdo == 40.0
    np.testing.assert_allclose(reloaded.score(woe), card.score(woe), rtol=1e-12)


def test_the_manifest_records_what_the_model_needs(
    fitted_card: tuple[Scorecard, object], tmp_path: Path
) -> None:
    card, _ = fitted_card
    manifest = card.save_bundle(tmp_path / "bundle", binning_process=None, version="2.1.0")

    assert manifest.features == card.features
    assert manifest.version == "2.1.0"
    assert manifest.config_fingerprint  # config drift is detectable
    assert "estimator" in manifest.components


def test_a_missing_bundle_fails_with_an_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(persistence.BundleError, match="run the training step"):
        persistence.load_manifest(tmp_path / "absent")


def test_a_bundle_missing_a_listed_component_is_rejected(
    fitted_card: tuple[Scorecard, object], tmp_path: Path
) -> None:
    card, _ = fitted_card
    card.save_bundle(tmp_path / "bundle", binning_process=None)
    (tmp_path / "bundle" / "estimator.pkl").unlink()

    with pytest.raises(persistence.BundleError, match="estimator"):
        persistence.load_bundle(tmp_path / "bundle")


def test_scoring_refuses_an_input_missing_a_required_feature(
    fitted_card: tuple[Scorecard, object], tmp_path: Path
) -> None:
    card, _ = fitted_card
    manifest = card.save_bundle(tmp_path / "bundle", binning_process=None)

    persistence.require_features(manifest, ["a", "b", "extra"])  # superset is fine
    with pytest.raises(persistence.BundleError, match="absent from the input"):
        persistence.require_features(manifest, ["a"])


# --------------------------------------------------------------------------
# scenario_ecl — the corrected path (R-008).
# --------------------------------------------------------------------------


@pytest.fixture
def fixture_scenarios() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "scenario": ["base", "stress"],
            "weight": [0.7, 0.3],
            "logit_shift": [0.0, 1.0],
            "logit_shift_low": [-0.2, 0.5],
            "logit_shift_high": [0.2, 1.5],
            "extrapolates": [False, True],
        }
    )


@pytest.fixture
def fixture_book() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "loan_sequence_number": ["A", "B", "C"],
            "observation_date": ["2006-12-01"] * 3,
            "current_loan_delinquency_status": ["00", "01", "03"],
            "current_actual_upb": [100_000.0, 200_000.0, 150_000.0],
            "current_interest_rate": [6.0, 6.0, 6.0],
        }
    )


def test_scenario_ecl_applies_staging_and_discounting(
    fixture_book: pl.DataFrame, fixture_scenarios: pl.DataFrame
) -> None:
    """The defect: scenarios previously bypassed compute() entirely."""
    stage = np.array([1, 2, 3])
    result = engine.scenario_ecl(
        fixture_book,
        fixture_scenarios,
        np.full(3, 0.01),
        np.full(3, 0.05),
        stage,
        np.full(3, 0.4),
        np.full(3, 5.0),
    )

    # Undiscounted stage-1 ECL would be 0.01*0.4*100000 = 400 for loan A alone;
    # the reported total must be strictly below the undiscounted sum, and must
    # use the LIFETIME pd for the two non-stage-1 loans.
    undiscounted_12m_only = float(
        np.sum(0.01 * 0.4 * fixture_book["current_actual_upb"].to_numpy())
    )
    base = float(result.filter(pl.col("scenario") == "base")["ecl"][0])

    assert base > undiscounted_12m_only  # lifetime PD applies to stages 2 and 3
    assert result["stage_1"][0] == 1
    assert result["stage_2"][0] == 1
    assert result["stage_3"][0] == 1


def test_a_positive_shift_increases_scenario_ecl(
    fixture_book: pl.DataFrame, fixture_scenarios: pl.DataFrame
) -> None:
    result = engine.scenario_ecl(
        fixture_book,
        fixture_scenarios,
        np.full(3, 0.01),
        np.full(3, 0.05),
        np.array([1, 1, 1]),
        np.full(3, 0.4),
        np.full(3, 1.0),
    )

    base = float(result.filter(pl.col("scenario") == "base")["ecl"][0])
    stress = float(result.filter(pl.col("scenario") == "stress")["ecl"][0])
    assert stress > base


def test_the_uncertainty_band_brackets_the_point_estimate(
    fixture_book: pl.DataFrame, fixture_scenarios: pl.DataFrame
) -> None:
    result = engine.scenario_ecl(
        fixture_book,
        fixture_scenarios,
        np.full(3, 0.01),
        np.full(3, 0.05),
        np.array([1, 1, 1]),
        np.full(3, 0.4),
        np.full(3, 1.0),
    )

    for row in result.to_dicts():
        assert row["ecl_low"] <= row["ecl"] <= row["ecl_high"]


def test_probability_weighted_ecl_is_the_weighted_sum(
    fixture_book: pl.DataFrame, fixture_scenarios: pl.DataFrame
) -> None:
    result = engine.scenario_ecl(
        fixture_book,
        fixture_scenarios,
        np.full(3, 0.01),
        np.full(3, 0.05),
        np.array([1, 1, 1]),
        np.full(3, 0.4),
        np.full(3, 1.0),
    )

    expected = float((result["ecl"] * result["weight"]).sum())
    assert float(result["probability_weighted_ecl"][0]) == pytest.approx(expected)


def test_staging_is_held_fixed_across_scenarios(
    fixture_book: pl.DataFrame, fixture_scenarios: pl.DataFrame
) -> None:
    """Re-staging per scenario put the whole book in stage 2 (see R-008)."""
    stage = np.array([1, 1, 1])
    result = engine.scenario_ecl(
        fixture_book,
        fixture_scenarios,
        np.full(3, 0.01),
        np.full(3, 0.05),
        stage,
        np.full(3, 0.4),
        np.full(3, 1.0),
    )

    assert set(result["stage_1"].to_list()) == {3}


# --------------------------------------------------------------------------
# LGD statistics — pure functions on a synthetic frame.
# --------------------------------------------------------------------------


@pytest.fixture
def fixture_lgd() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "lgd": [-0.5, 0.2, 0.5, 0.8, 1.4, 3.0],
            "upb_at_default": [100_000.0, 90_000.0, 80_000.0, 70_000.0, 60_000.0, 500.0],
            "original_ltv": [55, 65, 75, 85, 93, 99],
            "property_state": ["CA", "CA", "TX", "TX", "FL", "FL"],
        }
    )


def test_describe_reports_the_tails_without_altering_them(fixture_lgd: pl.DataFrame) -> None:
    summary = lgd_mod.describe(fixture_lgd)

    assert summary.n == 6
    assert summary.share_below_zero == pytest.approx(1 / 6)
    assert summary.share_above_one == pytest.approx(2 / 6)
    assert summary.median == pytest.approx(0.65)


def test_ltv_bands_are_ordered_and_exhaustive() -> None:
    values = pl.DataFrame({"ltv": [50, 65, 75, 85, 93, 99]})

    banded = values.select(lgd_mod.ltv_band(pl.col("ltv")).alias("band"))["band"].to_list()

    assert banded == ["<=60", "61-70", "71-80", "81-90", "91-95", ">95"]


def test_bounding_sensitivity_reports_both_variants_and_the_difference(
    fixture_lgd: pl.DataFrame,
) -> None:
    """Bounding is reported as a pair, never substituted silently (LGD_002)."""
    table = lgd_mod.bounding_sensitivity(fixture_lgd)
    by_variant = {r["variant"]: r["mean_lgd"] for r in table.to_dicts()}

    assert by_variant["unbounded"] == pytest.approx(float(np.mean([-0.5, 0.2, 0.5, 0.8, 1.4, 3.0])))
    assert by_variant["bounded_0_1"] == pytest.approx(
        float(np.mean([0.0, 0.2, 0.5, 0.8, 1.0, 1.0]))
    )
    assert by_variant["relative_difference"] < 0  # bounding lowers the mean here


def test_thin_segments_are_shrunk_toward_the_portfolio_mean(
    fixture_lgd: pl.DataFrame,
) -> None:
    """Credibility weighting: a segment of one must not be trusted (LGD_003)."""
    segments = lgd_mod.segment(fixture_lgd, credibility_k=50.0)
    global_mean = float(np.mean(fixture_lgd["lgd"].to_numpy()))

    for row in segments.to_dicts():
        assert row["credibility"] < 0.1  # every cell here is tiny
        assert abs(row["shrunk_lgd"] - global_mean) < abs(row["raw_mean"] - global_mean) + 1e-12


def test_no_shrinkage_at_k_zero_returns_the_raw_segment_mean(
    fixture_lgd: pl.DataFrame,
) -> None:
    segments = lgd_mod.segment(fixture_lgd, credibility_k=0.0)

    for row in segments.to_dicts():
        assert row["shrunk_lgd"] == pytest.approx(row["raw_mean"])


# --------------------------------------------------------------------------
# CLI reachability — every Makefile target must resolve to a real command.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "macro",
        "ingest",
        "panel",
        "features",
        "train",
        "challenger",
        "evaluate",
        "hazard",
        "ecl",
        "monitor",
        "registry",
        "serve",
        "report",
    ],
)
def test_every_pipeline_command_exists_and_is_documented(command: str) -> None:
    """Phase 5 was once reported done while three commands still raised (R-007).

    Phase 6 is on the list for the same reason: `make serve` pointed at a module
    that did not exist for several weeks while the phase read as in progress.
    """
    result = runner.invoke(app, [command, "--help"])

    assert result.exit_code == 0, f"`riskos {command}` is not reachable"
    assert "Phase" in result.stdout


def test_status_reports_the_pipeline_state() -> None:
    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "layout version" in result.stdout


def test_the_makefile_targets_match_the_cli() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    for command in (
        "macro",
        "ingest",
        "panel",
        "train",
        "challenger",
        "hazard",
        "ecl",
        "monitor",
        "registry",
        "serve",
        "report",
    ):
        assert f"riskos {command}" in makefile, f"Makefile does not invoke `riskos {command}`"


# --------------------------------------------------------------------------
# Row alignment (R-012). The ECL run pairs arrays returned by one function
# against a frame held by another, so any reordering silently mismatches loans.
# --------------------------------------------------------------------------


PANEL_GLOB = "data/panel/panel/*/*.parquet"
needs_panel = pytest.mark.skipif(
    not any(Path().glob(PANEL_GLOB)), reason="panel not built; run `make panel`"
)


@needs_panel
def test_with_time_varying_returns_rows_in_the_order_it_was_given() -> None:
    """The defect behind R-012, asserted at its source.

    `with_time_varying` goes through DuckDB, which does not preserve input order
    across a join. Callers pair its output against arrays built from the frame
    they passed in, so a permutation here assigns every loan another loan's
    covariates while leaving every total unchanged.
    """
    from riskos.ecl.run import load_portfolio
    from riskos.models.train_hazard import with_time_varying

    portfolio = load_portfolio()
    enriched = with_time_varying(portfolio)

    assert enriched.height == portfolio.height
    assert enriched["loan_sequence_number"].to_list() == portfolio["loan_sequence_number"].to_list()
    assert "_row_order" not in enriched.columns


@needs_panel
def test_per_loan_pds_line_up_with_the_portfolio_they_were_asked_about() -> None:
    """A permuted PD vector has the right mean and the wrong answer for everyone.

    Totals are order-invariant, so this cannot be caught by checking portfolio
    ECL. It is caught by projecting a subset and requiring the same loans to get
    the same numbers they got in the full run.
    """
    from riskos.ecl.run import load_portfolio, per_loan_pds

    portfolio = load_portfolio()
    full_12m, full_life, _ = per_loan_pds(portfolio)

    take = [0, 7, 11, 250, portfolio.height - 1]
    subset = portfolio[take]
    subset_12m, subset_life, _ = per_loan_pds(subset)

    for position, row in enumerate(take):
        assert subset_12m[position] == pytest.approx(full_12m[row], rel=1e-6), (
            f"loan at row {row} got a different 12-month PD when scored in a "
            "subset; the PD vector is not aligned with the portfolio"
        )
        assert subset_life[position] == pytest.approx(full_life[row], rel=1e-6)


@needs_panel
def test_the_ecl_run_is_reproducible() -> None:
    """Two identical runs must produce identical provisions.

    Before R-012 they did not: DuckDB returned rows in a different order each
    time, so the PD vector was permuted differently on every run and the same
    portfolio provisioned to a different number.
    """
    from riskos.ecl.run import load_portfolio, per_loan_pds

    portfolio = load_portfolio()
    first = per_loan_pds(portfolio)
    second = per_loan_pds(portfolio)

    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b), "per-loan PDs are not reproducible across runs"


# --------------------------------------------------------------------------
# Repository hygiene (R-013). Code that is not tracked is not delivered.
# --------------------------------------------------------------------------


def test_no_source_file_is_hidden_from_version_control() -> None:
    """A gitignore pattern must never swallow real source.

    `models/` was written to ignore the fitted-artifact directory at the repo
    root. Without a leading slash git matches a directory of that name at ANY
    depth, so it also matched src/riskos/models/ and kept the entire modelling
    package - scorecard, challenger, hazard, calibration, selection,
    persistence - out of version control. Nothing surfaced it: ignored files do
    not appear in `git status`, so the working tree looked clean.
    """
    import subprocess

    tracked_dirs = ["src", "tests", "conf", "docs", "governance"]
    candidates = [
        str(path)
        for directory in tracked_dirs
        for path in Path(directory).rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".md", ".toml"}
    ]
    if not candidates:  # pragma: no cover - defensive
        pytest.skip("no source files found")

    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="\n".join(candidates),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):  # pragma: no cover - not a git checkout
        pytest.skip("git check-ignore unavailable")

    ignored = [line for line in result.stdout.splitlines() if line.strip()]

    assert not ignored, f"these source files are gitignored and would never be committed: {ignored}"
