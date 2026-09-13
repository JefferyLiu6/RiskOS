"""The assumptions register is enforced, not merely encouraged.

Build plan rule 5: every numeric assumption lives in conf/assumptions.yaml with
a source. An entry missing a source, or a magic number in code that never
reaches the register, is the failure mode this project exists to avoid — so it
fails the test suite rather than a code review.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from riskos.config import CONF_DIR, load_yaml
from riskos.metrics import calibration, stability
from riskos.models import hazard

REGISTER = load_yaml(CONF_DIR / "assumptions.yaml")
ENTRIES = REGISTER["assumptions"]
BY_ID = {e["id"]: e for e in ENTRIES}


def test_register_is_not_empty() -> None:
    assert ENTRIES, "no assumptions registered"


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: str(e["id"]))
def test_every_entry_carries_the_required_fields(entry: dict[str, object]) -> None:
    missing = [f for f in REGISTER["schema"]["required_fields"] if not entry.get(f)]

    assert not missing, f"{entry.get('id')} is missing {missing}"


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: str(e["id"]))
def test_every_source_is_one_of_the_allowed_kinds(entry: dict[str, object]) -> None:
    assert entry["source"] in REGISTER["schema"]["allowed_sources"]


def test_ids_are_unique() -> None:
    assert len(BY_ID) == len(ENTRIES)


def test_registered_values_match_the_constants_the_code_actually_uses() -> None:
    # The register is only worth having if it cannot drift from the code. If a
    # default is retuned without updating the register, this fails.
    assert BY_ID["PSI_001"]["value"]["moderate"] == stability.MODERATE_SHIFT
    assert BY_ID["PSI_001"]["value"]["significant"] == stability.SIGNIFICANT_SHIFT
    assert BY_ID["PSI_002"]["value"] == stability.DEFAULT_EPSILON
    assert BY_ID["PSI_003"]["value"] == stability.DEFAULT_N_BINS
    assert BY_ID["CAL_001"]["value"] == calibration.DEFAULT_CONFIDENCE
    assert BY_ID["CAL_002"]["value"] == calibration.DEFAULT_N_BINS
    assert BY_ID["HAZ_002"]["value"] == list(hazard.AGE_BANDS)
    assert BY_ID["HAZ_005"]["value"] == list(hazard.DELINQUENCY_BANDS)


ASSUMPTION_REFERENCE = re.compile(r"\b(?:HAZ|ECL|LGD|EAD|MACRO|MON|PSI|CAL|PANEL)_[0-9]{3}\b")


def test_every_assumption_cited_in_the_code_actually_exists() -> None:
    """A citation to an unregistered assumption is worse than no citation.

    It reads as evidence that a number was documented when it was not. HAZ_004
    was cited in `riskos.models.train_hazard` for the house-price fallback while
    absent from the register, and nothing caught it: the other tests here
    iterate over what IS registered, so an entry that never existed is invisible
    to them. This test runs the other way round, from the code to the register.
    """
    cited: dict[str, set[str]] = {}
    for path in sorted(Path("src").rglob("*.py")):
        for match in ASSUMPTION_REFERENCE.findall(path.read_text(encoding="utf-8")):
            cited.setdefault(match, set()).add(str(path))

    missing = {ref: sorted(paths) for ref, paths in cited.items() if ref not in BY_ID}

    assert not missing, f"assumption ids cited in code but absent from the register: {missing}"


def test_monitoring_thresholds_match_the_register() -> None:
    """The rulebook and the register are two files; they must not drift apart.

    conf/monitoring.yaml is what the control actually fires on and
    conf/assumptions.yaml is what a reviewer reads to understand why. A
    threshold changed in one and not the other produces a register that
    describes a control nobody is running.
    """
    from riskos.monitor.alerts import HIGH_FALSE_POSITIVE_RATE
    from riskos.monitor.config import monitoring_config

    cfg = monitoring_config()
    by_metric = {r.metric: r for r in cfg.rules}

    assert list(BY_ID["MON_001"]["value"]["warn"]) == list(by_metric["observed_over_expected"].warn)
    assert list(BY_ID["MON_001"]["value"]["breach"]) == list(
        by_metric["observed_over_expected"].breach
    )
    assert BY_ID["MON_002"]["value"]["breach"] == by_metric["gini"].breach
    assert BY_ID["MON_003"]["value"]["breach"] == by_metric["brier_reliability"].breach
    assert BY_ID["MON_004"]["value"]["min_rows_for_alert"] == cfg.window.min_rows_for_alert
    assert (
        BY_ID["MON_004"]["value"]["min_defaults_for_performance"]
        == cfg.window.min_defaults_for_performance
    )
    assert BY_ID["MON_005"]["value"]["frequency"] == cfg.window.frequency
    assert (
        BY_ID["MON_005"]["value"]["outcome_window_months"]
        == cfg.observability.outcome_window_months
    )
    assert BY_ID["MON_006"]["value"] == HIGH_FALSE_POSITIVE_RATE


def test_every_monitoring_rule_cites_a_registered_basis() -> None:
    """A rule whose basis is a bare word is a threshold with no provenance."""
    from riskos.config import load_yaml as _load

    rules = _load(CONF_DIR / "monitoring.yaml")["rules"]
    unregistered = {
        r["id"]: r["basis"]
        for r in rules
        if r["basis"] not in BY_ID and r["basis"] not in REGISTER["schema"]["allowed_sources"]
    }

    assert not unregistered, f"monitoring rules citing an unregistered basis: {unregistered}"


def test_csi_breadth_counts_match_the_register() -> None:
    from riskos.monitor.config import monitoring_config

    rule = next(r for r in monitoring_config().rules if r.metric == "n_features_csi_significant")

    assert BY_ID["MON_007"]["value"]["warn"] == rule.warn
    assert BY_ID["MON_007"]["value"]["breach"] == rule.breach


def _quarters_in(window: str) -> int:
    """Count quarters in a 'YYYYQn-YYYYQn' window, inclusive."""
    start, end = window.split("-")
    y0, q0 = int(start[:4]), int(start[-1])
    y1, q1 = int(end[:4]), int(end[-1])
    return (y1 * 4 + q1) - (y0 * 4 + q0) + 1


def test_the_macro_observation_count_matches_its_own_effective_window() -> None:
    """MACRO_001 claimed 32 observations while the fit used 24 (F-019).

    An observation count is a data outcome, not a code constant, so the other
    tests in this file could not see the disagreement. This one at least holds
    the register to internal consistency: the count must equal the quarters in
    the effective window it records.
    """
    value = BY_ID["MACRO_001"]["value"]

    assert value["n_observations"] == _quarters_in(value["effective_window"])
    # The effective window starts strictly after the requested one, and the
    # register says why: the HPI series begins a year before the first usable
    # year-on-year observation.
    assert value["effective_window"] != value["primary_window"]
    assert value["hpi_series_start"] < value["effective_window"].split("-")[0]


@pytest.mark.skipif(
    not Path("reports/figures/macro_coefficients.csv").exists(),
    reason="needs the fitted macro coefficient table",
)
def test_the_macro_observation_count_matches_what_the_fit_produced() -> None:
    """The register must agree with the artefact the fit wrote, not only with itself."""
    import polars as pl

    fitted = pl.read_csv("reports/figures/macro_coefficients.csv")
    # The fit labels its window by year ("1999-2006"); the register by quarter.
    start, end = BY_ID["MACRO_001"]["value"]["primary_window"].split("-")
    label = f"{start[:4]}-{end[:4]}"
    pre = fitted.filter(pl.col("window") == label)

    assert pre.height > 0, f"no rows labelled {label!r} in macro_coefficients.csv"
    assert pre["n_observations"].unique().to_list() == [
        BY_ID["MACRO_001"]["value"]["n_observations"]
    ]
