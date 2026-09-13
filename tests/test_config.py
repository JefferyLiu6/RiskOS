"""Config must fail loudly: typos rejected, and no ingest without a sourced layout."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from riskos.config import CONF_DIR, Layout, LayoutNotLoadedError, data_config, load_yaml


def test_real_data_yaml_loads() -> None:
    cfg = data_config()

    assert cfg.source.variant == "sample"
    assert cfg.paths.raw.is_absolute()


def test_all_required_vintages_are_configured() -> None:
    cfg = data_config()

    assert cfg.vintages.core == tuple(range(1999, 2013))
    assert cfg.vintages.benign == tuple(range(2015, 2020))
    assert len(cfg.vintages.all) == 19


def test_layout_is_populated_from_the_user_guide() -> None:
    cfg = data_config()

    assert cfg.layout.is_ready
    cfg.layout.require_ready()  # does not raise
    assert "Release 47" in (cfg.layout.version or "")
    assert len(cfg.layout.origination_columns) == 31
    assert len(cfg.layout.performance_columns) == 35


def test_an_unpopulated_layout_still_refuses_to_ingest() -> None:
    # The guard itself must keep working even though the real config now passes.
    with pytest.raises(LayoutNotLoadedError, match="User Guide"):
        Layout().require_ready()


def test_the_loan_key_is_named_not_inferred_from_position() -> None:
    # loan_sequence_number is field 20 of the origination file and field 1 of
    # the performance file. Anything positional is wrong on one of them.
    cfg = data_config()
    orig = [c.name for c in cfg.layout.origination_columns]
    perf = [c.name for c in cfg.layout.performance_columns]

    assert cfg.layout.key_column == "loan_sequence_number"
    assert orig.index("loan_sequence_number") == 19
    assert perf.index("loan_sequence_number") == 0


def test_zero_balance_mapping_covers_every_code_in_the_guide() -> None:
    cfg = data_config()

    assert cfg.zero_balance_codes.is_ready
    assert cfg.zero_balance_codes.column == "zero_balance_code"
    # Release 47 documents exactly these seven codes.
    assert sorted(cfg.zero_balance_codes.mapping) == ["01", "02", "03", "09", "15", "16", "96"]


def test_documented_sentinels_are_recorded_on_their_columns() -> None:
    cfg = data_config()
    by_name = {c.name: c for c in cfg.layout.origination_columns}

    assert by_name["credit_score"].null_values == ("9999",)
    assert by_name["original_ltv"].null_values == ("999",)
    assert by_name["original_dti"].null_values == ("999",)
    assert by_name["number_of_borrowers"].null_values == ("99",)


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Layout(version="v1", unexpected_key=True)  # type: ignore[call-arg]


def test_default_definition_is_locked_in_panel_config() -> None:
    panel = load_yaml(CONF_DIR / "panel.yaml")

    dd = panel["default_definition"]
    assert dd["locked"] is True
    assert dd["dpd_threshold_days"] == 90
    # 90+ DPD is status >= "03" in months, plus REO Acquisition. "XX" is
    # Not Available and must never be read as current.
    assert dd["dpd_min_months"] == 3
    assert dd["dpd_also_default_statuses"] == ["RA"]
    assert dd["dpd_unknown_statuses"] == ["XX"]
    # Prepayment and maturity are not defaults; defect repurchase is excluded.
    assert "prepaid_or_matured" in dd["not_default"]
    assert dd["excluded_not_counted_as_good"] == ["defect_prior_to_credit_event"]
    assert panel["panel"]["horizon_months"] == 12
    assert panel["panel"]["observation_start"] == "1999-Q1"


def test_splits_are_out_of_time_and_disjoint_in_period() -> None:
    splits = load_yaml(CONF_DIR / "panel.yaml")["splits"]

    assert splits["train"]["observation_dates"] == ["1999-Q1", "2006-Q4"]
    assert splits["oot_stress"]["observation_dates"] == ["2007-Q1", "2009-Q4"]
    assert splits["oot_benign"]["observation_dates"] == ["2015-Q1", "2019-Q4"]
    assert "random" not in str(splits).lower()
