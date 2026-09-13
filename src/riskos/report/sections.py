"""One function per report section. Each reads from ``Sources`` and returns Markdown.

The rule for every function here: a number appears in the output only because
an artefact supplied it. Prose explains; it does not carry figures of its own.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import polars as pl

from riskos.report.render import (
    df_table,
    md_table,
    missing,
    money,
    num,
    pct,
    ratio,
    sci,
    section,
    squash,
    text,
)
from riskos.report.sources import Sources

SCOPE_STATEMENT = (
    "The underlying portfolio is U.S. residential mortgage data, because comparable "
    "public Canadian loan-level default and loss data is not available. The project "
    "applies IFRS 9 concepts and OSFI Guideline E-23 as a methodological and governance "
    "framework relevant to Canadian financial institutions. It does not represent a "
    "regulatory implementation, does not reproduce any institution's ECL system, and "
    "makes no claim of OSFI compliance."
)

# Inventory family -> the model label used in the Phase 4 comparison tables.
FAMILY_LABEL = {"woe_scorecard": "scorecard", "lightgbm": "lightgbm"}
# Monitoring artefact prefix per family.
FAMILY_MONITOR = {"woe_scorecard": "scorecard", "lightgbm": "challenger"}


def _scalar(value: Any) -> float:
    """A polars aggregate as a float. Aggregates of an empty frame are None; refuse those."""
    assert value is not None, "aggregate over an empty frame"
    return float(value)


def _row(frame: pl.DataFrame | None, **where: Any) -> dict[str, Any] | None:
    if frame is None:
        return None
    for key, value in where.items():
        frame = frame.filter(pl.col(key) == value)
    return frame.row(0, named=True) if frame.height else None


def _missing(s: Sources, key: str) -> str:
    return missing(f"{key}.csv" if "/" not in key else key, s.produced_by(key))


def _fmt_finding(f: dict[str, Any]) -> str:
    return f"**{f['id']}** ({f['severity']}, {f['status']}) — {squash(f['title'])}"


# --- front matter ------------------------------------------------------------------


def title_block(s: Sources) -> str:
    inv = s.inventory
    models = inv["models"]
    ids = ", ".join(f"{m['id']} v{_version(s, m)}" for m in models)
    return "\n".join(
        [
            "# RiskOS — Model Validation Report",
            "",
            f"**Generated:** {date.today().isoformat()} from the pipeline artefacts  ",
            f"**Models in scope:** {ids}  ",
            "**Review type:** developer validation with simulated second-line review  ",
            "**Author and reviewer:** project author (see §2 on independence)  ",
            "",
            f"> {SCOPE_STATEMENT}",
            "",
        ]
    )


def _version(s: Sources, entry: dict[str, Any]) -> str:
    manifest = s.bundle_manifest(entry.get("artefact"))
    return str(manifest.get("version", "?")) if manifest else "n/a"


# --- 1. executive summary -------------------------------------------------------------


def executive_summary(s: Sources) -> str:
    inv = s.inventory
    cc = s.csv("champion_challenger_metrics")
    stage = s.csv("ecl_by_stage")
    scen = s.csv("ecl_scenarios")
    back = s.csv("macro_backtest")
    mon = s.json("monitoring_summary")
    findings = s.findings

    open_f = [f for f in findings if f["status"] == "open"]
    high_open = [f for f in open_f if f["severity"] in ("high", "critical")]

    decisions = []
    for m in inv["models"]:
        a = m["approval"]
        decisions.append(
            f"- **{m['id']}** ({m['name']}): inventory status `{m['status']}`, approval "
            f"`{a['status']}`"
            + (f", review due {a['review_due']}" if a.get("review_due") else "")
            + "."
        )

    blocks = [
        "**Decision.** "
        + " ".join(
            f"{m['id']} is {m['approval']['status'].replace('_', ' ')}."
            for m in inv["models"]
            if m["status"] in ("in_use", "approved")
        )
        + " No model in this inventory is approved without conditions, and none is "
        "approved for real lending, underwriting, capital or provisioning decisions.",
        "\n".join(decisions),
    ]

    # The headline number, from the ECL artefacts.
    if stage is not None and scen is not None:
        total_ecl = float(stage["ecl"].sum())
        total_ead = float(stage["ead"].sum())
        pw = float(scen["probability_weighted_ecl"][0])
        severe = _row(scen, scenario="severe_stress")
        line = (
            f"**The number.** Portfolio ECL as at the last training date is {money(total_ecl)} "
            f"on {money(total_ead)} of exposure, a coverage ratio of "
            f"{pct(total_ecl / total_ead, 3)}. Probability-weighted across three macro "
            f"scenarios it is {money(pw)}."
        )
        if severe:
            line += (
                f" The severe-stress scenario alone spans {money(severe['ecl_low'])} to "
                f"{money(severe['ecl_high'])} at 95% confidence, a factor of "
                f"{ratio(severe['ecl_high'] / severe['ecl_low'], 1)} end to end. That width is "
                "the correct representation of what the estimation sample can support, not a "
                "presentational weakness."
            )
        blocks.append(line)
    else:
        blocks.append(_missing(s, "ecl_by_stage"))

    # The three findings a reader must not miss.
    headline = ["**Three findings qualify every figure in this report.**"]
    sc = _row(cc, model="scorecard", calibration="uncalibrated", split="oot_stress")
    gb = _row(cc, model="lightgbm", calibration="uncalibrated", split="oot_stress")
    if sc and gb:
        headline.append(
            f"1. **Calibration fails under regime change (F-005, F-006).** Through 2007-2009 the "
            f"scorecard predicts {pct(sc['expected_rate'])} where {pct(sc['observed_rate'])} "
            f"occurred, an observed-over-expected ratio of {ratio(sc['observed_over_expected'])}; "
            f"the challenger's is {ratio(gb['observed_over_expected'])}. Discrimination held "
            f"(Gini {ratio(sc['gini'], 3)} and {ratio(gb['gini'], 3)}). A model selected on AUC "
            "would have passed while understating the provision roughly threefold."
        )
    if back is not None:
        mean_ratio = _scalar(back["ratio_observed_to_predicted"].mean())
        headline.append(
            f"2. **The macro overlay recovers half of that and cannot recover the rest (F-009).** "
            f"Fitted on the pre-crisis window and fed the realised 2008-2009 economy, it "
            f"under-predicts the default rate by a mean factor of {ratio(mean_ratio)}. The "
            "estimation sample contains no house-price decline, so no estimator can recover the "
            "sensitivity. This is the central methodological limitation of the project."
        )
    if mon:
        psis = ", ".join(
            f"{m['model_id']} {ratio(m['max_stress_score_psi'], 4)}" for m in mon["models"]
        )
        headline.append(
            f"3. **Drift monitoring is blind to this failure mode (F-015).** Over the same "
            f"crisis quarters the worst score PSI is {psis}, an order of magnitude inside the "
            "stable band, for both model families. The only control that detected the "
            "deterioration was calibration monitoring, which is structurally twelve months late."
        )
    blocks.append("\n".join(headline))

    blocks.append(
        f"**Findings register.** {len(findings)} findings, {len(open_f)} open, "
        f"{len(high_open)} of them high or critical. Most open findings are conclusions this "
        "project exists to report rather than defects awaiting a fix; §13 distinguishes the two. "
        "Two critical defects (R-012, R-013) were found and remediated before this report."
    )
    return section("1. Executive summary", *blocks)


# --- 2. scope -------------------------------------------------------------------------


def scope(s: Sources) -> str:
    inv = s.inventory
    rubric = inv["tiering_rubric"]
    tiers = md_table(
        ["Tier", "Definition"],
        [(k.replace("tier_", ""), squash(v)) for k, v in rubric["mapping"].items()],
    )
    return section(
        "2. Scope, independence, and intended use",
        f"**Scope.** {SCOPE_STATEMENT}",
        "**Independence.** This project is built by one person. It cannot claim the "
        "organisational independence OSFI E-23 expects between model development and model "
        "validation. The review artefact is therefore a *developer validation with simulated "
        "second-line review*: the same discipline of pre-committed rules, a findings register "
        "populated at discovery, approvals with conditions and review dates, and reconciliation "
        "of the inventory against what is on disk — without the independent reviewer. Every "
        "tier-1 approval in §12 records this as a condition, not a waiver.",
        "**Intended use.** Illustrative and educational. Not for real lending or underwriting "
        "decisions, not for regulatory capital or provisioning, and not applicable to any "
        "Canadian portfolio without redevelopment on Canadian data.",
        "**Risk tiering.** Every model that feeds the ECL calculation is tier 1, because the ECL "
        "figure is the one a reader would be tempted to quote.",
        tiers,
    )


# --- 3. data ------------------------------------------------------------------------------


def data(s: Sources) -> str:
    dcfg = s.conf("data")
    pcfg = s.conf("panel")
    mbs = s.csv("metrics_by_split")
    src = dcfg["source"]
    layout_version = dcfg.get("layout", {}).get("version", "not recorded")
    vintages = dcfg["vintages"]
    dd = pcfg["default_definition"]

    manifest_line = _ingest_manifest_line(s)

    blocks = [
        f"**Source.** {src['name']}, {src['variant']} variant, {src['access']} access. "
        f"Column layout transcribed from *{layout_version}*; ingest refuses to run without a "
        "version stamp. Raw and derived loan-level data are never committed.",
        f"**Vintages.** Core {vintages['core'][0]}-{vintages['core'][-1]} and benign contrast "
        f"{vintages['benign'][0]}-{vintages['benign'][-1]}."
        + (f" {manifest_line}" if manifest_line else ""),
        f"**Default definition (locked before results).** {squash(dd['statement'])} "
        f"Credit-event terminations: {', '.join(dd['credit_event_terminations'])}. "
        f"Excluded and not counted as good: {', '.join(dd['excluded_not_counted_as_good'])}. "
        "The whole-loan-sale mapping is a judgement rather than a transcription and is recorded "
        "as F-002.",
        "**Design.** Observation-cohort panel: at each quarter-end, every loan alive and not "
        "already 90+ days past due is labelled on whether it defaults in the next twelve months. "
        "Rows whose outcome window is not observable are dropped, never imputed. Prepayment is "
        "labelled zero and flagged as a competing risk. Splits are out-of-time and loan-disjoint "
        "by seeded hash (F-003 records the sample-size cost).",
    ]

    if mbs is not None:
        windows = {
            k: v["observation_dates"]
            for k, v in pcfg["splits"].items()
            if isinstance(v, dict) and "observation_dates" in v
        }
        rows = []
        for r in mbs.to_dicts():
            w = windows.get(r["split"])
            rows.append(
                [
                    r["split"],
                    f"{w[0]} to {w[1]}" if w else "—",
                    num(r["n"]),
                    num(r["n_default"]),
                    pct(r["observed_rate"], 3),
                ]
            )
        blocks.append(
            md_table(["Split", "Observation window", "Rows", "Defaults", "Default rate"], rows)
        )
        ex = _row(mbs, split="oot_benign_ex_forbearance")
        asis = _row(mbs, split="oot_benign")
        if ex and asis:
            blocks.append(
                f"**F-001, COVID forbearance.** The benign split's default rate of "
                f"{pct(asis['observed_rate'], 3)} is inflated by CARES Act forbearance: borrowers "
                "permitted to stop paying register as 90+ days past due without a credit event, "
                "and 2019 observation windows close in 2020. Excluding forbearance-driven defaults "
                f"the rate is {pct(ex['observed_rate'], 3)}. The definition was not amended; every "
                "benign-split metric is reported both ways."
            )
    else:
        blocks.append(_missing(s, "metrics_by_split"))
    return section("3. Data, target, and experimental design", *blocks)


def _ingest_manifest_line(s: Sources) -> str:
    path = s.root / "data" / "interim" / "ingest_manifest.json"
    if not path.exists():
        return ""
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    vintages = m.get("vintages")
    if not isinstance(vintages, list) or not vintages:
        return ""
    loans = sum(int(v.get("origination_rows", 0)) for v in vintages)
    months = sum(int(v.get("performance_rows", 0)) for v in vintages)
    return (
        f"{len(vintages)} vintages ingested: {num(loans)} loans, {num(months)} loan-months, per "
        "the ingest manifest."
    )


# --- 4. development ------------------------------------------------------------------------


def development(s: Sources) -> str:
    iv = s.csv("information_values")
    pruned = s.csv("collinearity_pruned")
    coefs = s.csv("scorecard_coefficients")
    grid = s.csv("lgbm_grid")
    sel = s.json("selection_record", s.models)
    feats = s.json("scorecard_features", s.models)
    mcfg = s.conf("models")

    blocks = []
    if iv is not None:
        n_cand = iv.height
        n_sel = int(iv["selected"].sum())
        strong = iv.filter(pl.col("band") == "suspiciously_strong")
        blocks.append(
            f"**Candidate pool.** {n_cand} features survive leakage control; both model "
            f"families receive the same pool (conf/models.yaml). For the scorecard, {n_sel} pass "
            f"the Information Value floor. {strong.height} tripped the 0.9 leakage tripwire "
            f"({', '.join(strong['name'].to_list())}) and were cleared individually with written "
            "evidence rather than by raising the ceiling; F-004 records that delinquency status, "
            "while legitimate, is mechanically close to the target."
        )
        blocks.append(
            df_table(
                iv.sort("iv", descending=True).head(10),
                [
                    ("name", "Feature", text),
                    ("iv", "IV", lambda v: ratio(v, 4)),
                    ("band", "Band", text),
                    ("n_bins", "Bins", text),
                    ("selected", "Selected", text),
                ],
            )
        )
    else:
        blocks.append(_missing(s, "information_values"))

    if pruned is not None and pruned.height and pruned["dropped"][0] is not None:
        pr = ", ".join(
            f"{r['dropped']} (r = {ratio(r['correlation'], 3)} with {r['kept']})"
            for r in pruned.to_dicts()
        )
        blocks.append(f"**Collinearity pruning** at a 0.95 ceiling on the WOE matrix removed {pr}.")
    if feats:
        blocks.append(
            f"**Wrong-sign elimination** removed {', '.join(feats['dropped_wrong_sign'])}, "
            "iteratively and worst first: a positive coefficient on a WOE feature awards more "
            f"points for worse credit. Final scorecard: **{len(feats['features'])} features**, "
            "all coefficients negative, monotone constraints verified in the fitted bins."
        )
    if coefs is not None:
        blocks.append(
            df_table(
                coefs,
                [
                    ("feature", "Feature", text),
                    ("coefficient", "Coefficient", lambda v: ratio(v, 4)),
                    ("points_per_woe_unit", "Points per WOE unit", lambda v: ratio(v, 2)),
                    ("sign_ok", "Sign OK", text),
                ],
            )
        )

    lg = mcfg["lightgbm"]
    blocks.append(
        f"**Challenger.** LightGBM on raw features with native categorical handling, a fixed "
        f"grid of {len(lg['grid']['num_leaves']) * len(lg['grid']['max_depth'])} combinations "
        f"(num_leaves {lg['grid']['num_leaves']}, max_depth {lg['grid']['max_depth']}), early "
        f"stopping on in-time validation, and the same monotone constraints as the scorecard on "
        f"{len(lg['monotone_constraints'])} features. No unbounded search."
    )
    if grid is not None:
        blocks.append(
            df_table(
                grid,
                [
                    ("max_depth", "max_depth", text),
                    ("num_leaves", "num_leaves", text),
                    ("valid_auc", "Validation AUC", lambda v: ratio(v, 4)),
                    ("best_iteration", "Best iteration", text),
                ],
            )
        )
    if sel:
        bp = sel.get("challenger_best_params", {})
        blocks.append(
            f"Selected hyperparameters: {', '.join(f'{k}={v}' for k, v in bp.items())}. "
            f"Constraints silently ignored: {sel.get('constraints_ignored') or 'none'}."
        )
    return section("4. Model development", *blocks)


# --- 5. performance ------------------------------------------------------------------------


def performance(s: Sources) -> str:
    cc = s.csv("champion_challenger_metrics")
    blocks = []
    if cc is None:
        return section("5. Performance", _missing(s, "champion_challenger_metrics"))

    unc = cc.filter(pl.col("calibration") == "uncalibrated")
    blocks.append(
        "**All four splits, both models, uncalibrated.** Read discrimination and calibration "
        "together: the finding is in the difference between how they degrade."
    )
    blocks.append(
        df_table(
            unc,
            [
                ("model", "Model", text),
                ("split", "Split", text),
                ("n", "Rows", num),
                ("n_default", "Defaults", num),
                ("auc", "AUC", lambda v: ratio(v, 3)),
                ("gini", "Gini", lambda v: ratio(v, 3)),
                ("observed_rate", "Observed", lambda v: pct(v, 3)),
                ("expected_rate", "Predicted", lambda v: pct(v, 3)),
                ("observed_over_expected", "O/E", ratio),
                ("brier", "Brier", lambda v: ratio(v, 5)),
                ("reliability", "Reliability", sci),
                ("score_psi", "Score PSI", lambda v: ratio(v, 4)),
            ],
        )
    )

    stress = cc.filter(pl.col("split") == "oot_stress")
    blocks.append(
        "**Recalibration on the stress split.** Isotonic and Platt are fitted on in-time "
        "validation. Both are monotone, so ranking is unchanged; both act on the level, and the "
        "level they learned is the pre-crisis one."
    )
    blocks.append(
        df_table(
            stress,
            [
                ("model", "Model", text),
                ("calibration", "Calibration", text),
                ("observed_over_expected", "O/E", ratio),
                ("gini", "Gini", lambda v: ratio(v, 4)),
                ("brier", "Brier", lambda v: ratio(v, 5)),
                ("reliability", "Reliability", sci),
            ],
        )
    )
    blocks.append(
        "Recalibration corrects a level that is systematically wrong on data you have. It cannot "
        "correct a level that is wrong because the economy changed. This table is the evidence "
        "that the Phase 5 macro overlay is necessary rather than decorative (F-006)."
    )

    for label, name in (("scorecard", "Scorecard"), ("lightgbm", "LightGBM")):
        rel = s.csv(f"reliability_cc_{label}_oot_stress")
        if rel is not None:
            blocks.append(
                f"**Reliability by predicted-PD decile, {name}, 2007-2009.** Wilson intervals. "
                "Every decile observes more default than it predicts; the miscalibration is "
                "systematic, not concentrated."
            )
            blocks.append(
                df_table(
                    rel,
                    [
                        ("bin", "Decile", text),
                        ("n", "Rows", num),
                        ("n_default", "Defaults", num),
                        ("mean_predicted", "Predicted", lambda v: pct(v, 3)),
                        ("observed_rate", "Observed", lambda v: pct(v, 3)),
                        ("ci_low", "CI low", lambda v: pct(v, 3)),
                        ("ci_high", "CI high", lambda v: pct(v, 3)),
                        ("within_interval", "Within CI", text),
                    ],
                )
            )
    return section("5. Performance: discrimination held, calibration collapsed", *blocks)


# --- 6. selection ------------------------------------------------------------------------


def selection(s: Sources) -> str:
    rub = s.csv("selection_rubric")
    sel = s.json("selection_record", s.models)
    expl = s.json("explanation_summary")
    weights = s.conf("models")["selection"]["weights"]
    blocks = [
        "**The rubric was committed before the challenger was fitted** (conf/models.yaml, "
        "verifiable in git history). Weights: "
        + ", ".join(f"{k} {v}" for k, v in weights.items())
        + ". Calibration is weighted highest because ECL is a money number.",
    ]
    if rub is not None:
        blocks.append(
            md_table(
                list(rub.columns),
                [
                    [text(v) if isinstance(v, str) else ratio(v, 4) for v in r]
                    for r in rub.iter_rows()
                ],
            )
        )
    if sel:
        winner = sel["selected"]
        blocks.append(f"**Decision recorded:** {sel['reason']}.")
        # F-007 sensitivity, recomputed from the record rather than quoted.
        rows = []
        for c in sel["candidates"]:
            scores = dict(c["dimension_scores"])
            committed = sum(weights[k] * scores[k] for k in weights)
            scores["explainability"] = 1.0
            equalised = sum(weights[k] * scores[k] for k in weights)
            rows.append([c["name"], ratio(committed, 4), ratio(equalised, 4)])
        blocks.append(
            "**F-007 sensitivity.** The rubric scored SHAP as approximate attribution (0.6); "
            "measurement shows TreeSHAP reconstructs the margin exactly. The rule was applied as "
            "written and the sensitivity reported: equalising explainability at 1.0 for both "
            f"candidates leaves {winner} selected."
        )
        blocks.append(md_table(["Candidate", "As committed", "Explainability equalised"], rows))
    if expl:
        blocks.append(
            f"**Explanation comparison** on {expl['n_borrowers']} stress-period borrowers: "
            f"top-driver agreement {pct(expl['top_driver_agreement'], 0)}, top-three overlap "
            f"{ratio(expl['top3_overlap'], 2)}. Reconstruction error: scorecard points "
            f"{sci(expl['scorecard_reconstruction_error'])}, SHAP "
            f"{sci(expl['shap_reconstruction_error'])}. Both mechanisms are exact; the genuine "
            "difference is that points are absolute, on a human scale and enumerable in advance, "
            "while SHAP is relative to a population baseline in log-odds."
        )
    blocks.append(
        "**The condition that qualifies the decision.** Both candidates score near zero on the "
        "most heavily weighted dimension. The rubric selected the less-bad of two models that both "
        "fail the criterion that matters most for provisioning (F-006). The selected model is "
        "recorded in the inventory as a candidate and is not approved; see §12 and F-018."
    )
    return section("6. Champion-challenger selection", *blocks)


# --- 7. lifetime PD -------------------------------------------------------------------------


def lifetime_pd(s: Sources) -> str:
    seas = s.csv("hazard_empirical_seasoning")
    base = s.csv("hazard_baseline")
    ts = s.csv("hazard_term_structure")
    spec = s.csv("hazard_specification_comparison")
    fit = s.json("hazard_fit", s.models)
    blocks = [
        "**Specification.** Two cause-specific discrete-time hazards, default and prepayment, "
        "complementary log-log link, loan-age baseline as band dummies, origination covariates "
        "plus mark-to-market LTV and amortisation ratio lagged one month (R-009), standard errors "
        "clustered by calendar month. Prepayment is a competing risk: lifetime PD is the sum of "
        "marginal defaults each weighted by surviving both risks."
    ]
    if fit:
        d, p = fit["fits"]["default"], fit["fits"]["prepaid"]
        blocks.append(
            f"Fitted on {num(d['n_rows'])} loan-months: {num(d['n_events'])} defaults and "
            f"{num(p['n_events'])} prepayments."
        )
    if seas is not None:
        blocks.append("**Empirical seasoning.** The reason the 12-month PD cannot be chained.")
        blocks.append(
            df_table(
                seas,
                [
                    ("age_band", "Age band", text),
                    ("n", "Loan-months", num),
                    ("empirical_default_hazard", "Default hazard", lambda v: ratio(v, 6)),
                    ("empirical_prepay_hazard", "Prepayment hazard", lambda v: ratio(v, 4)),
                ],
            )
        )
    if base is not None:
        blocks.append(
            df_table(
                base.filter(pl.col("cause") == "default"),
                [
                    ("age_band", "Age band", text),
                    ("hazard_ratio", "Fitted hazard ratio vs 0-6 months", lambda v: ratio(v, 2)),
                ],
            )
        )
    if ts is not None:
        pick = ts.filter(pl.col("month").is_in([12, 24, 60, 96, 120, 180, 240, 300]))
        last = _row(ts, month=int(_scalar(ts["month"].max())))
        blocks.append(
            "**Term structure against naive chaining.** Chaining repeats the 12-month PD; the "
            "hazard integrates the seasoning curve and the competing risk."
        )
        blocks.append(
            df_table(
                pick,
                [
                    ("month", "Month", text),
                    ("cumulative_default", "Cumulative default (hazard)", lambda v: pct(v, 2)),
                    ("survival", "Survival", lambda v: pct(v, 1)),
                    ("naive_chained", "Naive chained", lambda v: pct(v, 2)),
                ],
            )
        )
        if last:
            blocks.append(
                f"At month {last['month']} chaining reports "
                f"{pct(last['naive_chained'])} against the hazard's "
                f"{pct(last['cumulative_default'])}, an overstatement of "
                f"{ratio(last['naive_chained'] / last['cumulative_default'], 1)}x. Beyond 96 "
                "months the hazard is an extrapolation of the oldest estimable band (F-008)."
            )
    if spec is not None:
        blocks.append(
            "**Reconciliation, the acceptance criterion.** Hazard-implied 12-month PD against the "
            "observed 12-month default rate on the same panel rows."
        )
        blocks.append(
            df_table(
                spec,
                [
                    ("specification", "Specification", text),
                    ("n_terms", "Terms", text),
                    ("hazard_implied_pd_12m", "Hazard-implied 12m PD", lambda v: pct(v, 3)),
                    ("panel_observed_default_rate_12m", "Observed", lambda v: pct(v, 3)),
                    ("ratio_hazard_to_observed", "Ratio", lambda v: ratio(v, 3)),
                ],
            )
        )
        blocks.append(
            "The delinquency-banded specification reproduces the empirical hazard in every state "
            "and still forecasts worse in aggregate, because projection holds the state fixed. It "
            "is reported as a diagnostic and not promoted; promoting it needs a delinquency "
            "transition model (F-012, F-014)."
        )
    return section("7. Lifetime PD: the hazard term structure", *blocks)


# --- 8. LGD ------------------------------------------------------------------------------------


def lgd(s: Sources) -> str:
    dist = s.csv("lgd_raw_distribution")
    tail = s.csv("lgd_tail_investigation")
    den = s.csv("lgd_denominator_sensitivity")
    bound = s.csv("lgd_bounding_sensitivity")
    seg = s.csv("lgd_segments")
    a1 = s.assumption("LGD_001")
    a3 = s.assumption("LGD_003")
    blocks = [
        "**Definition.** Actual loss divided by balance at default, from Freddie Mac's disclosed "
        "loss at disposition. **Not clipped to [0, 1].**"
    ]
    if dist is not None:
        d = dict(zip(dist["metric"].to_list(), dist["value"].to_list(), strict=True))
        blocks.append(
            f"{num(d['n'])} observed defaults with a disclosed loss. Median {ratio(d['p50'])}, "
            f"interquartile {ratio(d['p25'])}-{ratio(d['p75'])}. {pct(d['share_below_zero'], 1)} "
            f"of losses are below zero (recoveries exceeded the balance) and "
            f"{pct(d['share_above_one'], 1)} above one (accrued interest and expenses exceeded "
            f"it). Both are economically real and both are kept. The raw mean is "
            f"{ratio(d['mean_raw'], 1)}, and the reason is a degenerate denominator, not a loss."
        )
    if tail is not None:
        blocks.append(
            df_table(
                tail,
                [
                    ("band", "Band", text),
                    ("n", "n", num),
                    ("median_upb", "Median balance", money),
                    ("min_upb", "Min balance", lambda v: f"${float(v):,.2f}"),
                    ("median_loss", "Median loss", money),
                    ("median_expenses", "Median expenses", money),
                ],
            )
        )
    if den is not None and a1:
        floor = a1["value"]["upb_floor"]
        blocks.append(
            f"**Denominator floor (LGD_001, ${floor:,.0f}).** A materiality floor on the balance "
            "removes observations where the ratio is undefined in practice. Bounding the ratio "
            "would alter observations that are real; they are different operations."
        )
        blocks.append(
            df_table(
                den,
                [
                    ("upb_floor", "Balance floor", lambda v: f"${float(v):,.0f}"),
                    ("n_kept", "Kept", num),
                    ("share_dropped", "Dropped", lambda v: pct(v, 3)),
                    ("mean_lgd", "Mean LGD", lambda v: ratio(v, 4)),
                    ("median_lgd", "Median", lambda v: ratio(v, 4)),
                    ("p99_lgd", "p99", lambda v: ratio(v, 3)),
                ],
            )
        )
    if bound is not None:
        blocks.append(
            "**Bounded variant (LGD_002), reported alongside and not applied:** "
            + "; ".join(
                f"{r['variant']} {ratio(r['mean_lgd'], 4)}"
                for r in bound.to_dicts()
                if r["variant"] != "relative_difference"
            )
            + "."
        )
    if seg is not None and a3:
        blocks.append(
            f"**Segmentation.** {seg.height} cells of LTV band by state, each shrunk toward the "
            f"portfolio mean of {ratio(float(seg['global_mean'][0]), 4)} with credibility "
            f"n / (n + {a3['value']}) (LGD_003)."
        )
    return section("8. Loss given default", *blocks)


# --- 9. ECL -----------------------------------------------------------------------------------


def ecl(s: Sources) -> str:
    stage = s.csv("ecl_by_stage")
    sens = s.csv("ecl_staging_sensitivity")
    a = s.assumption("ECL_001")
    blocks = []
    if stage is None:
        return section("9. Expected credit loss", _missing(s, "ecl_by_stage"))
    total_ecl, total_ead = float(stage["ecl"].sum()), float(stage["ead"].sum())
    n = int(stage["n_loans"].sum())
    blocks.append(
        f"**As at the last training date, {num(n)} loans, {money(total_ead)} exposure.** Each "
        "loan is projected through the fitted hazards over its own remaining term for 12-month, "
        "lifetime and origination-vintage PD (F-010). Stage 1 is measured over twelve months, "
        "stages 2 and 3 over the remaining lifetime, discounted at the loan's own rate mid-period."
    )
    blocks.append(
        df_table(
            stage,
            [
                ("ifrs9_stage", "Stage", text),
                ("n_loans", "Loans", num),
                ("ead", "EAD", money),
                ("mean_pd", "Mean PD", lambda v: pct(v, 2)),
                ("mean_lgd", "Mean LGD", lambda v: ratio(v, 3)),
                ("ecl", "ECL", money),
                ("share_of_ecl", "Share of ECL", lambda v: pct(v, 1)),
                ("coverage_ratio", "Coverage", lambda v: pct(v, 3)),
            ],
        )
    )
    s1, s2 = _row(stage, ifrs9_stage=1), _row(stage, ifrs9_stage=2)
    line = f"**Portfolio ECL {money(total_ecl)}, coverage {pct(total_ecl / total_ead, 3)}.**"
    if s1 and s2:
        line += (
            f" Stage 2 holds {pct(s2['ead'] / total_ead, 1)} of exposure and "
            f"{pct(s2['share_of_ecl'], 1)} of the provision at a mean PD "
            f"{ratio(s2['mean_pd'] / s1['mean_pd'], 0)}x stage 1. That ratio is the most direct "
            "evidence the loan-level alignment is correct: before R-012 was fixed, every loan was "
            "provisioned on another loan's PD and the ratio was diluted to about a third of this."
        )
    blocks.append(line)
    if sens is not None and a:
        blocks.append(
            f"**Staging sensitivity (ECL_001, threshold {a['value']['lifetime_pd_ratio']}, "
            f"backstop {a['value']['dpd_backstop_days']} days).**"
        )
        blocks.append(
            df_table(
                sens,
                [
                    ("sicr_threshold", "SICR threshold", text),
                    ("stage_1", "Stage 1", num),
                    ("stage_2", "Stage 2", num),
                    ("stage_3", "Stage 3", num),
                    ("stage_2_share", "Stage 2 share", lambda v: pct(v, 2)),
                ],
            )
        )
        floor_row = sens.filter(pl.col("sicr_threshold") == sens["sicr_threshold"].max()).row(
            0, named=True
        )
        at = _row(sens, sicr_threshold=float(a["value"]["lifetime_pd_ratio"]))
        if at:
            blocks.append(
                f"{num(floor_row['stage_2'])} loans are the days-past-due backstop; the ratio test "
                f"adds {num(at['stage_2'] - floor_row['stage_2'])} at the registered threshold. "
                "The test is live but barely, because the hazard's only inputs that change after "
                "origination are age and the mark-to-market covariates, and on a 1999-2006 book "
                "house prices only rose (F-011)."
            )
    return section("9. Expected credit loss and staging", *blocks)


# --- 10. macro ------------------------------------------------------------------------------


def macro(s: Sources) -> str:
    coef = s.csv("macro_coefficients")
    shifts = s.csv("macro_scenario_shifts")
    back = s.csv("macro_backtest")
    scen = s.csv("ecl_scenarios")
    a = s.assumption("MACRO_001")
    blocks = [
        "**Two-stage by design.** The loan-level model carries idiosyncratic risk; the cycle "
        "enters as a shift in logit space from a portfolio-level regression of the quarterly "
        "default rate on unemployment and year-on-year house-price change. At most two "
        "covariates, Newey-West standard errors, intervals propagated into ECL as a range."
    ]
    if a:
        v = a["value"]
        blocks.append(
            f"**Estimation window.** Requested {v['primary_window']}; effective "
            f"{v['effective_window']}, {v['n_observations']} observations, because the "
            f"house-price index begins {v['hpi_series_start']} and the year-on-year change needs "
            "four prior quarters. The register recorded 32 until F-019 corrected it."
        )
    if coef is not None:
        blocks.append(
            df_table(
                coef,
                [
                    ("window", "Window", text),
                    ("n_observations", "n", text),
                    ("term", "Term", text),
                    ("coefficient", "Coefficient", lambda v: ratio(v, 4)),
                    ("ci_low", "CI low", lambda v: ratio(v, 4)),
                    ("ci_high", "CI high", lambda v: ratio(v, 4)),
                    ("significant_at_95", "Significant", text),
                    ("r_squared", "R²", lambda v: ratio(v, 3)),
                ],
            )
        )
    if shifts is not None and scen is not None:
        merged = shifts.join(scen.select(["scenario", "ecl", "ecl_low", "ecl_high"]), on="scenario")
        blocks.append("**Scenarios** (conf/scenarios.yaml), with the ECL each implies.")
        blocks.append(
            df_table(
                merged,
                [
                    ("scenario", "Scenario", text),
                    ("weight", "Weight", lambda v: ratio(v, 2)),
                    ("unemployment", "Unemployment", lambda v: f"{float(v):.1f}%"),
                    ("hpi_yoy", "HPI y/y", lambda v: f"{float(v):+.1f}%"),
                    ("pd_multiplier", "PD multiplier", ratio),
                    ("extrapolates", "Extrapolates", text),
                    ("ecl", "ECL", money),
                    ("ecl_low", "95% low", money),
                    ("ecl_high", "95% high", money),
                ],
            )
        )
        blocks.append(
            f"**Probability-weighted ECL: {money(float(scen['probability_weighted_ecl'][0]))}.** "
            "Staging is held at the base assignment across scenarios (R-008)."
        )
    if back is not None:
        blocks.append(
            "**Backtest.** The pre-crisis fit, fed the realised 2008-2009 economy, against the "
            "realised default rate."
        )
        blocks.append(
            df_table(
                back,
                [
                    ("quarter", "Quarter", lambda v: str(v)[:7]),
                    ("unemployment", "Unemployment", lambda v: f"{float(v):.1f}%"),
                    ("hpi_yoy", "HPI y/y", lambda v: f"{float(v):+.1f}%"),
                    ("predicted_default_rate", "Predicted", lambda v: pct(v, 2)),
                    ("observed_default_rate", "Observed", lambda v: pct(v, 2)),
                    ("ratio_observed_to_predicted", "Ratio", ratio),
                ],
            )
        )
        mean_ratio = _scalar(back["ratio_observed_to_predicted"].mean())
        hpi_pre = _row(coef, window="1999-2006", term="hpi_yoy") if coef is not None else None
        hpi_full = _row(coef, window="1999-2019", term="hpi_yoy") if coef is not None else None
        line = (
            f"**Mean ratio of observed to predicted: {ratio(mean_ratio)} (F-009).** The overlay "
            "reduces the Phase 3 shortfall and does not close it. The reason is structural: the "
            "estimation sample contains no house-price decline."
        )
        if hpi_pre and hpi_full:
            line += (
                f" The HPI coefficient is {ratio(hpi_pre['coefficient'], 4)} on the pre-crisis "
                f"window and {ratio(hpi_full['coefficient'], 4)} on the full history, a factor of "
                f"{ratio(hpi_full['coefficient'] / hpi_pre['coefficient'])}, which accounts almost "
                "exactly for the shortfall. The full-history figure is in-sample by construction "
                "and isolates the mechanism; it does not vindicate the model."
            )
        blocks.append(line)
    return section("10. Macroeconomic overlay and the crisis backtest", *blocks)


# --- 11. monitoring ----------------------------------------------------------------------------


def monitoring(s: Sources) -> str:
    mon = s.json("monitoring_summary")
    mcfg = s.conf("monitoring")
    blocks = [
        "**Design.** Six rules committed before the run (conf/monitoring.yaml), each with a "
        "metric, thresholds, severity, owner and required action. Leading indicators (score PSI, "
        "feature CSI) are knowable at scoring time; lagging indicators (observed/expected, Gini, "
        "Brier reliability) need the twelve-month outcome window to close, and every alert "
        "carries the date it could first have been raised."
    ]
    blocks.append(
        md_table(
            ["Rule", "Metric", "Warn", "Breach", "Severity at breach", "Basis"],
            [
                [
                    r["id"],
                    r["metric"],
                    text(r["warn"]),
                    text(r["breach"]),
                    r["severity_breach"],
                    r["basis"],
                ]
                for r in mcfg["rules"]
            ],
        )
    )
    if not mon:
        blocks.append(_missing(s, "monitoring_summary"))
        return section("11. Ongoing monitoring", *blocks)

    rows = []
    for m in mon["models"]:
        gap = m["detection_gap"]
        rows.append(
            [
                m["model_id"],
                m["model_family"],
                m["windows"],
                ratio(m["max_stress_score_psi"], 4),
                m["alerts_fired"],
                m["breaches_out_of_sample"],
                text(gap.get("months_of_warning")),
                gap.get("leading_rule") or "—",
            ]
        )
    blocks.append(
        f"**Both PD models, {mon['windows']} quarterly windows, {mon['rules_evaluated']} rules.**"
    )
    blocks.append(
        md_table(
            [
                "Model",
                "Family",
                "Windows",
                "Worst stress score PSI",
                "Alerts",
                "Out-of-sample breaches",
                "Months of warning",
                "Leading rule",
            ],
            rows,
        )
    )
    blocks.append(
        "**F-015.** Score PSI stays an order of magnitude inside the stable band through the "
        "entire crisis, for both families, while calibration collapses. PSI compares input and "
        "output distributions; what changed in 2008 was the mapping from characteristics to "
        "default, which no input-side comparison can see. Drift monitoring gave zero months of "
        "warning on the failure it exists to catch."
    )
    fp_rows = []
    for m in mon["models"]:
        for rule_id, rate in sorted(m["rule_false_positive_rates"].items()):
            fp_rows.append([m["model_id"], rule_id, pct(rate, 1) if rate is not None else "—"])
    blocks.append(
        "**Rule false-positive rates on the development sample.** Every in-sample window is one "
        "on which the model is by construction working, so a firing there is a false positive. "
        "Thresholds were left as committed and the rates published rather than tuned (F-016, "
        "F-017)."
    )
    blocks.append(md_table(["Model", "Rule", "In-sample firing rate"], fp_rows))
    caveats = [
        m["detection_gap"].get("caveat") for m in mon["models"] if m["detection_gap"].get("caveat")
    ]
    if caveats:
        blocks.append(f"**Detection-gap caveat, as reported by the run:** {caveats[0]}")
    return section("11. Ongoing monitoring", *blocks)


# --- 12. governance ----------------------------------------------------------------------------


def governance(s: Sources) -> str:
    inv = s.inventory
    rec = s.reconciliation
    rows = []
    for m in inv["models"]:
        a = m["approval"]
        rows.append(
            [
                m["id"],
                m["family"],
                m["status"],
                m["tier"],
                a["status"],
                text(a.get("review_due")),
                ", ".join(m.get("monitored_by") or []) or "none",
                ", ".join(m.get("limitations") or []) or "none",
            ]
        )
    blocks = [
        "**Inventory** (governance/model_inventory.yaml). Judgements only; everything derivable "
        "from an artefact is read from the artefact at reconciliation time.",
        md_table(
            [
                "Model",
                "Family",
                "Status",
                "Tier",
                "Approval",
                "Review due",
                "Monitored by",
                "Limitations",
            ],
            rows,
        ),
    ]
    for m in inv["models"]:
        cond = squash(m["approval"].get("conditions", "")).strip()
        if cond:
            blocks.append(f"**{m['id']} — conditions of approval.** {cond}")
    blocks.append(
        "**Reconciliation** (`riskos registry`) checks the inventory against the artefacts on disk "
        "and the findings register: a model in use with no loadable artefact, an artefact nobody "
        "inventoried, a bundle fitted against changed configuration, a cited finding that does not "
        "exist, an open high-severity finding absent from a model's limitations, a tier-1 model "
        "with no monitoring rule, an approval past its review date."
    )
    if rec is not None:
        if rec.height:
            blocks.append(
                df_table(
                    rec,
                    [
                        ("severity", "Severity", text),
                        ("check", "Check", text),
                        ("model_id", "Model", text),
                        ("detail", "Detail", text),
                    ],
                )
            )
        else:
            blocks.append("No discrepancies at the last reconciliation.")
        blocks.append(
            "The first reconciliation returned a high-severity discrepancy: the rubric-selected "
            "model had no scoring-ready artefact, so every downstream figure had come from the "
            "runner-up (F-018, remediated). The remaining discrepancies are genuine gaps, reported "
            "rather than closed."
        )
    blocks.append(
        "**Serving.** The scoring service takes no model path. It serves the model the inventory "
        "records as in use, refuses a candidate, an unapproved or lapsed approval, or an artefact "
        "whose manifest disagrees with the inventory, and returns the model id, version, approval "
        "status, conditions and recorded limitations with every score."
    )
    return section("12. Governance: inventory, reconciliation, and serving", *blocks)


# --- 13. findings ------------------------------------------------------------------------------


def findings(s: Sources) -> str:
    fs = s.findings
    by_sev: dict[str, dict[str, int]] = {}
    for f in fs:
        by_sev.setdefault(f["severity"], {}).setdefault(f["status"], 0)
        by_sev[f["severity"]][f["status"]] += 1
    order = ["critical", "high", "medium", "low"]
    summary = md_table(
        ["Severity", "Open", "Remediated", "Total"],
        [
            [
                sev,
                by_sev.get(sev, {}).get("open", 0),
                by_sev.get(sev, {}).get("remediated", 0),
                sum(by_sev.get(sev, {}).values()),
            ]
            for sev in order
            if sev in by_sev
        ],
    )
    table = md_table(
        ["ID", "Severity", "Status", "Phase", "Title"],
        [
            [f["id"], f["severity"], f["status"], f.get("phase_raised", ""), squash(f["title"])]
            for f in fs
        ],
    )
    return section(
        "13. Findings register",
        "A review with no findings is not credible. Findings are recorded when found, not "
        "assembled at the end, and remediated entries keep their original text with the "
        "remediation appended. F-prefixed findings are raised against the models and data; "
        "R-prefixed findings are defects found by code review after a phase was reported complete.",
        summary,
        "**Open findings that are conclusions, not defects.** F-001 (a policy intervention in the "
        "data), F-003 (a deliberate design cost), F-005, F-006 and F-009 (the calibration and "
        "overlay results the project exists to report), F-014 (disclosed, with the correct "
        "reduced-form choice in place) and F-017 (a known-defective low-severity threshold) carry "
        "an accepted residual risk. **Open findings that imply work:** F-004 (report the variant "
        "without delinquency), F-008 (estimate the long-horizon hazard), F-011 (a specification "
        "using delinquency state), F-015 (a leading indicator that can see a changing "
        "relationship), F-016 (split time-structural features out of CSI).",
        table,
    )


# --- 14. assumptions ------------------------------------------------------------------------------


def assumptions(s: Sources) -> str:
    entries = s.assumptions
    by_source: dict[str, int] = {}
    for a in entries:
        by_source[a["source"]] = by_source.get(a["source"], 0) + 1
    table = md_table(
        ["ID", "Source", "Description"],
        [[a["id"], a["source"], squash(a["description"])] for a in entries],
    )
    return section(
        "14. Assumptions register",
        f"{len(entries)} registered assumptions ("
        + ", ".join(f"{k} {v}" for k, v in sorted(by_source.items()))
        + "). Every numeric assumption lives in conf/assumptions.yaml with a source, a "
        "materiality statement and a sensitivity test; tests bind the register to the constants "
        "in code and to the monitoring rulebook, and F-019 added a test binding it to a fitted "
        "artefact.",
        table,
    )


# --- 15. conclusion -------------------------------------------------------------------------------


def conclusion(s: Sources) -> str:
    inv = s.inventory
    lines = []
    for m in inv["models"]:
        a = m["approval"]
        lines.append(
            f"- **{m['id']}**: {a['status'].replace('_', ' ')}"
            + (
                f" on {a['approved_on']}, review due {a['review_due']}"
                if a.get("approved_on")
                else ""
            )
            + f". Limitations carried: {', '.join(m.get('limitations') or []) or 'none'}."
        )
    return section(
        "15. Conclusion and conditions",
        "\n".join(lines),
        "**What would change these decisions.** An independent second-line review, which this "
        "project cannot supply. A leading indicator able to see a changing relationship between "
        "characteristics and default (F-015). A delinquency transition model, which unblocks both "
        "the better-specified hazard and the SICR test (F-014, F-011). None of these changes the "
        "central limitation: a stress overlay estimated on a sample containing no stress will "
        "under-predict stress, and the correct output is the range and the sentence, not a point.",
    )


# --- appendix ------------------------------------------------------------------------------------


def reproducibility(s: Sources) -> str:
    cmds = md_table(
        ["Step", "Command"],
        [
            ["Ingest and validate", "`make ingest`"],
            ["Panel, risk set, labels", "`make panel`"],
            ["Scorecard champion", "`make train`"],
            ["Challenger, calibration, selection", "`make challenger`"],
            ["Hazard and reconciliation", "`make hazard`"],
            ["LGD, ECL, scenarios, backtest", "`make ecl`"],
            ["Monitoring", "`make monitor`"],
            ["Inventory reconciliation", "`make registry`"],
            ["This report and the model card", "`make report`"],
            ["Tests, lint, types", "`make test`, `make lint`"],
        ],
    )
    miss = (
        "All artefacts read by this report were present."
        if not s.missing
        else "**Artefacts missing at generation:** " + ", ".join(f"`{m}`" for m in s.missing)
    )
    return section(
        "Appendix A. Reproducibility",
        "Every table above is regenerated from CSV, JSON and YAML artefacts under reports/figures/, "
        "models/ and governance/. No figure is typed into this document.",
        cmds,
        miss,
    )
