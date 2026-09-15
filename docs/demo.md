# RiskOS · a five-minute walkthrough

**Question:** can a mortgage model rank risk well while underestimating defaults?

**Answer:** yes. Both models under-predict crisis defaults, while score drift stays
below its warning threshold. This walkthrough uses saved results from the Freddie
Mac mortgage sample; it does not retrain models.

## 1 · Set the scene — 1 minute

Train on 1999–2006 observations, then test on later periods with separate loans.
The target is default in the next 12 months.

![Default-rate timeline showing training, crisis, and later test periods](../reports/figures/default_rate_by_quarter.png)

**Takeaway:** “The crisis tests whether relationships learned before 2007 still hold.”

The blank interval is outside the evaluation splits. The 2019 observation windows
extend into 2020, when forbearance affects delinquency-based labels; that later
period needs separate interpretation.

## 2 · Compare the models — 1 minute

Run `make evaluate` from the repository root after `make setup`, or use this saved
comparison. AUC measures ranking; observed / expected defaults measures calibration.
A ratio of 1 means the predicted total matches the observed total.

| Crisis period, 2007–09 | AUC | Observed / expected defaults |
| --- | ---: | ---: |
| Logistic scorecard | 0.819 | 3.21× |
| LightGBM challenger | 0.832 | 2.80× |

**Takeaway:** “LightGBM improves both measures, but both models substantially
under-predict defaults.”

Source: [comparison CSV](../reports/figures/champion_challenger_metrics.csv),
uncalibrated models, `oot_stress` rows. These are whole-period averages.

## 3 · Explain the distinction — 1 minute

![Scorecard ranking and calibration across evaluation periods](../reports/figures/discrimination_vs_calibration.png)

**Takeaway:** “The scorecard can still put riskier borrowers above safer borrowers,
while assigning probabilities that are too low.”

This chart shows the **scorecard only**. Gini is another ranking measure
(`Gini = 2 × AUC − 1`). The right-hand bars show observed defaults divided by
expected defaults. The dashed line at 1 is the target.

## 4 · Show the monitoring gap — 2 minutes

![Quarterly calibration deteriorates during the crisis while score PSI remains below warning](../reports/figures/monitoring_blind_spot.png)

**Takeaway:** “Score distributions barely shift during the crisis, but prediction
accuracy deteriorates. We need to monitor outcomes as well as score drift.”

PSI compares score distributions with the development sample. It stays below the
0.10 warning threshold during the shaded crisis period; it crosses that threshold
earlier in training. The top panel shows individual quarters, so its peak is
higher than the whole-period ratios in step 2.

These curves are retrospective: the 12-month default outcome becomes available
12 months after the observation. Calibration monitoring would therefore detect
the deterioration with a delay.

## Close

**The takeaway:** model evaluation needs ranking, calibration, and monitoring
that accounts for when outcomes become observable.

This is an educational experiment on U.S. mortgages, with developer validation.
It is not ready for lending or provisioning. ECL, macro scenarios, inventory,
and serving are optional extensions to this core demonstration.

## Backup for questions

- [Reliability chart](../reports/figures/reliability_by_split.png): calibration by risk bin, with uncertainty intervals.
- [Selection memo](../reports/model_selection_memo.md): why the comparison rubric preferred LightGBM; this is distinct from inventory deployment status.
- [Monitoring reference](06-monitoring.md): thresholds, false alarms, and outcome delay.
- [Full validation report](../reports/validation_report.md): methodology and limitations.
- [Reproduction guide](reproduce.md): inputs and commands for rebuilding results.
