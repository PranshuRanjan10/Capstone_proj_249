# CARE-Paddy — Project Specification

Paste this whole document into Kiro (or Antigravity) as the opening prompt. It is
written to be the requirements document, not a chat message. If a partial repository
is already open, treat it as the starting point and continue from it rather than
regenerating files.

---

## 0. Role and standing instructions

You are my technical co-researcher on a final-year B.Tech project. Two rules override
everything else:

1. **Never invent a result.** Do not write an accuracy, F1, R², percentage reduction,
   water saving, or yield figure that was not produced by code in this repository. If a
   number is a target, label it "hypothesis". If it came from a simulator, label it
   "simulation result". If a label was derived from a rule, label it
   "rule-derived / pseudo-label".
2. **Challenge me.** If part of this design is over-engineered, scientifically weak, or
   impossible to validate with the data available, say so before implementing it. A
   smaller system that genuinely works beats a large architecture that exists only in
   diagrams.

---

## 1. What the system is, in one sentence

CARE-Paddy forecasts what a rice field's water level will be over the next 1–3 days,
checks that forecast against the crop's current growth stage, and recommends whether to
irrigate, hold, drain, or defer to a human — refusing to automate when its own forecast
is too uncertain to trust.

Everything else is scaffolding around that sentence.

---

## 2. Why this framing, and what it replaces

The earlier design had a supervised model predicting "anaerobic risk", "crop stress
risk" and "water inefficiency risk". None of those are measurable quantities. Generating
them from rules applied to soil moisture, water level and rainfall, then training a model
on those same features, means the model just learns the rule back. The reported F1 would
measure how well a gradient-boosted tree memorises an if-statement. The question
"what is the ground truth for anaerobic risk?" would end the review.

**The fix is to change what is predicted, not which model predicts it.**

- The ML predicts **field water level**, a real, physical, forecastable quantity with a
  real target, a real error metric and real uncertainty.
- The decision layer is a **transparent agronomic policy**, not a model. It maps
  forecast water level plus crop stage onto MAINTAIN / IRRIGATE / DELAY / DRAIN / INSPECT.
- Stage-awareness becomes a **hard agronomic constraint** rather than a learned weight
  vector: safe-AWD guidance says never let the field dry down between panicle initiation
  and flowering, because that is when rice is most water-stress sensitive.
- Uncertainty gating becomes **meaningful**: the width of the forecast prediction
  interval. Wide interval → do not actuate → INSPECT.

This keeps every part of the project's identity — stage-aware, multimodal,
uncertainty-gated, explainable — and removes the circularity. It also makes
"false actuation rate" definable, which it currently is not.

### The environmental motivation

Continuously flooded paddies keep soil anaerobic, which is what methanogens need.
Alternate wetting and drying (AWD) breaks that. Wu et al. (2023) report 39.5% global
warming potential mitigation under frequent drainage versus continuous flooding, and
18.4% under mid-season drainage. Ma et al. (2025) describe the water table as an on-off
switch for CH₄ emission.

The gap is **not** "we don't know AWD works". The gap is that farmers under-adopt it
because drying the field at the wrong moment costs yield. A farmer burned once by a
badly timed drawdown will flood continuously forever, and rationally so.

**This project attacks adoption risk, not the science.** We are not claiming to reduce
methane. We are claiming to make safe AWD easier to follow. That framing is honest and
it is the one that survives questioning.

---

## 3. Architecture — five layers

```
  SENSING          ESP32: water level, soil moisture, air temp/humidity
     |
  CONTEXT          NASA POWER (historical) + Open-Meteo (forecast) + Sentinel-1 (later)
     |
  PHENOLOGY        cumulative GDD -> crop stage        [RULE-BASED, not ML]
     |
  FORECAST         quantile XGBoost -> water level at t+1/t+2/t+3, with 5/50/95 percentiles
     |
  DECISION + GATE  hard constraint -> uncertainty gate -> thresholds -> action
     |
  ACTUATION        pump/valve relay, or recommendation only
     |
  FEEDBACK         back to the field
```

### 3.1 Sensing layer

Three sensors. **Resist adding more.**

- Water level: ultrasonic or pressure/float sensor in a perforated field tube
- Soil moisture: capacitive probe at root depth
- Air temperature and humidity: DHT22

**No methane sensor.** MQ4 is a semiconductor combustible-gas sensor with heavy
temperature and humidity cross-sensitivity. It is not a flux instrument. Ouyang et al.
(2023) needed 23 eddy-covariance towers and still reached only NSE 0.59 on 8-day fluxes.
Claiming methane inference from an MQ4 invites exactly the question we do not want.

### 3.2 Context layer

Weather forecast and observed weather, plus satellite-derived inundation signals later.
This is what lets the system see rain coming rather than react after the field floods.

### 3.3 Phenology layer

Cumulative growing degree days from transplanting, base 10 °C, capped at 30 °C, mapped
to stage boundaries from a published rice calendar.

**Rule-based, deterministic, auditable. Not machine learning.** If asked why: GDD-based
staging is standard agronomy, and inventing an ML stage classifier would add error
without adding capability, on data that has no stage labels to train against.

### 3.4 Forecast layer — the actual machine learning

Input: current water level and soil moisture, their recent trends and lags, rainfall over
3 and 7 day windows, forecast rainfall, reference evapotranspiration, cumulative GDD,
stage index, and satellite indices when available.

Output: water level at t+1, t+2, t+3 — as a median plus 5th and 95th percentiles, so
every prediction carries an interval rather than just a point.

**Gradient-boosted trees with a quantile objective.** Justification, cited rather than
arbitrary:

- Wu et al. (2023): XGBoost best for CH₄ (R² = 0.754) and N₂O (R² = 0.762), ahead of SVR,
  RF, BPNN and 1D-CNN. They attribute the neural networks' weakness to dataset size.
- Ma et al. (2025): Adaboost > XGBoost > RF > MDS > BPNN > SVM and KNN, with SVM and KNN
  effectively useless (R² 0.045 and 0.053).
- Menefee et al. (2023): stacked ensemble r² = 0.87, and model success tracked training
  data volume.

Do **not** implement an MLP. We have less data than any of those studies.

### 3.5 Decision and gating layer

A pure function, evaluated in this exact order. Order matters and must not be rearranged.

1. **Hard agronomic constraint.** If the crop is between panicle initiation and
   flowering, drainage is forbidden regardless of what any model says. A model cannot
   override agronomy.
2. **Uncertainty gate.** If the 90% forecast interval is wider than the safety
   threshold, do not actuate. Issue INSPECT with the reason "prediction uncertainty
   exceeds safety threshold". Sensor drift, a stuck reading and unusual weather all widen
   the interval, so one gate catches the failure modes you would otherwise have to
   enumerate individually.
3. **Thresholds.** Safe AWD practice lets water fall to roughly 15 cm below the soil
   surface before re-flooding, monitored via a field water tube. Act on *approach* to the
   stage floor, not after it is breached — by the time the median forecast is below the
   floor the crop has already been stressed for a day.
4. **Otherwise MAINTAIN.**

Then attach a TreeSHAP explanation for the specific decision on screen.

---

## 4. Data

### 4.1 Sources to use

| Source | Purpose | Access |
|---|---|---|
| NASA POWER daily | historical weather, the main model forcing | `power.larc.nasa.gov/api/temporal/daily/point`, free, no key. Parameters: `T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, RH2M, ALLSKY_SFC_SW_DWN, WS2M` |
| Open-Meteo | forecast rainfall, and the historical-forecast archive | free, no key |
| SoilGrids 250m | percolation-rate and texture parameters | `rest.isric.org/soilgrids/v2.0/properties/query`, free |
| FAO-56 | rice Kc coefficients | published constants, hardcode with citation |
| Sentinel-1 SAR | inundation cross-check | Google Earth Engine, `COPERNICUS/S1_GRD` — Review-3 |
| RiceAtlas | sanity-check GDD staging | Laborte et al. 2017 |

**Critical design point.** Train on **forecast** rainfall, not observed. At deployment the
system only ever has a forecast. Training on observed rain and deploying on forecast rain
is a silent leakage bug: validation looks excellent and field performance collapses.
Open-Meteo's historical-forecast archive lets you reconstruct "what was the forecast on
day X" for past dates — that is the correct training feature.

**Use SoilGrids for simulator parameters, not as a daily ML feature.** Ouyang et al. flag
that static variables cause spatial overfitting. Say that this choice was deliberate.

### 4.2 If field data cannot be obtained

It does not stop the project, and this must be handled correctly rather than hidden.

The validation target is water level, which is governed by a water balance: rain in,
irrigation in, evapotranspiration out, percolation out, bund overflow. That is a physical
model with parameters that can be sourced (Kc by stage from FAO-56, percolation rates by
soil texture from SoilGrids). Driven with real weather for 10–20 real seasons at real
coordinates, plus realistic sensor noise, it produces a dataset where the ML has a
genuine, non-circular job: forecast a nonlinear, weather-forced dynamical system whose
equations it cannot see.

This is categorically different from the original label problem. The old design invented
a label and then predicted it. This design predicts a physically-determined quantity that
would be identical in kind if measured by a real sensor. **The simulator stands in for
the field, not for the answer.**

Two obligations follow:

1. **Say so unprompted, on a slide.** Exact wording in section 8.
2. **Run a sensitivity analysis.** Perturb percolation rate and Kc by ±30% and show that
   model ranking and decision-policy behaviour are stable. That demonstrates the
   conclusions are not artifacts of simulator parameter choices — precisely the objection
   a good reviewer raises.

Keep pursuing real data regardless. ICAR-NRRI Cuttack appears as flux site IN-CRRI in
Ouyang et al., so they run instrumented paddies. TNAU's Cauvery delta stations are
geographically convenient. Ask narrowly: "daily water level and soil moisture logs, one
plot, one season, for model validation, with acknowledgment" gets answered.
"We'd like to collaborate" does not. Even one real season used purely as a held-out test
set transforms Review-3.

---

## 5. Validation rules — non-negotiable

- **Leave-one-season-out cross-validation.** Never a random split. Daily field state is
  strongly autocorrelated and a random split leaks tomorrow into training, inflating every
  metric substantially. Note as a critical-reading point that Wu et al. used a random 7:3
  split — this shows you read the literature critically.
- **No feature may use a future observed value.** Forecast rainfall is the sole exception
  and is itself a degraded transform.
- **Seasons never straddle train and test.**
- **Fit any scaler on training data only.**
- **Always report a persistence baseline** (tomorrow = today). If persistence wins at
  t+1, say so. Reviewers trust people who report where their model is not needed and
  interrogate people who report uniform wins.
- **Report prediction-interval coverage.** Does the nominal 90% interval contain the truth
  90% of the time? Cheap to compute, and it is the metric that shows the uncertainty is
  real rather than decorative.

### False actuation rate, defined properly

- Apply the reference policy to the **observed** future state → reference actions
- Apply the same policy to the **forecast** state → system actions
- False actuation = fraction of system-issued IRRIGATE/DRAIN commands the reference policy
  would not have issued

Now it is a confusion matrix on a real comparison rather than an assertion.

### The single most important figure

False-actuation rate plotted against the fraction of decisions deferred to INSPECT. If
gating works, the curve falls as more decisions are deferred. It is the only figure that
proves the gate does something rather than merely existing.

---

## 6. Repository layout

```
CARE-Paddy/
├── config/config.yaml          every site + agronomy parameter, single source of truth
├── src/
│   ├── config.py               config loader
│   ├── fetch_power.py          NASA POWER downloader (run locally)
│   ├── weather.py              ET0 (Hargreaves-Samani), POWER loader, forecast degradation
│   ├── staging.py              GDD -> crop stage, rule-based
│   ├── water_balance.py        the physical simulator
│   ├── features.py             lags, rolling windows, leakage rules
│   ├── build_dataset.py        assemble multi-season master dataset
│   ├── train.py                quantile XGBoost vs RF vs persistence, LOSO
│   ├── decision.py             pure-function policy + uncertainty gate
│   ├── evaluate.py             false actuation, gating curve, figures
│   └── explain.py              TreeSHAP
├── tests/test_decision.py      unit tests; DRAIN must be impossible during flowering
├── dashboard/app.py            Streamlit
├── iot/
│   ├── sensor_sim.py           virtual ESP32 on the real interface
│   └── esp32/                  Arduino sketch
├── data/{raw,processed}/
├── outputs/{figures,metrics,models}/
└── docs/
```

---

## 7. Build order

Each step must leave something demonstrable, so that stopping anywhere still yields a demo.

1. Repo scaffold + `config.yaml` with coordinates, seasons, GDD boundaries, Kc, AWD floors
2. `fetch_power.py` — I run this locally; POWER is outside most sandboxes
3. Water balance simulator + sensor noise model
4. GDD staging + feature engineering with leakage rules enforced
5. Quantile XGBoost, leave-one-season-out, against persistence and RF baselines
6. Decision engine as a pure function, with unit tests
7. TreeSHAP + the gating curve figure
8. Streamlit dashboard
9. Virtual ESP32 publisher writing to the same interface as real hardware
10. Sensitivity analysis (±30% on percolation and Kc)

### Environment notes

- `pip install` may need `--break-system-packages`
- NASA POWER, Open-Meteo, SoilGrids and Google Earth Engine are outside most agent
  sandboxes. Either I run the fetch scripts and hand over CSVs, or the allowlist is
  extended.
- xgboost ≥ 2.0 supports `objective="reg:quantileerror"` with `quantile_alpha`. Use it.
  Do not hand-roll pinball loss.

---

## 8. Honesty requirements

### Never fabricate

accuracy, F1, AUROC, methane reduction, water savings, false-actuation reduction, yield
preservation, statistical significance, field performance, real-world validation.

### Required labels

| If | Say |
|---|---|
| result came from the simulator | "Simulation result" |
| label came from a rule | "Rule-derived / pseudo-label" |
| SHAP is shown | "Feature attribution" — not causal inference |
| stage weights chosen by hand | "Stage-conditioned heuristic weights" |

### Scientific cautions

MQ4 ≠ methane flux instrument. SHAP ≠ causal inference. Correlation ≠ causation.
Simulated labels ≠ ground truth. Model prediction ≠ measured emissions. A hypothesis ≠ a
result. A target reduction ≠ an achieved reduction. Do not claim "first" unless a
comprehensive literature review actually supports it — it does not here. Stage-aware
irrigation DSS exists; uncertainty-gated automation exists. Claim integration, not primacy.

### The slide that must appear verbatim

> Field hydrology is currently generated by a physically-based water-balance model driven
> by observed weather for [location, seasons]. The ML layer is a forecasting surrogate
> validated against that simulator, not against field observations. The ESP32 replaces the
> simulator without any change to the pipeline interface. No methane has been measured.

Panels accept this readily at Review-2. What they do not forgive is discovering it during
Q&A.

---

## 9. Anticipated panel questions and the answers

**"What is your ground truth?"** Water level from the water balance. The ML never sees the
balance equations, only forcing and lagged state.

**"Isn't this just a threshold rule?"** The policy is, deliberately — that is what makes it
auditable. The ML is the forecast that feeds it. Acting on a 3-day forecast instead of
today's reading is what buys the farmer lead time before rain, and that is the contribution.

**"Why not deep learning?"** Sample size and tabular structure. Cite Ma et al.'s ranking and
Wu et al.'s XGBoost result.

**"Where's the methane?"** Not measured, not claimed. The system targets AWD adherence;
emissions reduction is a downstream, literature-supported consequence, not our result.

**"Why is drainage blocked during flowering?"** Because rice is most water-stress sensitive
from panicle initiation through flowering. It is a hard constraint that no model output can
override, and there is a unit test asserting it.

---

## 10. First task

Do not write code yet.

Read this specification and produce:

1. A `requirements.md` capturing the functional and validation requirements above
2. A `design.md` with the module boundaries and data contracts between layers
3. A `tasks.md` sequencing the build order in section 7

Then tell me, before implementing anything: which parts of this design you think are
over-engineered or unverifiable given the data available, and what you would cut.
