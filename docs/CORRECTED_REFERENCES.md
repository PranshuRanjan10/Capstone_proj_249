# Corrected references and Literature Review slides

Fix this before anything else in the deck. Three of the four citations in Review-1
carry the right journal and volume but the wrong author, title, year and pages, and
two Literature Review slides describe findings that appear in none of the four papers
you actually have. A panel member who opens one DOI finds this in about thirty seconds,
and at that point the conversation stops being about your architecture.

The corrected versions are stronger for your argument than the originals were.

---

## Replace the reference list with this

[1] D. Menefee, T. O. Lee, K. C. Flynn, J. Chen, M. Abraha, J. Baker, and A. Suyker,
"Machine learning algorithms improve MODIS GPP estimates in United States croplands,"
*Frontiers in Remote Sensing*, vol. 4, art. 1240895, 2023. doi:10.3389/frsen.2023.1240895

[2] Z. Ouyang, R. B. Jackson, G. McNicol, E. Fluet-Chouinard, B. R. K. Runkle, D. Papale,
*et al.*, "Paddy rice methane emissions across Monsoon Asia," *Remote Sensing of
Environment*, vol. 284, art. 113335, 2023. doi:10.1016/j.rse.2022.113335

[3] Q. Wu, J. Wang, Y. He, Y. Liu, and Q. Jiang, "Quantitative assessment and mitigation
strategies of greenhouse gas emissions from rice fields in China: A data-driven approach
based on machine learning and statistical modeling," *Computers and Electronics in
Agriculture*, vol. 210, art. 107929, 2023. doi:10.1016/j.compag.2023.107929

[4] L. Ma, Q. Yu, Y. Cui, B. Liu, W. Tian, B. Liao, L. Liu, W. Dong, and J. Huang,
"Improving gap-filling performance for CH4 fluxes of eddy covariance data by combining
marginal distribution sampling and machine learning algorithm over paddy fields,"
*Journal of Cleaner Production*, vol. 510, art. 145615, 2025.
doi:10.1016/j.jclepro.2025.145615

---

## Replace the four Literature Review slides with these

### Slide: Why AWD is the mitigation lever

Wu et al. (2023) assembled 782 CH4 and 679 N2O observations from 118 published field
studies across 19 Chinese provinces. Frequent drainage delivered the largest global
warming potential reduction at **39.5% versus continuous flooding**, with mid-season
drainage second at **18.4%**. Irrigation mode ranked as the single most important
feature for CH4 emissions in their XGBoost model.

Use this instead of the current "30-50% methane reduction" claim. It is sourced, it is
specific, and it belongs to the literature rather than to your results.

### Slide: Water table is the physical switch

Ma et al. (2025) treat the presence or absence of a water table as an **on-off switch**
for CH4 emission in paddy fields, and show that substituting that switch for a
continuous soil-moisture variable improves CH4 flux gap-filling at five of eight sites,
with the improvement scaling with the number of wetting and drying cycles in the season
(R-squared 0.875 against cycle count).

This is your justification for building the entire system around water level as the
central state variable.

### Slide: Tree ensembles, not neural networks

Two independent studies on tabular agricultural data converge:

- Wu et al. (2023): XGBoost best for CH4 (R2 = 0.754, RMSE = 0.485 kg/ha) and N2O
  (R2 = 0.762), ahead of SVR, RF, BPNN and 1D-CNN. They attribute the neural networks'
  weakness to insufficient dataset size.
- Ma et al. (2025): Adaboost > XGBoost > RF > MDS > BPNN > SVM and KNN. SVM and KNN
  were effectively useless (R2 of 0.045 and 0.053).

Menefee et al. (2023) reach r2 = 0.87 with a stacked ensemble on MODIS GPP, and note
that model success tracked training-data volume.

This is why CARE-Paddy uses gradient-boosted trees. It is a cited choice, not an
arbitrary one, and it pre-empts the "why not deep learning?" question.

### Slide: The gap our ground sensors fill

Ouyang et al. (2023) upscaled CH4 flux from 23 eddy-covariance paddy sites across
Monsoon Asia, reaching Nash-Sutcliffe efficiency of 0.59 for 8-day fluxes and 0.69 for
site means. Critically, they state that **water table depth and soil moisture are
important CH4 controlling variables that they could not include**, because the data
were missing across sites and no gridded product existed for upscaling.

That is a stated gap in the literature that field-level IoT addresses directly. It is a
much better framing than "first to unify", which you cannot support.

Their uncertainty method is also your citable precedent: **500 bootstrapped random
forests** producing a standard deviation around each prediction.

---

## Claims to delete or relabel

| Current claim | Replace with |
|---|---|
| "reduce methane emissions by 30-50%" | "Wu et al. (2023) report 39.5% GWP mitigation under frequent drainage versus continuous flooding. Our objective is to improve adherence to AWD scheduling. We have not measured methane." |
| ">20% false actuation reduction" | Label as **hypothesis**, not result, until the gating curve is produced |
| "First to unify multimodal fusion, phenological conditioning and uncertainty gating" | "We integrate three established components; we do not claim primacy." Stage-aware irrigation DSS exists, and so does uncertainty-gated automation. |
| "Learned phenological embeddings" | "Stage-conditioned heuristic thresholds, GDD-derived" — you do not have the data to identify a learned gating network, and a reviewer will know it |
| Patent / journal-target slides | Remove from Review-2. They read as substitutes for results. |
| MQ4 methane sensing | Remove. MQ4 is a semiconductor combustible-gas sensor with heavy temperature and humidity cross-sensitivity; it is not a flux instrument. Ouyang et al. needed 23 eddy-covariance towers and still landed at NSE 0.59 on 8-day fluxes. |

---

## The honesty slide — put this on screen verbatim

> Field hydrology is currently generated by a physically-based water-balance model
> driven by observed weather for the Cauvery delta, across 20 seasons (Kuruvai and
> Samba, 2015-2024). The ML layer is a forecasting surrogate validated against that
> simulator, not against field observations. The ESP32 replaces the simulator without
> any change to the pipeline interface. No methane has been measured.

Panels accept this readily at Review-2. What they do not forgive is discovering it
during Q&A.
