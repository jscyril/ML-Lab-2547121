# Model Card — Mission Health Diabetes Screening

## Model and decision

The released artifact is a scikit-learn pipeline containing input validation, deterministic feature engineering, imputation, encoding/scaling, and a soft-voting classifier. The voters are balanced Logistic Regression, tuned balanced Random Forest, and tuned weighted AdaBoost. The voting weights are 1:2:1. Hyperparameters are selected with 3-fold stratified CV on the training partition; the model family is selected by validation PR-AUC; the operating threshold is selected by validation F2.

The output is a **risk-ranking score** for source-defined current diabetes status. Because balanced weights alter the fitted prior and no calibration study was conducted, it must not be described as a literal probability of present or future diabetes.

## Intended use

- Classroom demonstration of an end-to-end, explainable ensemble workflow.
- Research prototype for prioritizing voluntary confirmatory assessment when follow-up resources are limited.
- Human-reviewed first-stage screening only, after prospective validation on the intended local population.

## Prohibited use

- Diagnosis, treatment, medication, emergency decisions, or replacing laboratory tests.
- Denying care, insurance, employment, credit, or public benefits.
- Autonomous outreach or action without a trained human reviewing context and uncertainty.
- Use on children, pregnant people, or a new country/clinic/year without a suitable validation study.
- Interpreting SHAP values or feature changes as causal clinical advice.

## Data

The input is the imbalanced binary file from the UCI *CDC Diabetes Health Indicators* dataset, derived from 2015 US BRFSS survey data. After removing 24,206 exact duplicate rows, 229,474 rows are split 70%/15%/15% using stratification and seed 42. The source includes 21 self-reported or coded health, access, behavior, and demographic predictors.

There are no direct identifiers in the provided files. Sex, age, education, income, disability and health indicators are nevertheless sensitive and can enable profiling or re-identification when linked with other data.

## Performance

On the untouched 34,422-row test set:

- ROC-AUC: 0.8135
- PR-AUC: 0.4303
- Default 0.50 threshold: precision 0.3187, recall 0.7751, F1 0.4517
- Validation-selected 0.425 threshold: precision 0.2851, recall 0.8532, F2 0.6100, specificity 0.6136
- Confusion counts at 0.425: TN 17,892; FP 11,265; FN 773; TP 4,492
- PR-AUC gain over logistic baseline: 0.0156 absolute

The test set estimates technical discrimination on similar historical survey data. It does not establish clinical benefit, transportability, prospective safety, or causal validity.

## Explainability

Model-agnostic permutation SHAP is computed on the final full pipeline in the original 21-feature space. Global mean absolute SHAP highlights general health, high blood pressure, BMI, age, and high cholesterol. A local waterfall explains the non-personal default demo record. Explanations can be unstable under correlated inputs, describe model behavior rather than truth, and must not be treated as intervention recommendations.

## Subgroup findings

At threshold 0.425:

| Group | n | Recall/TPR | FPR | Precision |
|---|---:|---:|---:|---:|
| Female | 19,235 | 0.848 | 0.364 | 0.280 |
| Male | 15,187 | 0.859 | 0.416 | 0.291 |
| Age 18–44 | 7,253 | 0.513 | 0.078 | 0.247 |
| Age 45–64 | 14,508 | 0.839 | 0.363 | 0.295 |
| Age 65+ | 12,661 | 0.908 | 0.628 | 0.281 |
| Higher income codes 5–8 | 25,747 | 0.823 | 0.339 | 0.262 |
| Lower income codes 1–4 | 8,675 | 0.903 | 0.547 | 0.328 |

Different prevalence, label quality and score distributions contribute to these gaps. The audit is descriptive, has no uncertainty intervals, and covers only broad single attributes. It is not evidence of equalized odds or overall fairness.

## Risks and mitigations

- **False negative:** delayed assessment/care. Mitigation: recall-oriented threshold, routine screening remains available, clinician override, monitor misses.
- **False positive:** anxiety, cost, stigma and overloaded clinics. Mitigation: clearly label as screening, confirm with standard tests, tune threshold to real capacity, monitor referral yield.
- **Historical/measurement bias:** access affects whether diabetes was diagnosed and how survey items were answered. Mitigation: audit label process, collect local prospective data, involve affected groups, compare with standard workflows.
- **Dataset shift:** 2015 US survey patterns may not match current India or another clinic. Mitigation: external and temporal validation, calibration, drift monitoring, stop-use triggers.
- **Privacy:** sensitive health/demographic data can harm people if exposed. Mitigation: minimization, consent, encryption, role-based access, audit logs, retention limits, deletion and incident response.
- **Automation bias:** a plausible score may be overtrusted. Mitigation: mandatory trained reviewer, display uncertainty/limitations, document overrides, prohibit autonomous adverse action.

## Deployment gate

Do not deploy until owners define the clinical pathway and capacity, complete prospective external validation, assess calibration and confidence intervals, conduct intersectional fairness and privacy reviews, obtain governance/ethics approval, train users, establish monitoring and incident response, and specify who can pause or retire the model.

