Mission Health — Explainable Diabetes Risk Screening

ML for Social Good Ensemble Challenge. The project uses 2015 US BRFSS survey indicators to prioritize adults for confirmatory diabetes assessment in a resource-constrained screening workflow.
Submission at a glance

    Mission domain: Health
    Beneficiaries: adults who may have undetected diabetes, especially people facing access/cost barriers
    Unit of analysis: one BRFSS survey respondent
    Target: Diabetes_binary (source-defined present diabetes status)
    Data used: realistic imbalanced binary file, 253,680 raw rows and 21 predictors
    Leakage control: exact deduplication before a seeded stratified 70/15/15 split; all learned preprocessing is inside CV pipelines
    Baseline: balanced Logistic Regression
    Ensembles: tuned Random Forest (bagging), tuned AdaBoost (boosting), heterogeneous soft Voting
    Primary selection metric: validation average precision (PR-AUC)
    Final model: soft Voting; validation-only F2 threshold 0.425
    Explainability: model-agnostic permutation SHAP at global and individual levels
    Demo: Streamlit form and JSON CLI

The main assessment artifact is the fully executed Jupyter notebook. Reusable code lives in src/ so the notebook, training job, CLI, and demo share exactly the same serialized preprocessing and model.
Definitive results

All values below use the same untouched 34,422-row test partition. Classification metrics in the comparison use the default 0.50 threshold.
Model 	Type 	ROC-AUC 	PR-AUC 	F1 	Precision 	Recall
Logistic Regression 	Baseline 	0.8103 	0.4147 	0.4539 	0.3227 	0.7649
Random Forest 	Bagging 	0.8128 	0.4299 	0.4535 	0.3224 	0.7643
AdaBoost 	Boosting 	0.8077 	0.4141 	0.4437 	0.3121 	0.7670
Soft Voting 	Heterogeneous ensemble 	0.8135 	0.4303 	0.4517 	0.3187 	0.7751

Soft Voting improves PR-AUC over the baseline by 0.0156 absolute. This satisfies the comparison objective but is a modest gain, not a clinical breakthrough.

At the recall-oriented 0.425 threshold selected only on validation F2, the final model has recall 0.8532, precision 0.2851, specificity 0.6136, F2 0.6100, 4,492 true positives, 773 false negatives, 11,265 false positives, and 17,892 true negatives. The workload implied by the false positives is why this can only be a first-stage screen.
Repository map

.
├── notebook/Mission_Health_Diabetes.ipynb  # executed Q1–Q5 walkthrough
├── dataset/                                # supplied UCI CSV files
├── src/
│   ├── config.py                           # schema, paths, labels, valid ranges
│   ├── preprocessing.py                    # validation, features, fold-safe transforms
│   ├── modeling.py                         # baseline, bagging, boosting, voting
│   └── evaluation.py                       # metrics, thresholding, subgroup audit
├── train.py                                # full reproducible training/evaluation/SHAP
├── app.py                                  # Streamlit live demo
├── predict.py                              # JSON command-line prediction
├── examples/synthetic_record.json          # non-personal demo input
├── artifacts/                              # serialized pipeline, metadata, demo result
├── results/figures/                        # EDA, evaluation, fairness, SHAP PNGs
├── results/tables/                         # audits, CV, metrics, fairness, SHAP CSV/JSON
├── MODEL_CARD.md                           # intended use and ethical limitations
├── VIDEO_SCRIPT.md                         # required 0:00–3:00 pitch/demo plan
├── tests/test_pipeline.py                  # preprocessing/prediction tests
└── requirements.txt                        # pinned full-run environment

Reproduce from a clean environment

Python 3.12 is recommended. From the project root:

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python train.py

The full command reads the raw CSV, rebuilds the audit, splits, performs two 3-fold grid searches, fits all candidates, selects using validation PR-AUC, selects the threshold using validation F2, evaluates the untouched test set, audits subgroups, generates SHAP explanations, and serializes the winning pipeline. On the recorded environment it took about 6.8 minutes. Hardware changes may alter runtime but not the seeded splits.

For a fast integration check that does not replace the official artifacts:

python train.py --quick --skip-shap

That command overwrites artifacts with quick-mode results; run python train.py again before submission.
Open or re-execute the notebook

jupyter lab notebook/Mission_Health_Diabetes.ipynb

The committed notebook already contains the definitive executed outputs. Its optional RUN_FULL_TRAINING cell is False so opening the notebook is instant. Set it to True to invoke train.py from the notebook.
Run the live demo

streamlit run app.py

Open the local URL printed by Streamlit, retain the default synthetic record for the video, and click Run screening. The app reports a risk-ranking score, the validation-selected screen decision, limitations, and the saved local SHAP explanation for that exact default record.

The Choose a synthetic test case menu also provides healthy, lifestyle-risk, borderline-negative, borderline-positive, healthcare-access, and high-burden examples. The SHAP chart remains specific to the default video case.
Run the command-line demo

python predict.py --input examples/synthetic_record.json

Expected full-run output: risk score approximately 0.726, threshold 0.425, and screen_positive: true. Tiny floating-point differences can occur across platforms.
Data audit and design decisions

    The source file declares no missing values; the supplied CSV also has zero missing cells. Imputation remains in the pipeline for robust inference and is fitted only on training folds.
    24,206 exact duplicate rows are removed before splitting because no respondent identifier is supplied. This prevents identical records from crossing partitions, but may underweight distinct people with identical answers.
    Values match the source-coded ranges. BMI 12–98 is unusual but possible; these cases are retained. Robust scaling controls leverage without erasing high-risk respondents.
    Source binary categories and engineered BMI group are one-hot encoded. Ordered BRFSS variables retain their order and are robust-scaled.
    Domain features are BMI group, combined unhealthy days, cardiovascular burden, protective lifestyle score, access barrier, and BMI×age-code interaction.
    Balanced class/sample weights address the minority outcome. No SMOTE is used because synthetic mixtures of survey/health attributes can be implausible.
    The voting model uses tuned base hyperparameters and no meta-learner, so it does not learn from in-sample base predictions.

Explainability and fairness summary

Global SHAP ranks general health, high blood pressure, BMI, age, and high cholesterol as the most influential raw inputs on average. For the default synthetic record, BMI 34, high blood pressure, general health 3, and high cholesterol contribute most strongly upward. SHAP describes the model, not clinical causality.

At threshold 0.425, recall is 0.848 for source-coded females and 0.859 for males, while false-positive rates are 0.364 and 0.416. Age gaps are larger: recall is 0.513 for ages 18–44 and 0.908 for 65+, while false-positive rates are 0.078 and 0.628. These descriptive metrics do not prove fairness. See MODEL_CARD.md and results/tables/fairness_subgroups.csv.
Dataset source and citation

Dataset page: UCI CDC Diabetes Health Indicators
DOI: 10.24432/C53919

Suggested citation:

    CDC Diabetes Health Indicators [Dataset]. (2017). UCI Machine Learning Repository. https://doi.org/10.24432/C53919.

UCI states that the dataset was created to study relationships between lifestyle and diabetes in the US, that each instance represents a participating person, and that sex, income, and education may be sensitive.
Acknowledgements and academic integrity

    Data: CDC/BRFSS dataset distributed through the UCI Machine Learning Repository.
    Software: Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn, SHAP, Streamlit, joblib, Jupyter, and pytest.
    All project-specific code, analysis text, tables, and plots in this repository were produced for this submission. No external figures or copied implementation snippets are included.
    If this repository is extended with borrowed code or media, add the author, URL, license, and exact location here before submission.
