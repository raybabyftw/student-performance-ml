# Student Performance ML

Predicting which students are likely to be high-performing using a clean, reproducible ML/DL workflow.  
This project walks from EDA → preprocessing → baseline → deep learning (Sequential MLP + Wide&Deep) → analysis, with a focus on clarity, reproducibility, and responsible use.

---

## Why this project?
- Explore an end-to-end modelling pipeline on a real tabular dataset.
- Compare classical baselines with modern deep learning on small data.
- Practice production-minded hygiene: pipelines, seeded runs, ablations, and documentation.

## Dataset
- **Source:** Student Performance dataset (UCI Machine Learning Repository).
- **Target:** Binary label indicating “high performance” (derived from grade columns).
- **Features:** Socio-demographic, school, lifestyle, and study-related attributes.
- **Note:** `data/student-mat.csv` is included for convenience; if you reuse it, check the original dataset license/terms.

## Approach (high level)
- **EDA:** Distributions, class imbalance, correlations, and leakage checks.
- **Preprocessing:** `ColumnTransformer` for one-hot/label encoding + scaling; **stratified** train/val/test split.
- **Models:** Logistic regression baseline; Keras dense networks (Sequential MLP, Functional **Wide&Deep**).
- **Training:** Class weighting for imbalance, EarlyStopping, checkpointing, seeded runs, and ablations (e.g., wide branch on/off).
- **Evaluation & Analysis:** Consolidated plots/tables and notes on limitations, bias, and interpretability.
