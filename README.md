# Disease Prediction Using Machine Learning

An **educational** multiclass machine learning experiment comparing Random Forest, Logistic Regression, SVM, and Gaussian Naive Bayes, with a Flask demo.

> **Not a medical diagnostic system.** The models have not been independently clinically validated. Do not use results to make healthcare decisions.

## Dataset
`disease_diagnosis.csv`: 2,000 records across five labels (Healthy, Cold, Flu, Bronchitis, Pneumonia). Source, collection method and label-generation rules have **not been verified**. The model uses symptoms, vitals, age, and gender. Patient ID and Treatment Plan are excluded; Severity is deliberately excluded from the final model because of potential target leakage.

The CSV is supplied in the [review bundle](../../releases) if a release is later published, or from the separately provided local project ZIP. **It is not currently committed to this repository.**

## Evaluation
80/20 stratified train/test split (random_state=42). For model comparison: 3-fold StratifiedKFold and GridSearchCV optimize macro F1. Training-fold-only feature fitting, Yeo-Johnson PowerTransformer, StandardScaler and SMOTE are included inside an imbalanced-learn Pipeline. The untouched test partition is not SMOTE-resampled.

| Model | Held-out accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 99.75% | 0.9865 |
| Logistic Regression | 93.00% | 0.8548 |
| SVM | 94.00% | 0.8692 |
| Gaussian Naive Bayes | 79.25% | 0.7811 |

Results are specific to this dataset and do **not** indicate clinical diagnostic accuracy. A simple decision tree also performed unusually well in exploratory sanity checks, suggesting possible rule-based label patterns. Further independent validation is required.

## Files
- `disease_leakage_free_pipeline.py`: data preparation, four-model CV evaluation, and severity ablation.
- `train_deployment_model.py`: trains the final educational Random Forest pipeline and exports the full fitted model.
- `app.py`, `templates/index.html`: Flask form and prediction API.
- `requirements.txt`: Python dependencies.
- `holdout_evaluation.json`: saved metrics for the educational deployment model.
- `disease_type_prediction.ipynb` and original `.pkl` files: **legacy experiment**, retained for history. They are not compatible with the corrected Flask app.

## Local setup
Python 3.11 or 3.12 recommended. Download the verified `disease_diagnosis.csv` into this folder, then:

```bash
python -m pip install -r requirements.txt
python train_deployment_model.py
python app.py
```

Open http://127.0.0.1:5000. The training script creates `educational_disease_pipeline.joblib` and regenerates `holdout_evaluation.json`.

**Important:** The fitted `educational_disease_pipeline.joblib` file is not committed to GitHub. You must train it locally (or copy it from your trusted local project bundle) before running `app.py`. Never load untrusted joblib/pickle files.

## Limitations
- Dataset provenance and labeling rules are unclear.
- Unusually high accuracy may result from simple synthetic/rule-based patterns.
- No external clinical validation.
- No production deployment or security hardening.

## Author
[Syble Jan](https://github.com/syblejan)
