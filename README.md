# 🩺 Disease Prediction Using Machine Learning

### End-to-End ML Classification | Leakage-Aware Evaluation | Flask Web Application

**Tech Stack:** Python • Scikit-learn • Pandas • NumPy • Random Forest • SMOTE • GridSearchCV • Flask

An end-to-end machine learning project that classifies five disease-related categories from patient symptoms, demographic information, and physiological measurements.

The project demonstrates a complete ML workflow: **data preprocessing, feature engineering, handling class imbalance, comparing four classification models, hyperparameter tuning, leakage-aware evaluation, and Flask integration**.

The original experiment was subsequently improved to address data leakage and strengthen the reliability of the evaluation methodology.

> **Disclaimer:** This is an educational ML demonstration, not a clinically validated medical diagnostic system. Predictions must not be used for healthcare decisions.

---

## 1. Project Highlights

| Component | Details |
|---|---|
| Problem Type | Multiclass Classification |
| Dataset | 2,000 records, 13 original columns |
| Target Classes | Healthy, Cold, Flu, Bronchitis, Pneumonia |
| Models Evaluated | Random Forest, Logistic Regression, SVM, Gaussian Naive Bayes |
| Data Splitting | Stratified 80/20 |
| Cross-Validation | GridSearchCV with StratifiedKFold |
| Class Imbalance | SMOTE |
| Selected Model | Random Forest (without Severity) |
| Recorded Test Accuracy | **99.75%** |
| Recorded Test Macro F1 | **0.9865** |
| Application | Flask Web Interface |

**Key achievement:** Built and evaluated a leakage-aware multiclass classification pipeline and integrated a trained model into a locally tested Flask application.

**Evaluation note:** High accuracy on this particular dataset does not imply reliable performance on independent clinical data.

---

## 2. Problem Statement and Objectives

### Problem Statement

Structured health-related datasets contain a combination of numerical measurements and categorical symptoms. Transforming these inputs into features suitable for machine learning requires careful preprocessing and evaluation.

This project investigates the classification of diagnosis labels using patient health indicators.

### Objectives

- Clean and preprocess structured patient records
- Engineer numerical features from blood pressure readings
- Encode categorical symptoms using multi-hot encoding
- Apply appropriate numerical transformations
- Address imbalanced classes using SMOTE
- Compare four supervised classification algorithms
- Perform hyperparameter tuning using GridSearchCV
- Evaluate accuracy, precision, recall, F1-score and confusion matrices
- Correct preprocessing and cross-validation leakage
- Build a Flask-based inference interface

---

## 3. Dataset and Exploratory Analysis

### Dataset Overview

**File:** `disease_diagnosis.csv`

The dataset contains 2,000 records with 13 original columns and five diagnosis categories.

| Diagnosis | Records | Percentage |
|---|---:|---:|
| Healthy | 1,167 | 58.35% |
| Bronchitis | 334 | 16.70% |
| Flu | 292 | 14.60% |
| Cold | 163 | 8.15% |
| Pneumonia | 44 | 2.20% |
| **Total** | **2,000** | **100%** |

The dataset is imbalanced. Healthy records represent the majority class, while Pneumonia is the smallest category.

### Input Features

**Demographic features**
- Age
- Gender

**Symptoms**
- Symptom_1
- Symptom_2
- Symptom_3

**Physiological measurements**
- Heart_Rate_bpm
- Body_Temperature_C
- Blood_Pressure_mmHg
- Oxygen_Saturation_%

**Additional dataset columns**
- Patient_ID
- Severity
- Treatment_Plan

**Prediction target:** `Diagnosis`

### Data Quality Findings

- Missing values: 0
- Exact duplicate records: 0
- Five diagnosis categories
- Strong class imbalance

Patient_ID and Treatment_Plan are excluded from model training.

Severity is also excluded from the selected corrected model because of potential target leakage.

**Dataset provenance:** The original source, collection method and label-generation procedure have not been independently established. The CSV is available in the separately provided project bundle, but is not currently committed to GitHub.

---

## 4. Data Preprocessing and Feature Engineering

The project applies several preprocessing methods to convert structured health-related data into numerical model inputs.

### Data Cleaning

Whitespace is removed from string columns.

Identifiers and unsuitable target-related fields are excluded from the feature set.

### Blood Pressure Transformation

Blood pressure readings such as `120/80` are split into:

- BP_Systolic
- BP_Diastolic

This creates two numerical features for the classification models.

### Symptom Multi-Hot Encoding

Reported symptoms are represented using binary indicators.

For example:

| Symptom Feature | Value |
|---|---:|
| Sym_Fever | 1 |
| Sym_Cough | 1 |
| Sym_Headache | 0 |
| Sym_Fatigue | 0 |

The corrected feature builder learns the symptom vocabulary from training data.

### Numerical Transformation

The Yeo-Johnson PowerTransformer is applied to numerical features including:

- Age
- Heart rate
- Body temperature
- Oxygen saturation
- Systolic blood pressure
- Diastolic blood pressure

StandardScaler is subsequently used to standardize features.

### Class Imbalance Handling

SMOTE (Synthetic Minority Over-sampling Technique) is used to balance classes during training.

In the corrected approach, SMOTE operates only within training folds and is not applied to validation or held-out test samples.

---

## 5. Machine Learning Algorithms and Training

Four supervised learning algorithms were trained and compared.

| Algorithm | Purpose |
|---|---|
| Random Forest | Ensemble-based nonlinear classification |
| Logistic Regression | Linear classification baseline |
| Support Vector Machine | Margin-based classification |
| Gaussian Naive Bayes | Probabilistic classification baseline |

### Original Training Configuration

- Train/test split: 80/20
- Stratification: Enabled
- Random state: 42
- Cross-validation: 5-fold StratifiedKFold
- Optimization: GridSearchCV
- Primary optimization metric: Macro F1

### Hyperparameter Optimization

| Algorithm | Parameters explored |
|---|---|
| Random Forest | n_estimators, max_depth |
| Logistic Regression | C |
| SVM | C, kernel |
| Gaussian Naive Bayes | var_smoothing |

### Original Recorded Results

| Model | Test Accuracy | CV Macro F1 |
|---|---:|---:|
| Random Forest | 100.00% | 1.0000 |
| Logistic Regression | 97.00% | 0.9876 |
| SVM | 94.75% | 0.9884 |
| Naive Bayes | 98.25% | 0.9906 |

These are the original notebook's recorded results.

The original workflow was later found to contain preprocessing and cross-validation leakage risks. Therefore, these scores are preserved as historical experimental results rather than treated as reliable estimates of generalization.

---

## 6. Leakage-Aware Pipeline Improvements

A technical audit identified three important risks in the original workflow.

### Identified Issues

**1. Preprocessing leakage**

PowerTransformer was fitted before the train/test split.

**2. Cross-validation leakage**

SMOTE and StandardScaler were applied before GridSearchCV rather than within individual training folds.

**3. Potential target leakage**

Severity showed a strong relationship with Diagnosis and could reveal target-related information.

### Corrected Pipeline Architecture

```text
Raw Dataset
     |
     v
Feature / Target Separation
     |
     v
Stratified Train-Test Split
     |
     +------------------------+
     |                        |
     v                        v
Training Data              Test Data
     |                        |
     v                        |
GridSearchCV                  |
     |                        |
     v                        |
Training Fold                 |
     |                        |
     v                        |
Feature Engineering           |
     |                        |
     v                        |
PowerTransformer              |
     |                        |
     v                        |
StandardScaler                |
     |                        |
     v                        |
SMOTE                         |
     |                        |
     v                        |
Classifier Training           |
     |                        |
     v                        |
Best Model Selection          |
     |                        |
     +-----------+------------+
                 |
                 v
         Held-Out Evaluation
                 |
                 v
      Accuracy / F1 / Reports
```

### Improvements Implemented

- Fitted preprocessing restricted to training partitions
- Feature engineering incorporated into the pipeline
- SMOTE performed within cross-validation folds
- Final test partition kept outside model selection
- Severity included/excluded in separate experiments
- Complete fitted inference pipeline saved as one artifact

The corrected comparison uses three-fold StratifiedKFold cross-validation.

---

## 7. Model Evaluation and Results

### Corrected Experiment A — Without Severity

| Model | Test Accuracy | Test Macro F1 | CV Macro F1 |
|---|---:|---:|---:|
| **Random Forest** | **99.75%** | **0.9865** | **0.9856** |
| Logistic Regression | 93.00% | 0.8548 | 0.8491 |
| SVM | 94.00% | 0.8692 | 0.8839 |
| Gaussian Naive Bayes | 79.25% | 0.7811 | 0.7597 |

Random Forest achieved the highest recorded held-out accuracy.

### Corrected Experiment B — With Severity

| Model | Test Accuracy |
|---|---:|
| Random Forest | 100.00% |
| Logistic Regression | 96.75% |
| SVM | 95.75% |
| Gaussian Naive Bayes | 98.50% |

The difference between experiments highlights the importance of investigating whether a feature is legitimately available at prediction time.

### Selected Final Model

**Random Forest — without Severity**

| Metric | Recorded Value |
|---|---:|
| Test Accuracy | 99.75% |
| Test Macro F1 | 0.9865 |
| Correct Predictions | 399 / 400 |
| Test Samples | 400 |

One Pneumonia sample was classified as Flu in the corrected model comparison.

### Model Sanity Checks

Additional analysis found:

- No exact duplicate feature records in the examined data
- Strong predictive relationships involving temperature, oxygen saturation, cough and fever
- A shallow decision tree also achieved unusually high classification performance
- Possible rule-based or synthetic label-generation patterns

These findings suggest that performance may reflect structured dataset rules rather than broad medical generalization.

### Feature Importance

A separate Random Forest sanity-check experiment produced these approximate impurity-based importances:

| Feature | Importance |
|---|---:|
| Body Temperature | 23.98% |
| Oxygen Saturation | 18.09% |
| Cough | 16.66% |
| Fever | 13.66% |
| Heart Rate | 5.88% |

These values come from an independent sanity-check model, not the final tuned deployment model.

**Important:** Feature importance indicates statistical model behavior within the dataset, not medical causation.

---

## 8. Flask Web Application and Testing

The trained classification pipeline is integrated into a Flask-based web application.

### Application Features

- Web-based patient information form
- Symptom selection
- Numerical input validation
- Blood pressure consistency validation
- Trained pipeline integration
- Prediction result display
- Error handling
- Educational-use disclaimer

### Flask Application Architecture

```text
User
  |
  v
Flask Web Interface
  |
  v
Input Validation
  |
  v
Pandas DataFrame
  |
  v
Saved ML Pipeline
  |
  +--> Feature Engineering
  |
  +--> Numerical Preprocessing
  |
  +--> Random Forest Classifier
  |
  v
Predicted Dataset Category
  |
  v
Result Display
```

SMOTE is used during training only, not while making individual predictions.

### Flask Routes

| Route | Method | Function |
|---|---|---|
| `/` | GET | Display prediction form |
| `/predict` | POST | Validate inputs and display prediction |

### Application Testing

Basic testing was performed using the locally running Flask interface.

| Test | Expected Behavior | Result |
|---|---|---|
| Invalid blood pressure | Reject inconsistent values | Passed |
| Valid prediction form | Display predicted category | Passed |
| Invalid age/oxygen input | Reject out-of-range values | Pending |
| Missing required fields | Reject incomplete inputs | Pending |

One successful form submission displayed the prediction category **Healthy** with the educational disclaimer.

The browser tests demonstrate that selected application workflows operate successfully. They do not establish clinical correctness.

---

## 9. Project Structure and Installation

### Repository Structure

```text
Disease-Prediction-Using-Machine-Learning/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── train_deployment_model.py
├── disease_leakage_free_pipeline.py
├── templates/
│   └── index.html
├── notebooks/
│   └── disease_type_prediction.ipynb
├── models/
│   └── legacy/
│       ├── disease_prediction_rf_model.pkl
│       ├── scaler.pkl
│       ├── label_encoder.pkl
│       ├── power_transformer.pkl
│       └── model_columns.pkl
└── results/
    └── holdout_evaluation.json
```

### Original vs Updated Components

The original notebook is preserved in `notebooks/` and the old `.pkl` model artifacts in `models/legacy/`. These are legacy experiment files, not inputs for the corrected Flask app.

The updated Flask application uses a separately trained, complete fitted pipeline:

`educational_disease_pipeline.joblib`

This file is not currently committed to GitHub and must be created locally or obtained from the trusted local project bundle.

### Installation

**Requirements:** Python 3.11 or 3.12 recommended, pip, Git, and the original dataset.

**Step 1 — Clone Repository**

```bash
git clone https://github.com/syblejan/Disease-Prediction-Using-Machine-Learning.git
cd Disease-Prediction-Using-Machine-Learning
```

**Step 2 — Create Virtual Environment**

```bash
python -m venv .venv
```

Activate on Windows:

```bat
.venv\Scripts\activate
```

**Step 3 — Install Dependencies**

```bash
python -m pip install -r requirements.txt
```

**Step 4 — Add Dataset**

Place `disease_diagnosis.csv` in the project root directory.

**Step 5 — Train Deployment Model**

```bash
python train_deployment_model.py
```

This generates the fitted pipeline in the project root and writes the evaluation summary to `results/holdout_evaluation.json`.

**Step 6 — Run Flask**

```bash
python app.py
```

Open:

`http://127.0.0.1:5000`

### Run Full Four-Model Comparison

```bash
python disease_leakage_free_pipeline.py --csv disease_diagnosis.csv --out disease_evaluation
```

This generates model comparison results and detailed evaluation reports.

**Reproducibility note:** Dependency compatibility should be verified when using a new Python environment. Never load untrusted Joblib or pickle files.

---

## 10. Limitations and Future Improvements

### Current Limitations

**Dataset origin:** Source, licensing and label-generation details are not independently verified.

**Generalization:** The exceptionally high accuracy may reflect simple or synthetic dataset patterns.

**Clinical validation:** The models have not been evaluated for real clinical use.

**Class imbalance:** Pneumonia has relatively few examples, limiting confidence in performance estimates for this class.

**Deployment:** The Flask application is a local educational prototype, not a production-ready healthcare system.

**Testing:** Automated integration tests, independent data validation and robust deployment testing remain incomplete.

### Future Improvements

- Verify original dataset provenance and licensing
- Evaluate using independently collected data
- Investigate synthetic label-generation patterns
- Expand minority-class evaluation
- Add calibration and uncertainty analysis
- Add full automated tests
- Improve model explainability
- Package the application with Docker
- Introduce CI/CD checks
- Improve deployment security
- Add model versioning and experiment tracking
- Include Flask screenshots and demo documentation

---

## 11. Learning Outcomes and Author

### Key Learning Outcomes

This project demonstrates practical experience in:

- End-to-end machine learning development
- Data preprocessing and feature engineering
- Multiclass classification
- Class imbalance handling
- Hyperparameter optimization
- Model evaluation and comparison
- Data leakage detection and correction
- Model interpretation
- Flask application development
- Responsible reporting of ML results

**Main takeaway:** High test accuracy alone does not establish real-world reliability. Proper data splitting, leakage prevention, reproducibility and independent validation are essential.

### Author

**Syble Jan**

Final-Year BS Computer Science Student  
Aspiring AI/ML Engineer | Machine Learning | Generative AI

[GitHub Profile](https://github.com/syblejan)

---

**Project Status:** Corrected educational ML pipeline evaluated and integrated with a locally tested Flask application. External validation and production readiness remain future work.
