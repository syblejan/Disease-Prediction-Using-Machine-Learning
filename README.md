# Disease Prediction Using Machine Learning

**A comparative machine learning classification project for predicting disease categories from patient symptoms and health indicators, with a Flask-based inference prototype.**

**Python | Scikit-learn | Random Forest | SMOTE | Flask | Pandas**

> **Medical Disclaimer:** This project is for educational and experimental use only. It is not clinically validated and must not be used for medical diagnosis or treatment decisions.

## 1. Project Overview

This project investigates machine learning approaches for classifying patients into five diagnosis categories using symptoms, vital signs and health indicators.

Four classification algorithms are trained and compared using stratified data splitting, class balancing, hyperparameter search and multiple evaluation metrics.

The selected Random Forest model is saved alongside its preprocessing components for integration into a Flask application.

## 2. Dataset

**Dataset file:** `disease_diagnosis.csv`

**Dataset size:** Approximately 2,000 records, inferred from the notebook's 400-sample test set and 20% test split.

**Target variable:** `Diagnosis`

**Diagnosis categories:**
- Bronchitis
- Cold
- Flu
- Healthy
- Pneumonia

**Input features:**
- Age and gender
- Three reported symptoms
- Heart rate
- Body temperature
- Blood pressure
- Oxygen saturation
- Severity

The notebook loads the dataset from a local CSV file uploaded to Google Colab. Its original source, collection methodology and licensing are not documented.

### Test dataset distribution

| Diagnosis | Test samples |
|---|---:|
| Bronchitis | 67 |
| Cold | 33 |
| Flu | 58 |
| Healthy | 233 |
| Pneumonia | 9 |
| **Total** | **400** |

## 3. Data Preprocessing

The following preprocessing steps are implemented:

1. **Data cleaning:** Remove surrounding whitespace from text values.
2. **Blood pressure extraction:** Split blood pressure into systolic and diastolic numerical features.
3. **Column removal:** Remove patient identifiers, original blood pressure strings and treatment plans.
4. **Numerical transformation:** Apply Yeo-Johnson PowerTransformer.
5. **Symptom encoding:** Convert symptom categories into multi-hot encoded features.
6. **Categorical encoding:** Encode gender and severity numerically.
7. **Target encoding:** Convert diagnosis labels using LabelEncoder.
8. **Train/test split:** Apply an 80/20 stratified split using random state 42.
9. **Class balancing:** Apply SMOTE to the training partition.
10. **Feature scaling:** Apply StandardScaler.

**Evaluation caveat:** The notebook fits PowerTransformer before the train/test split. A corrected workflow should fit all learned preprocessing transformations exclusively on training data.

## 4. Machine Learning Algorithms

Four supervised classifiers are evaluated:

- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

Hyperparameter optimization is performed using GridSearchCV with five-fold StratifiedKFold and macro F1 scoring.

The search explores Random Forest tree counts and depths, Logistic Regression regularization, SVM kernels and regularization, and Naive Bayes smoothing.

## 5. Model Evaluation

### Recorded test results

| Model | Accuracy | Cross-validation Macro F1 |
|---|---:|---:|
| Random Forest | 100.00% | 1.0000 |
| Gaussian Naive Bayes | 98.25% | 0.9906 |
| Logistic Regression | 97.00% | 0.9876 |
| SVM | 94.75% | 0.9884 |

These values are extracted from saved notebook outputs, not from an independently reproduced experiment.

Random Forest achieved the highest recorded test accuracy. Its classification report also shows precision, recall and F1-score of 1.00 for all five classes on the held-out test partition.

### Important limitations

These results require additional validation because:

- Numerical preprocessing was fitted before the train/test split.
- SMOTE and feature scaling were performed before cross-validation, potentially biasing CV scores.
- Severity may encode information closely associated with the target diagnosis.
- Duplicate-record checks and independent external validation are not documented.
- The unusually high classification scores may not generalize to independent real-world data.

The reported metrics should be considered preliminary experimental findings rather than evidence of clinical diagnostic performance.

## 6. Model Explainability

The notebook uses Random Forest feature importance to identify and visualize the 15 highest-ranked predictors.

Confusion matrices are generated for all four evaluated models.

## 7. Flask Application

The repository contains `app.py`, a Flask application that:

- Accepts patient information from a web form
- Encodes reported symptoms into expected model features
- Loads saved preprocessing components
- Generates a prediction with the trained Random Forest model
- Converts the predicted numeric class into a diagnosis-category name

### Saved model artifacts

- `disease_prediction_rf_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
- `power_transformer.pkl`
- `model_columns.pkl`

These artifacts are included in the repository.

## 8. Repository Structure

```text
Disease-Prediction-Using-Machine-Learning/
├── app.py
├── disease_type_prediction.ipynb
├── disease_prediction_rf_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── power_transformer.pkl
├── model_columns.pkl
└── README.md
```

The Flask application refers to `templates/index.html`, but this template has not been verified in the repository. It needs to be supplied before the web interface can run successfully.

## 9. Installation

Clone the repository:

```bash
git clone https://github.com/syblejan/Disease-Prediction-Using-Machine-Learning.git
cd Disease-Prediction-Using-Machine-Learning
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install flask numpy pandas scikit-learn imbalanced-learn scipy matplotlib seaborn joblib
```

Run the Flask application:

```bash
python app.py
```

The current application requires a compatible environment and the missing HTML template. A fresh-clone execution has not yet been verified.

## 10. Future Improvements

- Refactor preprocessing into a leakage-safe scikit-learn/imblearn Pipeline
- Perform SMOTE independently within training folds
- Investigate duplicate records and potential target leakage
- Re-evaluate all models on an untouched test set
- Add model evaluation plots and screenshots
- Complete the Flask HTML interface
- Add requirements.txt with verified dependency versions
- Improve input validation and security
- Add automated tests and deployment documentation

## 11. Author

**Syble Jan**

Final-Year BS Computer Science Student

Aspiring AI/ML Engineer

[GitHub Profile](https://github.com/syblejan)

---

*An educational machine learning project demonstrating multiclass classification, comparative evaluation, class imbalance handling and model integration.*
