from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load all required components
model = joblib.load('disease_prediction_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
pt = joblib.load('power_transformer.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture basic inputs
        age = float(request.form['Age'])
        gender = 1 if request.form['Gender'] == 'Male' else 0
        heart_rate = float(request.form['Heart_Rate_bpm'])
        body_temp = float(request.form['Body_Temperature_C'])
        oxy_sat = float(request.form['Oxygen_Saturation_%'])
        
        # 2. Capture and map Severity
        severity_map = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
        severity = severity_map[request.form['Severity']]
        
        # 3. Split Blood Pressure (e.g., "120/80" -> 120.0, 80.0)
        bp_string = request.form['Blood_Pressure_mmHg']
        bp_sys, bp_dia = map(float, bp_string.split('/'))
        
        # 4. Capture Symptoms (from checkboxes)
        user_symptoms = request.form.getlist('Symptoms')
        
        # 5. Build a dictionary for a single patient record
        patient_data = {
            'Age': age,
            'Gender': gender,
            'Heart_Rate_bpm': heart_rate,
            'Body_Temperature_C': body_temp,
            'Oxygen_Saturation_%': oxy_sat,
            'Severity': severity,
            'BP_Systolic': bp_sys,
            'BP_Diastolic': bp_dia
        }
        
        # Add symptom columns dynamically (1 if user checked it, 0 otherwise)
        # We loop through the saved columns to ensure the exact same names are generated
        for col in model_columns:
            if col.startswith('Sym_'):
                symptom_name = col.replace('Sym_', '')
                patient_data[col] = 1 if symptom_name in user_symptoms else 0
                
        # 6. Convert to DataFrame (ensure exact column order as training)
        input_df = pd.DataFrame([patient_data], columns=model_columns)
        
        # 7. Apply PowerTransformer to numerical columns (same as notebook)
        num_cols = ['Age', 'Heart_Rate_bpm', 'Body_Temperature_C', 'Oxygen_Saturation_%', 'BP_Systolic', 'BP_Diastolic']
        input_df[num_cols] = pt.transform(input_df[num_cols])
        
        # 8. Apply StandardScaler
        scaled_data = scaler.transform(input_df)
        
        # 9. Predict
        numeric_pred = model.predict(scaled_data)
        disease_name = le.inverse_transform(numeric_pred)[0]
        
        return render_template('index.html', prediction_text=f'Predicted Diagnosis: {disease_name}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error processing input: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)