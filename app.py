"""Educational Flask inference demo; NOT medical advice."""
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

APP_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
model = joblib.load(APP_DIR / 'educational_disease_pipeline.joblib')
ALLOWED_SYMPTOMS = {'Body ache','Cough','Fatigue','Fever','Headache','Runny nose','Shortness of breath','Sore throat'}

def number(form,key,minimum,maximum):
    value=float(form[key])
    if not minimum <= value <= maximum:
        raise ValueError(f'{key} must be between {minimum} and {maximum}')
    return value

@app.get('/')
def home():
    return render_template('index.html',symptoms=sorted(ALLOWED_SYMPTOMS))

@app.post('/predict')
def predict():
    try:
        age=number(request.form,'Age',0,120)
        hr=number(request.form,'Heart_Rate_bpm',30,240)
        temp=number(request.form,'Body_Temperature_C',30,44)
        spo2=number(request.form,'Oxygen_Saturation_%',50,100)
        systolic=number(request.form,'BP_Systolic',50,260)
        diastolic=number(request.form,'BP_Diastolic',30,180)
        if systolic <= diastolic:
            raise ValueError('Systolic blood pressure must exceed diastolic pressure')
        gender=request.form['Gender']
        if gender not in {'Male','Female'}:
            raise ValueError('Invalid gender option')
        symptoms=request.form.getlist('Symptoms')
        if len(symptoms)>3 or len(symptoms)!=len(set(symptoms)) or any(s not in ALLOWED_SYMPTOMS for s in symptoms):
            raise ValueError('Select up to three different valid symptoms')
        symptoms += ['']*(3-len(symptoms))
        row={'Age':age,'Gender':gender,'Heart_Rate_bpm':hr,'Body_Temperature_C':temp,
             'Oxygen_Saturation_%':spo2,'Blood_Pressure_mmHg':f'{systolic}/{diastolic}',
             'Symptom_1':symptoms[0],'Symptom_2':symptoms[1],'Symptom_3':symptoms[2]}
        prediction=model.predict(pd.DataFrame([row]))[0]
        return render_template('index.html',symptoms=sorted(ALLOWED_SYMPTOMS),prediction=prediction)
    except (ValueError,KeyError,TypeError) as exc:
        return render_template('index.html',symptoms=sorted(ALLOWED_SYMPTOMS),error=str(exc)),400
    except Exception:
        app.logger.exception('Prediction failed')
        return render_template('index.html',symptoms=sorted(ALLOWED_SYMPTOMS),error='Unable to process request.'),500

if __name__=='__main__':
    app.run(host='127.0.0.1',port=5000,debug=False)
