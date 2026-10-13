"""Leakage-aware educational disease classification. Not for clinical use."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

NUM = ['Age','Heart_Rate_bpm','Body_Temperature_C','Oxygen_Saturation_%','BP_Systolic','BP_Diastolic']
SYM = ['Symptom_1','Symptom_2','Symptom_3']

class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, use_severity=False):
        self.use_severity=use_severity
    def fit(self, X, y=None):
        self.symptoms_=sorted(set(X[SYM].fillna('').astype(str).values.ravel())-{''})
        return self
    def transform(self, X):
        X=X.copy()
        bp=X['Blood_Pressure_mmHg'].astype(str).str.split('/',n=1,expand=True)
        out=X[['Age','Heart_Rate_bpm','Body_Temperature_C','Oxygen_Saturation_%']].astype(float).copy()
        out['BP_Systolic']=pd.to_numeric(bp[0],errors='raise')
        out['BP_Diastolic']=pd.to_numeric(bp[1],errors='raise')
        out['Gender']=X['Gender'].map({'Male':1,'Female':0}).astype(float)
        if self.use_severity:
            out['Severity']=X['Severity'].map({'Mild':1,'Moderate':2,'Severe':3}).astype(float)
        vals=X[SYM].fillna('').astype(str)
        for s in self.symptoms_:
            out['Sym_'+s]=vals.eq(s).any(axis=1).astype(float)
        if not np.isfinite(out.to_numpy()).all():
            raise ValueError('Missing/unrecognized categories or nonfinite numeric inputs')
        return out

class NumericPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_=list(X.columns)
        self.pt_=PowerTransformer(method='yeo-johnson').fit(X[NUM])
        z=X.copy().astype(float)
        z.loc[:,NUM]=self.pt_.transform(X[NUM])
        self.scaler_=StandardScaler().fit(z)
        return self
    def transform(self, X):
        z=X[self.columns_].copy().astype(float)
        z.loc[:,NUM]=self.pt_.transform(z[NUM])
        return self.scaler_.transform(z)

def run(csv_path, out_dir):
    df=pd.read_csv(csv_path)
    df=df.apply(lambda c:c.str.strip() if c.dtype=='object' else c)
    y=df['Diagnosis']
    X=df.drop(columns=['Diagnosis','Patient_ID','Treatment_Plan'])
    xtr,xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    algorithms={
        'Random Forest':(RandomForestClassifier(random_state=42,n_jobs=1),{'model__n_estimators':[100,200],'model__max_depth':[10,None]}),
        'Logistic Regression':(LogisticRegression(max_iter=1500,random_state=42),{'model__C':[0.1,1,10]}),
        'SVM':(SVC(random_state=42),{'model__C':[0.1,1,10],'model__kernel':['linear','rbf']}),
        'Naive Bayes':(GaussianNB(),{'model__var_smoothing':[1e-9,1e-8]})}
    results=[];reports={}
    for use_severity in [False,True]:
        exp='with_severity' if use_severity else 'without_severity'
        for name,(est,params) in algorithms.items():
            pipe=Pipeline([('features',FeatureBuilder(use_severity=use_severity)),('preprocessing',NumericPreprocessor()),('smote',SMOTE(random_state=42,k_neighbors=5)),('model',est)])
            gs=GridSearchCV(pipe,params,scoring='f1_macro',cv=cv,n_jobs=1,refit=True)
            gs.fit(xtr,ytr)
            yp=gs.predict(xte)
            row={'experiment':exp,'model':name,'cv_macro_f1':round(gs.best_score_,4),'test_accuracy':round(accuracy_score(yte,yp),4),'test_macro_f1':round(f1_score(yte,yp,average='macro'),4),'best_params':str(gs.best_params_)}
            print(row,flush=True)
            results.append(row)
            reports[exp+' / '+name]={'classification_report':classification_report(yte,yp,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(yte,yp,labels=sorted(y.unique())).tolist(),'labels':sorted(y.unique())}
    out_dir=Path(out_dir);out_dir.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(results).to_csv(out_dir/'evaluation_results.csv',index=False)
    (out_dir/'detailed_reports.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',default='disease_diagnosis.csv')
    parser.add_argument('--out',default='disease_evaluation')
    args=parser.parse_args()
    run(args.csv,args.out)
