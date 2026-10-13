"""Train an educational Random Forest on an 80% stratified training split."""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from disease_leakage_free_pipeline import FeatureBuilder, NumericPreprocessor

here = Path(__file__).resolve().parent
df = pd.read_csv(here / 'disease_diagnosis.csv')
df = df.apply(lambda c: c.str.strip() if c.dtype == 'object' else c)
y = df['Diagnosis']
X = df.drop(columns=['Diagnosis','Patient_ID','Treatment_Plan'])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
model=Pipeline([('features',FeatureBuilder(use_severity=False)),
 ('preprocessing',NumericPreprocessor()),
 ('smote',SMOTE(random_state=42,k_neighbors=5)),
 ('model',RandomForestClassifier(n_estimators=200,max_depth=10,random_state=42,n_jobs=1))])
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
metrics={'test_accuracy':accuracy_score(y_test,y_pred),'test_macro_f1':f1_score(y_test,y_pred,average='macro'),'classification_report':classification_report(y_test,y_pred,output_dict=True,zero_division=0)}
(here/'holdout_evaluation.json').write_text(json.dumps(metrics,indent=2))
joblib.dump(model,here/'educational_disease_pipeline.joblib')
print('Accuracy:',metrics['test_accuracy'],'Macro F1:',metrics['test_macro_f1'])
print('Saved educational_disease_pipeline.joblib')
