from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import numpy as np

app = FastAPI()

BASE_DIR = os.getcwd()
model_1 = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'model_classification_placement.pkl'))
model_2 = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'model_regression_salary.pkl'))

@app.get("/")
def home():
    return{"message": "FastAPI is active!","version": "1.0" }

@app.get("/info")
def get_info():
    models_info = [
        {
            "name": "Classification_Placement",
            "target": "placement_status",
            "type": "classification"
        },
        {
            "name": "Regression_Salary",
            "target": "salary_log",
            "type": "regression"
        }
    ]
    return {
        "total_models": len(models_info),
        "details": models_info
    }
   

class StudentData(BaseModel):
    gender: str
    ssc_percentage: float
    hsc_percentage: float
    degree_percentage: float
    cgpa: float
    entrance_exam_score: float
    technical_skill_score: float
    soft_skill_score: float
    internship_count: int
    live_projects: int
    work_experience_months: int
    certifications: int
    attendance_percentage: int
    extracurricular_activities: str

def preprocess_data(data:dict):
    df = pd.DataFrame([data])

    df['skill_combined'] = df["technical_skill_score"] + df["soft_skill_score"]
    df['skill_ratio'] = df["technical_skill_score"] / (df["soft_skill_score"] + 1)
    df['cgpa_skill'] = df['cgpa'] * df['technical_skill_score']
    df['academic_avg'] = (
        df['ssc_percentage'] + df['hsc_percentage'] + df['degree_percentage']
    ) / 3
    df['academic_consistency'] = df[['ssc_percentage','hsc_percentage','degree_percentage']].std(axis=1)
    df['experience_score'] = (
        df['internship_count'] * 2 +
        df['live_projects'] +
        df['work_experience_months'] / 6
    )
    return df

@app.post("/predict")
def predict(data: StudentData):
    raw_data = data.dict()
    df = preprocess_data(raw_data)

    exclude_cols2 = ["student_id",  "hsc_percentage", "ssc_percentage", "degree_percentage",
                 "internship_count", "live_projects", "work_experience_months", "backlogs", "salary_package_lpa"]

    df_input_cls = df.drop(columns = exclude_cols2, errors = 'ignore')
    status_pred = model_1.predict(df_input_cls)[0]

    salary_pred = 0.0
    if status_pred == 1:
        salary_log = model_2.predict(df)
        salary_pred = float(np.expm1(salary_log)[0])

    return {
        "status": int(status_pred),
        "salary": salary_pred
    }
