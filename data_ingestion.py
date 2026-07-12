import pandas as pd
import numpy as np
import os

def ingest_data(filename):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', filename)
    OUT_DIR = os.path.join(BASE_DIR, "data", 'ingested')

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)

    target = ["placement_status", "salary_package_lpa"]
    #

    #Paste your data cleaning code here (FEATURE ENGINEERING)
    df['skill_combined'] = df["technical_skill_score"] + df["soft_skill_score"]
    df['skill_ratio'] = df["technical_skill_score"] / (df["soft_skill_score"] + 1)
    df['cgpa_skill'] = df['cgpa'] * df['technical_skill_score']

    df['academic_avg'] = (
        df['ssc_percentage'] + df['hsc_percentage'] + df['degree_percentage']
    ) / 3

    df['academic_consistency'] = df[['ssc_percentage','hsc_percentage','degree_percentage']].std(axis=1)

    # target = ["placement_status", "salary_package_lpa"]

    exclude_cols = ["student_id",  "hsc_percentage", "ssc_percentage", "degree_percentage",
                    "internship_count", "live_projects", "work_experience_months", "backlogs"]

    features = [col for col in df.columns if col not in exclude_cols]

    df_final_class = df[features]
    # df_final_reg = df_final_class[df_final_class['placement_status'] > 0]


    SAVE_PATH = os.path.join(OUT_DIR, 'data_cleaned.csv')
    df_final_class.to_csv(SAVE_PATH, index=False)
    print(f"Data berhasil diingest dan disimpan di {SAVE_PATH}")

if __name__ == "__main__":
    #change this
    ingest_data("B.csv")