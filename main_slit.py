import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle 
import os
import requests
import subprocess
import time
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

BASE_DIR = os.getcwd()

if "backend_started" not in st.session_state: #auto start
    try:
        subprocess.Popen(["uvicorn", "main_fapi:app", "--port", "8000", "--host", "127.0.0.1"])
        st.session_state["backend_started"] = True
        time.sleep(8)
    except Exception as e:
        st.error(f"Gagal memulai backend: {e}")

API_URL_BASE = 'http://127.0.0.1:8000/predict'

@st.cache_data
def get_api_info():
    try:
        res = requests.get(f"{API_URL_BASE}/info", timeout=5)
        return res.json()
    except:
        return None

info_res = get_api_info()

st.set_page_config(page_title="Placement Analytics Pro", layout="wide")

#Sidebar
with st.sidebar:
    st.header("System Control")
    st.info("Aplikasi ini memprediksi peluang kerja mahasiswa menggunakan model prediksi XGBoost berdasarkan metrik akademik dan skill.")

    st.divider()
    st.subheader("Backend Status")
    try:
        res = requests.get('http://127.0.0.1:8000/', timeout=2)
        if res.status_code == 200:
            st.success("● FastAPI Online")
        else:
            st.warning("○ API Ready (No Response)")
    except:
        st.error("○ API Offline")
    
    st.divider()
    st.caption("v1.0 | Sklearn 1.8.0")

#Main UI
st.title("Student Placement Prediction")
st.write("Masukkan data mahasiswa di bawah ini untuk memprediksi status penempatan.")
st.divider()

col1, col2, col3 = st.columns(3)

with st.form("prediction_form"):
    tab1, tab2, tab3 = st.tabs(["Akademik", "Skills & Projects", "Pengalaman"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ['Male', 'Female'])
            cgpa = st.slider("Current CGPA", 0.0, 10.0, 8.0, step=0.1)
        with c2:
            ssc_percentage = st.number_input("SSC % (10th Grade)", 0, 100, 75)
            hsc_percentage = st.number_input("HSC % (12th Grade)", 0, 100, 75)
            degree_percentage = st.number_input("Degree %", 0, 100, 75)

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            technical_skill_score = st.slider("Technical Skill", 0, 100, 70)
            soft_skill_score = st.slider("Soft Skill", 0, 100, 70)
        with c2:
            entrance_exam_score = st.number_input("Entrance Exam Score", 0, 100, 75)
            attendance_percentage = st.number_input("Attendance %", 0, 100, 85)
        with c3:
            live_projects = st.number_input("Live Projects", 0, 10, 1)
            extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"], horizontal=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            work_experience_months = st.number_input("Work Experience (Months)", 0, 60, 0)
        with c2:
            internship_count = st.number_input("Internship Count", 0, 10, 1)
            certifications = st.number_input("Certifications Count", 0, 10, 0)

    submit_button = st.form_submit_button("Analisis Peluang Kerja", type="primary")

st.divider()

if submit_button:
    payload = {
        "gender": gender, "ssc_percentage": ssc_percentage, "hsc_percentage": hsc_percentage,
        "degree_percentage": degree_percentage, "cgpa": cgpa, "entrance_exam_score": entrance_exam_score,
        "technical_skill_score": technical_skill_score, "soft_skill_score": soft_skill_score,
        "internship_count": internship_count, "live_projects": live_projects,
        "work_experience_months": work_experience_months, "certifications": certifications,
        "attendance_percentage": attendance_percentage, "extracurricular_activities": extracurricular
    }

    # Visualisasi Radar Chart untuk Profil Mahasiswa
    st.subheader("Profil Kompetensi Mahasiswa")
    categories = ['Technical', 'Soft Skills', 'Academic', 'Attendance', 'Entrance']
    values = [technical_skill_score, soft_skill_score, degree_percentage, attendance_percentage, entrance_exam_score]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Student Profile'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.spinner("Loading Prediction"):
        try:
            response = requests.post(API_URL_BASE, json=payload)
            if response.status_code == 200:
                result = response.json()
                
                st.divider()
                res_col1, res_col2 = st.columns(2)
                
                if result['status'] == 1:
                    with res_col1:
                        st.metric("Status Penempatan", "PLACED", delta="Tersedia")
                    with res_col2:
                        st.metric("Estimasi Gaji (LPA)", f"{result['salary']:,.2f}", delta="Annual Package")
                    st.success(f"Mahasiswa diprediksi mendapatkan penempatan dengan paket gaji sebesar {result['salary']:,.2f} LPA.")
                else:
                    with res_col1:
                        st.metric("Status Penempatan", "NOT PLACED", delta="-", delta_color="inverse")
                    st.warning("Berdasarkan data, mahasiswa memerlukan peningkatan pada skill teknis atau akademik.")
            else:
                st.error(f"Gagal processing data. Error Code: {response.status_code}")
        except Exception as e:
            st.error("Backend gagal tersambung. Apakah FastAPI berjalan di port:8000?")