import streamlit as st
import joblib
from pathlib import Path
import time

st.set_page_config(
    page_title="MediTrack | AI Health Screening",
    layout="wide",
)

st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #f4f6fa 0%, #fcfcfd 80%);
}
.header-bar {
    background: #23296d;
    color: #fff;
    border-radius: 15px 15px 0 0;
    padding: 32px 0 18px 0;
    margin-bottom: 0;
    box-shadow: 0 8px 32px #23296d14;
    text-align: center;
}
.header-bar img {
    border-radius: 50%;
    border: 3px solid #fff;
    margin-bottom: 10px;
}
.main-card {
    background: #fff;
    border-radius: 0 0 18px 18px;
    box-shadow: 0 10px 30px #23296d11;
    padding: 36px 32px 34px 32px;
    margin-top: 0;
    margin-bottom: 20px;
    max-width: 700px;
    margin-left:auto; margin-right:auto;
    border: 2px solid #E9ECF3;
}
.stButton>button {
    background: linear-gradient(90deg,#FFD700 0%,#F7C700 100%);
    color: #202E61 !important;
    border-radius: 50px !important;
    font-size: 1.17em !important;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 15px 38px !important;
    margin-top:10px;
    box-shadow: 0 3px 16px #f7c70035;
    border: none;
}
.result-wide {
    width:98vw;
    max-width:1180px;
    margin:auto;
    margin-top:18px;
}
.result-card-low {background:#e7fff3;color:#13764a;border-radius:22px;padding:38px 36px;border:2.5px solid #b2e5c3;box-shadow:0 4px 18px #17bd6742;}
.result-card-med {background:#fffde7;color:#b88a00;border-radius:22px;padding:38px 36px;border:2.5px solid #ffe399;box-shadow:0 4px 18px #e6c4001a;}
.result-card-high {background:#ffeaea;color:#e32636;border-radius:22px;padding:38px 36px; border:2.5px solid #ffabab; box-shadow:0 4px 18px #dc545433;}
.vital-pill {background:#23296d;color:#FFD700;border-radius:16px;padding:25px 0 13px 0;font-weight:700;font-size:1.25em;text-align:center;box-shadow:0 1px 16px #23296d25;margin-bottom:11px;}
@media (max-width:900px) {
    .main-card, .result-wide {padding:18px 4vw;}
    .result-card-low, .result-card-med, .result-card-high {padding:18px 4vw;}
}
</style>
""", unsafe_allow_html=True)

# --- HEADER
st.markdown("""
<div class="header-bar">
    <img src='/static/logo.png' width='75'>
    <h1 style='margin-bottom:7px;font-size:2em;'>MediTrack Health</h1>
    <div style='font-size:1.08em;margin-top:-7px;'>
        <span style="background:#FFD700;color:#23296d;border-radius:10px;padding:7px 24px;margin:0 6px;font-weight:700;letter-spacing:1.2px;">AI Patient Risk</span>
    </div>
    <div style='font-size:1em;margin-top:9px;color:#F7C700;'>Clinical risk assessment. For guidance only.</div>
</div>
""", unsafe_allow_html=True)

# --- INPUT FORM (main white card)
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;letter-spacing:0.2em;color:23296d;'>Patient Information</h3>", unsafe_allow_html=True)
with st.form("patient_form"):
    c1, c2 = st.columns(2)
    name = c1.text_input("Full Name")
    age = c2.number_input("Age", 1, 120, 30)
    disease_label = c1.selectbox("Screening For", ["Diabetes","Breast Cancer","Heart Disease","Kidney Disease"])
    bmi = c2.number_input("BMI", 10.0, 50.0, 22.0)
    glucose = c1.number_input("Glucose Level", 40.0, 350.0, 90.0)
    bp = c2.number_input("Blood Pressure", 40.0, 220.0, 120.0)
    hr = c1.number_input("Heart Rate", 30, 240, 72)
    temp = c2.number_input("Body Temp (°F)", 94.0, 105.0, 98.6)
    agree = c1.checkbox("I certify the input is correct")
    submitted = st.form_submit_button("Predict Risk")
st.markdown("</div>", unsafe_allow_html=True)

# --- VITALS (navy-gold pills, stretched wide)
v1,v2,v3,v4=st.columns(4)
v1.markdown(f"<div class='vital-pill'>💙 HR<br>{hr} bpm</div>", unsafe_allow_html=True)
v2.markdown(f"<div class='vital-pill'>⚡ BP<br>{bp} mmHg</div>", unsafe_allow_html=True)
v3.markdown(f"<div class='vital-pill'>🧬 GLU<br>{glucose} mg/dL</div>", unsafe_allow_html=True)
v4.markdown(f"<div class='vital-pill'>🌡️ TEMP<br>{temp} °F</div>", unsafe_allow_html=True)

# --- RESULTS, in a big, wide, colored card bar
if submitted:
    if not agree:
        st.warning("Please certify your input before submitting.")
    else:
        ART = Path("artifacts")
        disease = {"Diabetes":"diabetes","Breast Cancer":"breast","Heart Disease":"heart","Kidney Disease":"kidney"}[disease_label]
        features = {"Age":age,"BMI":bmi,"Glucose":glucose,"BP":bp,"HR":hr,"Temp":temp}
        with st.spinner("Analyzing profile..."):
            time.sleep(1.1)
            try:
                vec = joblib.load(ART / f"{disease}_vectorizer.joblib")
                model = joblib.load(ART / f"{disease}_calibrated_model.joblib")
                X = vec.transform([features])
                prob = float(model.predict_proba(X)[0][1])
                if prob >= 0.7: cls,head,emoji,text = "result-card-high","High Risk","🛑","Immediate clinical evaluation needed."
                elif prob >= 0.4: cls,head,emoji,text = "result-card-med","Moderate Risk","⚠️","Please consult your doctor soon."
                else: cls,head,emoji,text = "result-card-low","Low Risk","🟢","Maintain healthy lifestyle!"
                st.markdown(f"""
                    <div class='result-wide {cls}'><h2 style='text-align:center;'>{emoji} {head}</h2>
                    <p style='font-size:2.15em; text-align:center; font-weight:600;margin:18px 0 17px 0;color:inherit'>Probability: <b>{round(prob*100,1)}%</b></p>
                    <div style='text-align:center;font-size:1.14em;font-weight:600;margin-bottom:6px;'>{text}</div>
                    <progress value='{int(prob*100)}' max='100' style='width:85%;height:22px;margin-left:7%;border-radius:8px;'></progress>
                    </div>
                """, unsafe_allow_html=True)
                st.success(f"Prediction completed: {head} for {disease_label}.", icon="✅")
            except Exception as ex:
                st.error(f"Prediction failed: {ex}")

# --- FOOTER
st.markdown("""
<div style='text-align:center; padding:22px 0 14px 0; color:#23296d; font-size:1.01em; background:#fff2; border-top:3px solid #FFD700; margin-top:20px;'>
© 2025 MediTrack Health &mdash; AI Clinical Screening System. For informational use only.
</div>
""", unsafe_allow_html=True)

