import streamlit as st
import pandas as pd
import joblib
import time

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Hepatitis Risk Prediction",
    layout="wide"
)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("artifacts/best_model.pkl")

model = load_model()

# =========================
# Helper Mappings
# =========================
YES_NO = {"No": 2, "Yes": 1}
SEX_MAP = {"Male": 1, "Female": 2}

# =========================
# UI Header
# =========================
st.title("🩺 Hepatitis Disease Risk Prediction")
st.caption("Clinical Decision Support Tool (Educational Use Only)")

# =========================
# Sidebar – Patient Info
# =========================
with st.sidebar:
    st.header("🧑 Patient Information")

    age = st.number_input("Age (years)", min_value=1, max_value=100, value=50)

    sex = st.selectbox("Sex", ["Male", "Female"])

    st.subheader("🩸 Clinical Symptoms")
    fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    malaise = st.selectbox("Malaise (general discomfort)", ["No", "Yes"])
    anorexia = st.selectbox("Loss of appetite (Anorexia)", ["No", "Yes"])

    st.subheader("🧠 Physical Examination Findings")
    liver_big = st.selectbox("Enlarged Liver", ["No", "Yes"])
    liver_firm = st.selectbox("Firm Liver", ["No", "Yes"])
    spleen_palpable = st.selectbox("Enlarged Spleen", ["No", "Yes"])
    spiders = st.selectbox("Spider Angiomas", ["No", "Yes"])
    ascites = st.selectbox("Ascites (fluid in abdomen)", ["No", "Yes"])
    varices = st.selectbox("Esophageal Varices", ["No", "Yes"])

    st.subheader("💊 Medical History")
    steroid = st.selectbox("Steroid Treatment", ["No", "Yes"])
    antivirals = st.selectbox("Antiviral Therapy", ["No", "Yes"])

    st.subheader("🧪 Lab Test Results")
    bilirubin = st.number_input("Bilirubin (mg/dL) [Normal <1.2]", value=1.0)
    alk_phosphate = st.number_input("Alkaline Phosphatase (IU/L)", value=85.0)
    sgot = st.number_input("SGOT / AST (IU/L)", value=30.0)
    albumin = st.number_input("Albumin (g/dL) [Normal 3.5–5.0]", value=4.0)
    protime = st.number_input("Prothrombin Time (seconds)", value=10.0)

    histology = st.selectbox("Histology (Biopsy Performed)", ["No", "Yes"])

    predict_btn = st.button("🔍 Assess Risk")

# =========================
# Prediction Logic
# =========================
if predict_btn:
    input_data = {
        "age": age,
        "sex": SEX_MAP[sex],
        "steroid": YES_NO[steroid],
        "antivirals": YES_NO[antivirals],
        "fatigue": YES_NO[fatigue],
        "malaise": YES_NO[malaise],
        "anorexia": YES_NO[anorexia],
        "liver_big": YES_NO[liver_big],
        "liver_firm": YES_NO[liver_firm],
        "spleen_palpable": YES_NO[spleen_palpable],
        "spiders": YES_NO[spiders],
        "ascites": YES_NO[ascites],
        "varices": YES_NO[varices],
        "bilirubin": bilirubin,
        "alk_phosphate": alk_phosphate,
        "sgot": sgot,
        "albumin": albumin,
        "protime": protime,
        "histology": YES_NO[histology],
    }

    df = pd.DataFrame([input_data])

    start = time.time()
    prob = model.predict_proba(df)[0]
    pred = model.predict(df)[0]
    latency = (time.time() - start) * 1000

    # =========================
    # Results
    # =========================
    st.subheader("📊 Risk Assessment Result")

    if pred == 1:
        st.success(f"🟢 Lower Risk Profile — Likely Survival ({prob[1]*100:.2f}%)")
    else:
        st.error(f"🔴 High Risk Profile — Increased Mortality Risk ({prob[0]*100:.2f}%)")

    st.metric("Inference Time (ms)", f"{latency:.2f}")

    # =========================
    # Explanation (RULE-BASED)
    # =========================
    st.subheader("🧠 Why this prediction?")

    reasons = []

    if bilirubin > 1.5:
        reasons.append("Elevated bilirubin indicates impaired liver function.")
    if albumin < 3.5:
        reasons.append("Low albumin suggests poor liver protein synthesis.")
    if ascites == "Yes":
        reasons.append("Ascites is a marker of advanced liver disease.")
    if varices == "Yes":
        reasons.append("Varices increase risk of life-threatening bleeding.")
    if protime > 12:
        reasons.append("Prolonged prothrombin time indicates clotting dysfunction.")

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("• No major high-risk clinical indicators detected.")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption(
    "⚠️ This tool is for educational and research purposes only. "
    "It is NOT a substitute for professional medical advice."
)
