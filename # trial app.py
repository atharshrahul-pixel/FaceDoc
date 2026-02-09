# app.py
import streamlit as st
from model_utils import analyze_image

# --- Basic page config ---
st.set_page_config(page_title="FaceDoc AI Prototype (OpenCV)", page_icon="🧠", layout="centered")

# Disclaimer + header
st.title("🧠 FaceDoc — Visual Health Chatbot (Prototype)")
st.markdown(
    """
<div style="background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px;">
⚠️ <strong>Prototype only:</strong> This tool uses image heuristics and is <u>not</u> a medical diagnostic system.
Do not rely on it for health decisions. Consult a licensed medical professional for clinical concerns.
</div>
""", unsafe_allow_html=True
)

# session state for multi-turn flow
if 'stage' not in st.session_state:
    st.session_state.stage = 0
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'model_output' not in st.session_state:
    st.session_state.model_output = None
if 'mapped_condition' not in st.session_state:
    st.session_state.mapped_condition = None
if 'user_response' not in st.session_state:
    st.session_state.user_response = None

# Follow-up questions mapping
KEY_SYMPTOMS = {
    "Inflammation / Allergy": "Have you noticed itchiness, sudden onset after exposure, or breathing difficulty?",
    "Eye Strain / Inflamed Eye (possible conjunctival redness)": "Do you have eye pain, discharge, or sensitivity to light?",
    "Skin Issue / Lesions (possible acne, spots, or pigmented lesions)": "Is the area painful, bleeding, or rapidly changing?",
    "Paleness - (possible anemia / fatigue)": "Have you been feeling unusually tired, dizzy, or short of breath recently?",
    # mild/default mappings
    "Mild Inflammation / Irritation": "Is this irritation accompanied by itching or spreading?",
}

DEFAULT_QUESTION = "Aside from the image, are you experiencing any other symptoms like fever, pain, or dizziness?"

# Stage 0: upload
if st.session_state.stage == 0:
    st.markdown("### Step 1 — Upload a clear photo of your face")
    prompt = "Upload a front-facing photo (jpg/jpeg/png). Good lighting helps."
    uploaded_file = st.file_uploader(prompt, type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.stage = 1
        st.rerun()

# Stage 1: analyze
elif st.session_state.stage == 1:
    st.markdown("### Step 2 — Analysis")
    uploaded_file = st.session_state.uploaded_file
    if uploaded_file is None:
        st.session_state.stage = 0
        st.experimental_rerun()
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
    with st.spinner("Analyzing image (OpenCV heuristics)..."):
        result = analyze_image(uploaded_file)
    st.success("Analysis complete")
    st.session_state.model_output = result

    # parse lines
    try:
        raw_line, cond_line = result.split("\n")
    except ValueError:
        raw_line = result
        cond_line = ""
    st.markdown(f"**Model output:** `{raw_line}`")
    st.markdown(f"**Suggestion:** {cond_line}")
    # store simpler mapped condition for follow-up lookup
    # we store the human-readable condition part before parentheses
    mapped = cond_line.split("(")[0].replace("Possible condition: ", "").strip()
    st.session_state.mapped_condition = mapped

    # move to next step
    st.session_state.stage = 2
    st.experimental_rerun()

# Stage 2: follow-up question
elif st.session_state.stage == 2:
    st.markdown("### Step 3 — Quick follow-up")
    st.image(st.session_state.uploaded_file, caption="Image context", use_column_width=True)
    condition = st.session_state.mapped_condition or ""
    q = KEY_SYMPTOMS.get(condition, DEFAULT_QUESTION)
    st.markdown(f"**FaceDoc:** {q}")

    col1, col2 = st.columns(2)
    if col1.button("Yes, I do."):
        st.session_state.user_response = "Yes"
        st.session_state.stage = 3
        st.experimental_rerun()
    if col2.button("No, not really."):
        st.session_state.user_response = "No"
        st.session_state.stage = 3
        st.experimental_rerun()

# Stage 3: summary
elif st.session_state.stage == 3:
    st.markdown("### Step 4 — Summary")
    user_resp = st.session_state.user_response
    cond = st.session_state.mapped_condition or "No condition detected"
    if user_resp == "Yes":
        st.info(f"Confirmation: You reported symptoms consistent with **{cond}**. Please consult a clinician for further evaluation.")
    else:
        st.info(f"Confirmation: You did not report further symptoms for **{cond}**. If you are worried, consult a clinician.")
    st.markdown("---")
    st.warning("This is a prototype. For any medical concern, seek professional healthcare advice.")
    if st.button("Start new scan"):
        # reset minimal state
        st.session_state.stage = 0
        st.session_state.uploaded_file = None
        st.session_state.model_output = None
        st.session_state.mapped_condition = None
        st.session_state.user_response = None
        st.experimental_rerun()
