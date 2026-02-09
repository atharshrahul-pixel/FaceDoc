import streamlit as st
from PIL import Image
import random
from googletrans import Translator

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="FaceDoc", page_icon="🩺", layout="centered")

# Initialize Translator
translator = Translator()

# ---------- LANGUAGE SELECTION ----------
lang_options = {
    "English": "en",
    "தமிழ் (Tamil)": "ta",
    "हिन्दी (Hindi)": "hi",
    "తెలుగు (Telugu)": "te",
    "മലയാളം (Malayalam)": "ml",
    "ಕನ್ನಡ (Kannada)": "kn"
}
selected_lang = st.selectbox("🌍 Select your language", list(lang_options.keys()))
lang_code = lang_options[selected_lang]

def t(text):
    """Translate text into selected language"""
    if lang_code == "en":
        return text
    try:
        return translator.translate(text, src="en", dest=lang_code).text
    except:
        return text  # fallback to English if API fails

# ---------- HEADER ----------
st.title("🩺 FaceDoc")
st.markdown(t("Upload your face image to get a quick AI-based facial wellness analysis."))

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader(t("📤 Upload an image"), type=["jpg", "jpeg", "png"])

# ---------- SIMULATED AI ANALYSIS ----------
def analyze_image(image):
    labels = ["facial_redness", "dark_circles", "paleness", "acne"]
    label = random.choice(labels)
    confidence = round(random.uniform(80, 98), 2)

    if label == "facial_redness":
        condition = t("Inflammation / Allergy / Skin Redness")
        followup = t("Have you recently experienced itching, burning, or used new skincare products?")
    elif label == "dark_circles":
        condition = t("Fatigue / Lack of Sleep / Dehydration")
        followup = t("Have you been sleeping less than 6 hours or feeling tired recently?")
    elif label == "paleness":
        condition = t("Possible Anemia / Low Iron Levels")
        followup = t("Do you feel weak or dizzy frequently?")
    else:
        condition = t("Mild Acne / Skin Oil Imbalance")
        followup = t("Do you often eat oily food or use heavy skincare products?")

    return label, confidence, condition, followup

# ---------- MAIN APP ----------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=t("📸 Uploaded Image"), use_column_width=True)

    st.subheader(t("💬 AI Analysis Result:"))

    label, confidence, condition, followup = analyze_image(image)

    st.write(f"**{t('Predicted Label')}:** {label.replace('_',' ').title()}")
    st.write(f"**{t('Possible Condition')}:** {condition} ({t('Confidence')}: {confidence}%)")
    st.markdown(f"🧠 **{t('Follow-up')}:** {followup}")

    # ---------- INTERACTIVE FOLLOW-UP ----------
    if "facial_redness" in label:
        st.markdown(t("👩‍⚕️ FaceDoc: I noticed some redness on your skin. Have you also felt warm, feverish, or unusually tired lately?"))
        col1, col2 = st.columns(2)
        with col1:
            yes = st.button(t("🔥 Yes, I feel feverish"))
        with col2:
            no = st.button(t("🌿 No, just skin redness"))

        if yes:
            st.warning(t("👩‍⚕️ FaceDoc: You might be experiencing mild fever or inflammation. Stay hydrated 💧 and rest well. If temperature rises, consult a doctor 🏥."))
            temp = st.number_input(t("🌡️ Enter your body temperature (°C):"), min_value=34.0, max_value=42.0, step=0.1)
            if temp:
                if temp >= 37.5:
                    st.error(t(f"🌡️ {temp}°C suggests a fever. Take rest and monitor symptoms closely."))
                else:
                    st.success(t(f"🌿 {temp}°C is normal. Redness likely due to skin irritation or allergy."))
        elif no:
            st.info(t("👩‍⚕️ FaceDoc: The redness appears skin-related, not due to fever. Try aloe vera gel or moisturizer 🌱."))

    elif "dark_circles" in label:
        col1, col2 = st.columns(2)
        with col1:
            yes = st.button(t("😴 Yes, tired/lack sleep"))
        with col2:
            no = st.button(t("😊 No, I sleep well"))

        if yes:
            st.warning(t("💤 Lack of rest can cause dark circles. Maintain 7–8 hours of sleep 🛏️."))
        elif no:
            st.info(t("😌 Dark circles may be genetic or due to dehydration. Drink plenty of water 💧."))

    elif "paleness" in label:
        col1, col2 = st.columns(2)
        with col1:
            yes = st.button(t("😓 Yes, I feel weak"))
        with col2:
            no = st.button(t("💪 No, I feel fine"))

        if yes:
            st.warning(t("🩸 This could indicate low iron levels. Consider an iron-rich diet or blood checkup."))
        elif no:
            st.info(t("🌞 Your skin tone seems naturally light. No health concern detected."))

    elif "acne" in label:
        col1, col2 = st.columns(2)
        with col1:
            yes = st.button(t("🍔 Yes, I eat oily food"))
        with col2:
            no = st.button(t("🥗 No, I eat clean"))

        if yes:
            st.warning(t("🍟 Try reducing oily foods. Wash face twice daily and stay hydrated 💧."))
        elif no:
            st.info(t("🌿 Acne may be hormonal. Consider consulting a dermatologist."))

else:
    st.info(t("👆 Please upload a face image to start analysis."))