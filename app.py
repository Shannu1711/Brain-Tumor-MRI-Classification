import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #0b1120 100%);
            color: #f8fafc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 22px;
            background: linear-gradient(
                135deg,
                rgba(37,99,235,0.22),
                rgba(124,58,237,0.18)
            );
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            margin-bottom: 1.8rem;
        }

        .hero h1 {
            font-size: 2.8rem;
            margin-bottom: 0.4rem;
        }

        .hero p {
            color: #cbd5e1;
            font-size: 1.05rem;
            margin-bottom: 0;
        }

        .info-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
        }

        .result-card {
            background: linear-gradient(
                135deg,
                rgba(34,197,94,0.18),
                rgba(16,185,129,0.12)
            );
            border: 1px solid rgba(34,197,94,0.25);
            border-radius: 18px;
            padding: 1.2rem;
            margin-top: 0.8rem;
            margin-bottom: 1rem;
        }

        .warning-card {
            background: rgba(245,158,11,0.12);
            border: 1px solid rgba(245,158,11,0.25);
            border-radius: 18px;
            padding: 1rem;
            margin-top: 0.8rem;
        }

        .small-muted {
            color: #94a3b8;
            font-size: 0.9rem;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            height: 3rem;
            font-weight: 600;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            color: white;
            border: none;
        }

        div.stButton > button:hover {
            border: none;
            color: white;
            transform: translateY(-1px);
        }

        section[data-testid="stSidebar"] {
            background-color: #0b1220;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("best_mobilenetv2.h5")

model = load_trained_model()

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# ---------------------------------------------------------
# BASIC MRI-LIKE VALIDATION
# ---------------------------------------------------------
def is_probably_mri(image):
    """
    Simple heuristic to reject obvious colorful non-MRI images.
    This is NOT a medical validation system.
    """

    image = image.convert("RGB")
    arr = np.array(image)

    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)

    color_difference = (
        np.mean(np.abs(r - g))
        + np.mean(np.abs(g - b))
        + np.mean(np.abs(r - b))
    ) / 3

    return color_difference < 18

# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.title("🧠 Project Info")

    st.markdown("### Model")
    st.write("MobileNetV2 Transfer Learning")

    st.markdown("### Classes")
    st.write("• Glioma")
    st.write("• Meningioma")
    st.write("• No Tumor")
    st.write("• Pituitary")

    st.markdown("### Model Performance")
    st.write("Custom CNN Accuracy: **76.88%**")
    st.write("MobileNetV2 Accuracy: **82.00%**")

    st.markdown("### Batch Upload")
    st.write(
        "Upload multiple MRI images together and review predictions in one run."
    )

    st.markdown("---")

    st.caption(
        "Educational project only. Not intended for clinical diagnosis."
    )

# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧠 Brain Tumor MRI Classification</h1>
        <p>
            Upload one or multiple brain MRI images and receive
            AI-powered predictions with confidence scores using
            MobileNetV2 transfer learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# UPLOAD SECTION
# ---------------------------------------------------------
st.markdown("### 📤 Upload Brain MRI Images")

st.info(
    "Upload brain MRI scans only. "
    "You can select multiple JPG, JPEG, or PNG images at once."
)

uploaded_files = st.file_uploader(
    "Choose MRI images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# ---------------------------------------------------------
# PROCESS BATCH
# ---------------------------------------------------------
if uploaded_files:

    st.write(f"**{len(uploaded_files)} image(s) selected**")

    run_prediction = st.button(
        "🔍 Analyze All Images"
    )

    if run_prediction:

        results = []

        progress_bar = st.progress(0)

        for index, uploaded_file in enumerate(uploaded_files):

            image = Image.open(uploaded_file)

            st.markdown("---")
            st.markdown(f"## {index + 1}. {uploaded_file.name}")

            left_col, right_col = st.columns(
                [0.9, 1.1],
                gap="large"
            )

            with left_col:
                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

            with right_col:

                if not is_probably_mri(image):

                    st.warning(
                        "⚠️ This image does not appear to be a typical grayscale MRI image."
                    )

                    results.append(
                        {
                            "File Name": uploaded_file.name,
                            "Prediction": "Invalid / Uncertain Input",
                            "Confidence (%)": 0.0,
                            "Status": "Rejected by basic MRI check"
                        }
                    )

                else:

                    processed_image = preprocess_image(image)

                    prediction = model.predict(
                        processed_image,
                        verbose=0
                    )[0]

                    predicted_index = int(
                        np.argmax(prediction)
                    )

                    predicted_class = (
                        class_names[predicted_index]
                    )

                    confidence = float(
                        prediction[predicted_index] * 100
                    )

                    # Extra confidence safeguard
                    if confidence < 70:

                        st.warning(
                            f"⚠️ Model confidence is only {confidence:.2f}%. "
                            "The image may be unclear, unusual, or outside the model's expected input."
                        )

                        display_prediction = (
                            f"Uncertain ({predicted_class})"
                        )

                        status = "Low confidence"

                    else:

                        display_prediction = predicted_class
                        status = "Prediction accepted"

                        st.markdown(
                            f"""
                            <div class="result-card">
                                <h3>Prediction</h3>
                                <h2>{predicted_class}</h2>
                                <p class="small-muted">
                                    Confidence: {confidence:.2f}%
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown("### Class Confidence Scores")

                    for name, probability in zip(
                        class_names,
                        prediction
                    ):
                        score = float(
                            probability * 100
                        )

                        st.write(
                            f"**{name}** — {score:.2f}%"
                        )

                        st.progress(
                            min(int(score), 100)
                        )

                    results.append(
                        {
                            "File Name": uploaded_file.name,
                            "Prediction": display_prediction,
                            "Confidence (%)": round(confidence, 2),
                            "Status": status
                        }
                    )

            progress_bar.progress(
                int(((index + 1) / len(uploaded_files)) * 100)
            )

        # -------------------------------------------------
        # BATCH SUMMARY
        # -------------------------------------------------
        st.markdown("---")
        st.markdown("## 📊 Batch Prediction Summary")

        results_df = pd.DataFrame(results)

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        accepted_count = (
            results_df["Status"]
            .eq("Prediction accepted")
            .sum()
        )

        low_confidence_count = (
            results_df["Status"]
            .eq("Low confidence")
            .sum()
        )

        rejected_count = (
            results_df["Status"]
            .eq("Rejected by basic MRI check")
            .sum()
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Images",
                len(results_df)
            )

        with col2:
            st.metric(
                "Accepted",
                int(accepted_count)
            )

        with col3:
            st.metric(
                "Low Confidence",
                int(low_confidence_count)
            )

        with col4:
            st.metric(
                "Rejected",
                int(rejected_count)
            )

else:

    st.markdown(
        """
        <div class="info-card">
            <b>How to test the application:</b><br><br>
            1. Select one or more brain MRI images<br>
            2. Click <b>Analyze All Images</b><br>
            3. Review each prediction and confidence score<br>
            4. Check the batch summary table at the bottom
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        font-size:0.9rem;
    ">
        Brain Tumor MRI Image Classification |
        Deep Learning • TensorFlow • MobileNetV2 • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)