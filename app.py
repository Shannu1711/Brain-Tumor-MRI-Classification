import streamlit as st
import tensorflow as tf
import numpy as np
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
            max-width: 1200px;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(124,58,237,0.18));
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
            background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(16,185,129,0.12));
            border: 1px solid rgba(34,197,94,0.25);
            border-radius: 20px;
            padding: 1.5rem;
            margin-top: 1rem;
        }

        .metric-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            text-align: center;
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
# MODEL
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
# PREPROCESSING
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
            Upload a brain MRI image and receive an AI-powered prediction
            with class-wise confidence scores using MobileNetV2 transfer learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------
left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.markdown("### 📤 Upload MRI Image")

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded MRI Scan",
            use_container_width=True
        )

        predict_clicked = st.button("🔍 Predict Tumor Type")

    else:
        st.markdown(
            """
            <div class="info-card">
                <b>How to use:</b><br><br>
                1. Upload a brain MRI image<br>
                2. Click <b>Predict Tumor Type</b><br>
                3. Review the predicted class and confidence scores
            </div>
            """,
            unsafe_allow_html=True
        )
        predict_clicked = False

with right_col:
    st.markdown("### 📊 Prediction Result")

    if uploaded_file is None:
        st.markdown(
            """
            <div class="info-card">
                <h4>No prediction yet</h4>
                <p class="small-muted">
                    Upload an MRI image to view the classification result here.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif predict_clicked:

        processed_image = preprocess_image(image)

        with st.spinner("Analyzing MRI image..."):
            prediction = model.predict(
                processed_image,
                verbose=0
            )[0]

        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = float(prediction[predicted_index] * 100)

        st.markdown(
            f"""
            <div class="result-card">
                <h3>Prediction</h3>
                <h2>{predicted_class}</h2>
                <p class="small-muted">
                    The model assigned the highest probability to this class.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        with metric2:
            st.metric(
                "Model",
                "MobileNetV2"
            )

        st.markdown("### Class Confidence Scores")

        for name, probability in zip(class_names, prediction):
            score = float(probability * 100)

            st.write(f"**{name}** — {score:.2f}%")
            st.progress(min(int(score), 100))

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; font-size:0.9rem;">
        Brain Tumor MRI Image Classification |
        Deep Learning • TensorFlow • MobileNetV2 • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)