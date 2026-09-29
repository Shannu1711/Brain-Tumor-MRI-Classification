
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Image Classification")

st.write("""
Upload a brain MRI image and the trained deep learning model will classify it into one of the following categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor
""")

st.info(
    "This application uses a MobileNetV2 transfer learning model for MRI image classification."
)

st.warning(
    "Educational project only. This application is not a substitute for professional medical diagnosis."
)

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

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded MRI Image")

    st.image(
        image,
        caption="Uploaded Brain MRI",
        use_container_width=True
    )

    if st.button("Predict Tumor Type"):

        processed_image = preprocess_image(image)

        with st.spinner("Analyzing MRI image..."):

            prediction = model.predict(
                processed_image,
                verbose=0
            )[0]

        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = float(prediction[predicted_index] * 100)

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Class Confidence Scores")

        for name, probability in zip(class_names, prediction):

            score = float(probability * 100)

            st.write(
                f"{name}: {score:.2f}%"
            )

            st.progress(
                min(int(score), 100)
            )

st.markdown("---")

st.caption(
    "Brain Tumor MRI Image Classification | MobileNetV2 Transfer Learning"
)
