# Brain Tumor MRI Image Classification

A deep learning project for classifying brain MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

## Project Overview

This project compares a Custom CNN model with a MobileNetV2 transfer learning model.

The final Streamlit application allows users to upload a brain MRI image and receive:

- Predicted tumor class
- Prediction confidence
- Confidence scores for all classes

## Model Performance

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 76.88% |
| MobileNetV2 | 82.00% |

MobileNetV2 was selected for deployment because it achieved better overall performance.

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- Deep Learning
- Transfer Learning
- Image Preprocessing
- Data Augmentation
- Streamlit
- NumPy
- Pillow

## Files

- app.py - Streamlit application
- best_mobilenetv2.h5 - trained model
- requirements.txt - project dependencies

## Run Locally

pip install -r requirements.txt

streamlit run app.py

## Disclaimer

This application is developed for educational and project demonstration purposes only.
It is not intended to replace professional medical diagnosis.