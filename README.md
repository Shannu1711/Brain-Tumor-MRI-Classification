# 🧠 Brain Tumor MRI Image Classification

A deep learning-based medical imaging project that classifies brain MRI images into four categories using **MobileNetV2 Transfer Learning** and deploys the trained model through a **Streamlit web application**.

---

## 🚀 Live Demo

👉 **Try the deployed Streamlit app here:**

https://brain-tumor-mri-classification-fkwciu5eztjenzhjdccxbh.streamlit.app

---

## 📌 Project Overview

This project was developed to classify brain MRI images into the following categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

Two deep learning approaches were evaluated:

- Custom CNN
- MobileNetV2 Transfer Learning

The final model was deployed using Streamlit so users can upload an MRI image and receive a predicted tumor class along with confidence scores.

---

## 📊 Model Performance

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| Custom CNN | 76.88% | 0.7992 |
| MobileNetV2 | 82.00% | 0.5460 |

### Final Selected Model
**MobileNetV2**

MobileNetV2 was selected because it achieved better overall performance with higher accuracy and lower test loss.

---

## 🧪 Classification Performance

MobileNetV2 achieved:

- Accuracy: **82.00%**
- Macro Precision: **0.83**
- Macro Recall: **0.82**
- Macro F1-score: **0.81**

---

## 🖥️ Streamlit Application

The deployed application allows users to:

- Upload brain MRI images
- Preview the uploaded scan
- Predict the tumor category
- View prediction confidence
- View class-wise confidence scores

### Supported Classes

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

---

## 🧠 Model Architecture

The final model uses:

- MobileNetV2
- ImageNet pretrained weights
- Global Average Pooling
- Dense layer
- Dropout
- Softmax classification

Input image size:

`224 x 224 x 3`

---

## 🔄 Project Workflow

1. Dataset exploration
2. Image preprocessing
3. Image resizing
4. Pixel normalization
5. Data augmentation
6. Custom CNN model development
7. MobileNetV2 transfer learning
8. Model training
9. Model evaluation
10. Confusion matrix analysis
11. Model comparison
12. Streamlit deployment

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pillow
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Google Colab
- GitHub

---

## 📂 Project Structure

```text
Brain-Tumor-MRI-Classification/
│
├── app.py
├── best_mobilenetv2.h5
├── requirements.txt
└── README.md


⚠️ Disclaimer
This project is developed for educational and demonstration purposes only.
The application is not intended for real-world clinical diagnosis and should not be used as a substitute for professional medical advice, diagnosis, or treatment.