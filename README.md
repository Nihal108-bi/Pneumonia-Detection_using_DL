# 🫁 Pneumonia Detection Using Deep Learning

This project implements a deep learning-based web application for detecting **Pneumonia** from chest X-ray images using a fine-tuned **VGG19** model. The frontend is built using **Streamlit**, enabling an intuitive interface for uploading images and receiving predictions.

---

## 🚀 Features

- 🔍 Predicts whether an X-ray image is **Normal** or indicates **Pneumonia**
- 🧠 Uses a **pre-trained VGG19 model** (ImageNet weights)
- 📷 Supports grayscale or RGB X-ray images
- 🖼️ Resizes and normalizes images before inference
- 🌐 Simple **web interface** using **Streamlit**
- 📁 Upload and analyze X-ray images in real-time

---

## 🧠 Model Architecture

- Base Model: `VGG19` (`include_top=False`, pre-trained on ImageNet)
- Additional Layers:
  - `Flatten`
  - `Dense(4608, relu)`
  - `Dropout(0.2)`
  - `Dense(1152, relu)`
  - `Dense(2, softmax)` – for binary classification (Pneumonia / Normal)

---

## For Running this app
streamlit run app.py

## 🖼️ Using the App
1. Open the web app in your browser (http://localhost:8501)

2. Click “Browse files” and upload a chest X-ray image (preferably .jpg or .png)

3. The model will output either:

 Normal

 Pneumonia



## 📊 Dataset

The model was trained on the **Kaggle Chest X-Ray Images (Pneumonia)** dataset:

* [https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## ✍️ Author

* **Nihal Jaiswal**
* GitHub: [Nihal108-bi](https://github.com/Nihal108-bi)
* LinkedIn: [Nihal Jaiswal](https://www.linkedin.com/in/nihal-jaiswal-908b52257/)

---




