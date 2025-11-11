# Emotion Detection via Deep Learning

## Overview
This project is a deep learning-based system for detecting human emotions from facial images and video frames. It uses a custom Convolutional Neural Network (CNN) to classify faces as Angry, Disgust, Fear, Happy, Sad, Surprise, or Neutral. The solution includes:
- Training and evaluation of the model in Google Colab
- Real-time inference and annotation on images, videos, and webcam photos
- An interactive Streamlit web app for file and webcam uploads, with remote access enabled by ngrok

## Features
- **Face Detection:** Identifies faces in input images and videos using Haarcascade (OpenCV)
- **Emotion Recognition:** Classifies each detected face into one of seven emotions using a trained CNN
- **Interactive App:** Upload images and videos or take webcam photos for instant emotion analysis and download results
- **Visualization:** Annotates emotions directly on faces for easy feedback

## Data Used
- **Dataset:** FER-2013 (Facial Expression Recognition), CSV format with 48x48 grayscale face images for seven emotion classes
- **Inputs:** User-provided images or video files; real-time webcam photos; demo video samples

## Tech Stack
- **Programming:** Python
- **Data & Model:** NumPy, Pandas, PyTorch, TorchVision
- **Image/Video Processing:** OpenCV, PIL (Pillow)
- **Interactive Web App:** Streamlit
- **Environment:** Google Colab (training & experimentation), VS Code (development)

## What I Learned (Model Focus)
- Designing custom CNN architectures for image classification
- Image preprocessing: normalization, resizing, grayscale conversion to improve model performance
- Model training, evaluation, and validation for accuracy
- Real-time inference deployment using PyTorch
- Saving and loading model checkpoints for production use
- Building interactive demos with Streamlit

## How to Run

### 1. Training/Experimentation in Colab
- Upload FER-2013 dataset and train the CNN using provided Colab notebooks
- Save best model as `best_emotion_model.pth`

### 2. Launch Interactive App (Colab or Local)
- Place the trained model file (`best_emotion_model.pth`) and code in your working directory
- Open terminal and run:
  ```bash
  streamlit run app.py
  ```

### 3. Use the App
- Upload images or videos, or capture a webcam photo via browser
- See immediate predictions and download annotated results

### 4. For Real-Time Webcam Detection:
- You can **directly run** the `webcam_emotion_detection.py` script for live webcam emotion inference on your local machine:
  ```bash
  python webcam_emotion_detection.py
  ```

## AI/ML Details
- Core Model: Convolutional Neural Network (CNN)
- Task: Multi-class emotion classification from facial images
- Supported Emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Real-time and batch prediction supported
