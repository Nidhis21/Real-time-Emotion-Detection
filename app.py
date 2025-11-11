import streamlit as st
import cv2
import tempfile
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from model import EmotionCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN()
model.load_state_dict(torch.load('best_emotion_model.pth', map_location=device))
model.eval()
model.to(device)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def process_and_draw(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    with torch.no_grad():
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_tensor = transform(roi).unsqueeze(0).to(device)
            outputs = model(roi_tensor)
            _, pred = torch.max(outputs, 1)
            emotion = emotions[pred.item()]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return frame

st.title("Emotion Detection ")
st.markdown("Choose an option below:")

option = st.radio(
    "Choose input type:",
    ["Image Upload", "Video Upload", "Webcam Photo"]
)

if option == "Image Upload":
    uploaded_img = st.file_uploader("Choose an Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_img is not None:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        processed_img = process_and_draw(img)
        st.image(processed_img, channels="BGR", caption="Result")
        is_success, buffer = cv2.imencode(".jpg", processed_img)
        if is_success:
            st.download_button(
                label="Download Result",
                data=buffer.tobytes(),
                file_name="emotion_result.jpg",
                mime="image/jpeg"
            )

elif option == "Video Upload":
    uploaded_vid = st.file_uploader("Choose a Video", type=['mp4', 'avi'])
    if uploaded_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_path = tfile.name + '_output.avi'
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        st.info("Processing video... (This may take time for large files)")
        frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = process_and_draw(frame)
            out.write(processed_frame)
            frames += 1
            if frames % 50 == 0:
                st.write(f"Processed {frames} frames...")

        cap.release()
        
        st.success("Video processing complete!")
        
        with open(out_path, 'rb') as f:
            st.download_button(
                label="Download Result Video",
                data=f.read(),
                file_name="emotion_result.avi",
                mime="video/x-msvideo"
            )

elif option == "Webcam Photo":
    camera_img = st.camera_input("Take a photo with your webcam")
    if camera_img is not None:
        img = Image.open(camera_img)
        img = np.array(img)
        if img.shape[-1] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        processed_img = process_and_draw(img)
        st.image(processed_img, channels="BGR", caption="Webcam Result")
        is_success, buffer = cv2.imencode(".jpg", processed_img)
        if is_success:
            st.download_button(
                label="Download Result",
                data=buffer.tobytes(),
                file_name="emotion_webcam.jpg",
                mime="image/jpeg"
            )
