import cv2
from torchvision import transforms
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import EmotionCNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN()
model.load_state_dict(torch.load('best_emotion_model.pth', map_location=device))
model.eval()
model.to(device)

# Face detector from OpenCV
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Transformation pipeline (should match training transforms)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Open webcam stream
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    with torch.no_grad():
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_tensor = transform(face_img).unsqueeze(0).to(device)  # add batch dimension

            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion = emotions[predicted.item()]

            # Draw rectangle and label on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (255, 0, 0), 2)

    # Show the video frame
    cv2.imshow('Webcam Emotion Detection - Press Q to quit', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



