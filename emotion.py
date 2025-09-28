import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os

# Ensure correct path for model file
MODEL_PATH = r"d:\I'm Batman\Emotion detection\fer2013_mini_XCEPTION.102-0.66.hdf5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = load_model(MODEL_PATH, compile=False)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not accessible")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)

            x, y = max(0, x), max(0, y)
            face_img = frame[y:y+h, x:x+w]

            if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                continue

            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64))
            roi = gray.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            preds = model.predict(roi, verbose=0)[0]
            label = emotion_labels[np.argmax(preds)]
            confidence = np.max(preds)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection (MediaPipe + Keras)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
