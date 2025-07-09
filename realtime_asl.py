import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib  # optional if you saved your encoder

# ----------------------------
# 1. Model Definition (same as training)
# ----------------------------

class ASLClassifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=25):
        super(ASLClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# 2. Load Model and LabelEncoder
# ----------------------------

# Load your trained model
model = ASLClassifier()
model.load_state_dict(torch.load('asl_model.pt', map_location=torch.device('cpu')))
model.eval()

# Recreate your label encoder (or load it if you saved it)
letters = [chr(ord('A') + i) for i in range(26)]  # ['A', 'B', ..., 'Z']
letters.remove('Q')
le = LabelEncoder()
le.fit(letters)

# ----------------------------
# 3. Init MediaPipe & OpenCV
# ----------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ----------------------------
# 4. Real-Time Loop
# ----------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural interaction
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            wrist = hand_landmarks.landmark[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
            # this is for the label of the hand
            # not implemented right now.
            landmarks.insert(0, 0)

            if len(landmarks) == 64:
                # Convert to tensor
                input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)

                # Predict
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    predicted_class_idx = predicted.item()         # class index
                    confidence = confidence.item()                  # confidence as float (0–1)
                    predicted_letter = le.inverse_transform([predicted_class_idx])[0]

                # Display prediction
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
                cv2.putText(frame, f'ASL: {predicted_letter}', (10, 70), font, 2, color, 3)
                cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 130), font, 1, color, 2)

    cv2.imshow('ASL Realtime Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
