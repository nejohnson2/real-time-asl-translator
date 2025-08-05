import cv2
import os
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib  # optional if you saved your encoder

#model_path = os.path.join('..', 'models', 'asl_model.pt')
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'asl_model.pt')
model_path = os.path.abspath(model_path)
print(f"Loading model from {model_path}")

# Mouse callback function
def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Trigger when mouse moves
        print(f"X: {x}, Y: {y}")

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
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

# If adjusting the window size, uncomment the next line
SCREEN_W, SCREEN_H = 1920, 1080  # enter actual screen resolution

cv2.namedWindow("ASL Realtime Recognition", cv2.WND_PROP_FULLSCREEN)
#cv2.setMouseCallback('ASL Realtime Recognition', show_coordinates)
# ----------------------------
# 4. Real-Time Loop
# ----------------------------

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    frame_aspect = w / h
    screen_aspect = SCREEN_W / SCREEN_H
    # Scale while keeping aspect ratio
    if frame_aspect > screen_aspect:
        # Match screen width
        new_w = SCREEN_W
        new_h = int(SCREEN_W / frame_aspect)
    else:
        # Match screen height
        new_h = SCREEN_H
        new_w = int(SCREEN_H * frame_aspect)
    
    if not ret:
        break

    # Flip for natural interaction
    frame = cv2.flip(frame, 1)

    # Font for displaying text
    font = cv2.FONT_HERSHEY_SIMPLEX
    #predicted_letter = ''
    #confidence = 0.0
    #color = (200, 200, 200)  # Light grey

    # Convert to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_draw.draw_landmarks(frame, 
                                   hand_landmarks, 
                                   mp_hands.HAND_CONNECTIONS,
                                   mp_drawing_styles.get_default_hand_landmarks_style(),
                                   mp_drawing_styles.get_default_hand_connections_style()
            )

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

                if confidence < 0.6:
                    predicted_letter = ''
                    confidence = 0.0

                # Display prediction
                y_values = [lm.y for lm in hand_landmarks.landmark]
                min_y = min(y_values)
                min_y = int(min_y * h)

                x_values = [lm.x for lm in hand_landmarks.landmark]
                avg_x = sum(x_values) / len(x_values)
                avg_x = int(avg_x * w)
                #mft_x, mft_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
                cv2.putText(frame, predicted_letter, (avg_x, min_y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                #color = (0, 255, 0) if confidence > 0.6 else (0, 0, 255)
     
                cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, h-20), 
                            font, 1, (255, 255, 255), 1)
    else:
        # If no hand detected, reset predictions
        predicted_letter = ''
        confidence = 0.0
        color = (255, 255, 255)  # Light grey

    # -- 
    # Draw the HUD
    # --
    hud_color = (200, 200, 200)  # Light grey
    # Create overlay (copy of the frame)
    overlay = frame.copy()

    alpha = 0.4  # Transparency factor (0: fully transparent, 1: fully opaque)

    # --
    # Draw the top bar
    # --
    top_bar_height = 50
    text = 'ASL Translator'
    font_scale = 1
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    Y_OFFSET = int((top_bar_height/2) + (text_height/2))
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], top_bar_height), hud_color, -1)  # Top bar
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], top_bar_height), (255, 255, 255), 1)  # Top bar border
    cv2.putText(frame, text, (10, Y_OFFSET), 
                font, font_scale, (255, 255, 255), thickness)

    text = 'Press "ESC" to quit'
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    Y_OFFSET = int((top_bar_height/2) + (text_height/2))
    cv2.putText(frame, text, (w - text_width - 10, Y_OFFSET), font, font_scale, (255, 255, 255), thickness)
    """"
    # --
    # Draw the results HUD
    # --
    hud_start_point = (20, 50)
    hud_end_point = (180, 215)
    #cv2.rectangle(overlay, hud_start_point, hud_end_point, hud_color, -1)  # Results hud
    #cv2.rectangle(overlay, hud_start_point, hud_end_point, (255, 255, 255), 2)  # Results hud border
    
    # reported values
    text = f'Letter: {predicted_letter}'
    font_scale = 1
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    Y_OFFSET = hud_start_point[1] + text_height + 10
    #Y_OFFSET = int((40/2) + (text_height/2))
    cv2.putText(frame, text, (hud_start_point[0] + 10, Y_OFFSET), font, 1, color, 2)
    cv2.putText(frame, f'Confidence: {confidence:.2f}', 
                (hud_start_point[0]+10, hud_start_point[1]+60), font, 0.5, color, 1)
    """
    # blend overlay with frame
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Add black padding to center it
    canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    x_offset = (SCREEN_W - new_w) // 2
    y_offset = (SCREEN_H - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame


    cv2.imshow('ASL Realtime Recognition', canvas)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == 27: # ESC key
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
