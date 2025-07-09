import cv2
import mediapipe as mp
import csv
import os

data_path = os.path.join('data', 'asl_landmarks.csv')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

data = []  # List to store samples

while True:
    ret, frame = cap.read()
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get wrist position for text placement
            wrist = hand_landmarks.landmark[0]
            handedness = results.multi_handedness[0] # if multiple hand are tracked, this only gets the first one
            
            hand_label = handedness.classification[0].label
            hand_label = 0 if hand_label == 'Right' else 1
            print(hand_label)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

            # Draw landmarks on frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Optional: Show current landmarks
            cv2.putText(frame, f'Collected: {len(data)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Collecting Hand Landmarks', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Save with label if key pressed
    elif key >= ord('a') and key <= ord('z'):
        label = chr(key).upper()
        if results.multi_hand_landmarks:
            data.append([label, hand_label] + landmarks)
            print(f'Saved sample for label: {label}')

cap.release()
cv2.destroyAllWindows()

file_exists = os.path.isfile(data_path)
with open(data_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    header = ['label', 'hand_label'] + [f'{axis}{i}' for i in range(1, 22) for axis in ('x', 'y', 'z')]
    if not file_exists:
        writer.writerow(header)
    writer.writerows(data)

print(f"Saved {len(data)} samples to {data_path}")
