import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the HandTracker class.
        
        Args:
            max_num_hands (int): Maximum number of hands to detect
            min_detection_confidence (float): Minimum confidence for hand detection
            min_tracking_confidence (float): Minimum confidence for hand tracking
        """
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure hands detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize webcam
        self.cap = None
        self.is_running = False
        
    def start_camera(self, camera_index=0):
        """
        Start the camera capture.
        
        Args:
            camera_index (int): Index of the camera to use (default: 0)
        
        Returns:
            bool: True if camera opened successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return False
        return True
        
    def process_frame(self, frame):
        """
        Process a single frame for hand detection.
        
        Args:
            frame (numpy.ndarray): Input frame from camera
            
        Returns:
            tuple: (processed_frame, hand_landmarks, handedness)
        """
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB (MediaPipe uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for hand detection
        results = self.hands.process(rgb_frame)
        
        return frame, results.multi_hand_landmarks, results.multi_handedness
        
    def draw_landmarks(self, frame, hand_landmarks, handedness=None):
        """
        Draw hand landmarks and connections on the frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            hand_landmarks: MediaPipe hand landmarks
            handedness: MediaPipe handedness classification
        """
        if hand_landmarks:
            for idx, landmarks in enumerate(hand_landmarks):
                # Draw landmarks and connections
                self.mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Add hand classification (Left/Right)
                if handedness and idx < len(handedness):
                    hand_label = handedness[idx].classification[0].label
                    
                    # Get wrist position for text placement
                    wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    
                    h, w, _ = frame.shape
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                    
                    # Draw hand label
                    cv2.putText(frame, hand_label, (wrist_x - 30, wrist_y - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def get_landmark_positions(self, hand_landmarks, frame_shape):
        """
        Get normalized landmark positions for a hand.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            list: List of (x, y) coordinates for each landmark
        """
        if not hand_landmarks:
            return []
        
        h, w = frame_shape[:2]
        positions = []
        
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            positions.append((x, y))
            
        return positions
    
    def run(self, camera_index=0, show_instructions=True):
        """
        Run the hand tracking application.
        
        Args:
            camera_index (int): Index of the camera to use
            show_instructions (bool): Whether to show instructions on screen
        """
        if not self.start_camera(camera_index):
            return
        
        self.is_running = True
        print("Hand tracking started. Press 'q' to quit.")
        
        while self.is_running:
            # Read frame from webcam
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame for hand detection
            processed_frame, hand_landmarks, handedness = self.process_frame(frame)

            if not hand_landmarks:
                h, w, _ = frame.shape
                
                cv2.putText(processed_frame, "No hands detected", (int(w/2), int(h/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Hand Tracking', processed_frame)
                
            if hand_landmarks:
                for idx, landmarks in enumerate(hand_landmarks):
                    print(idx)
                     # Add hand classification (Left/Right)
                    if handedness and idx < len(handedness):
                        hand_label = handedness[idx].classification[0].label
                        
                        # Get wrist position for text placement
                        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        print(wrist.x, wrist.y, wrist.z)
                        print(type(landmarks))
                        print(type(landmarks.landmark))
                        print(landmarks.landmark[1].x, landmarks.landmark[1].y, landmarks.landmark[1].z)
                        for idx1, lm in enumerate(landmarks.landmark):
                            # Get landmark positions relative to wrist
                            x_wrist = lm.x - wrist.x
                            y_wrist = lm.y - wrist.y
                            z_wrist = lm.z - wrist.z

                            # Print landmark positions relative to wrist
                            #print(f"Landmark {idx1}: x={x_wrist}, y={y_wrist:.2f}, z={z_wrist:.2f}, x1={lm.x}")

            # Draw landmarks
            self.draw_landmarks(processed_frame, hand_landmarks, handedness)
            
            # Add instructions
            if show_instructions:
                cv2.putText(processed_frame, "Press 'q' to quit", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Hand Tracking', processed_frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
        
        print("Hand tracking stopped.")
    
    def stop(self):
        """Stop the hand tracking."""
        self.is_running = False
    
    def release(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        self.release()


# Example usage
if __name__ == "__main__":
    # Create hand tracker instance
    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Run the tracker
    try:
        tracker.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        tracker.release()
        
    # Alternative usage - process individual frames
    """
    tracker = HandTracker()
    if tracker.start_camera():
        ret, frame = tracker.cap.read()
        if ret:
            processed_frame, landmarks, handedness = tracker.process_frame(frame)
            tracker.draw_landmarks(processed_frame, landmarks, handedness)
            cv2.imshow('Frame', processed_frame)
            cv2.waitKey(0)
        tracker.release()
    """