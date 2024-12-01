import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from math import hypot
import time

class HandGestureMouse:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Get screen size
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Smoothing parameters
        self.prev_x, self.prev_y = 0, 0
        self.smoothing = 0.5
        
        # Mode flags
        self.clicking_mode = False
        self.scroll_mode = False
        self.last_gesture = None
        self.gesture_start_time = 0
        
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

    def calculate_distance(self, p1, p2):
        """Calculate distance between two points"""
        return hypot(p1.x - p2.x, p1.y - p2.y)

    def get_gesture(self, hand_landmarks):
        """Determine the current hand gesture"""
        # Get relevant finger landmarks
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # Get MCP (base) positions for comparison
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

        # Check if fingers are extended
        index_up = index_tip.y < index_mcp.y
        middle_up = middle_tip.y < middle_mcp.y
        ring_up = ring_tip.y < ring_mcp.y
        pinky_up = pinky_tip.y < pinky_mcp.y
        
        # Calculate thumb-index distance for click detection
        thumb_index_distance = self.calculate_distance(thumb_tip, index_tip)
        
        # Determine gesture
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "MOVE"
        elif index_up and middle_up and not ring_up and not pinky_up:
            if thumb_index_distance < 0.1:  # Threshold for click
                return "CLICK"
            return "SCROLL"
        elif all([index_up, middle_up, ring_up, pinky_up]):
            return "PALM"
        return "NONE"

    def smooth_movement(self, x, y):
        """Apply smoothing to cursor movement"""
        smoothed_x = int(self.smoothing * self.prev_x + (1 - self.smoothing) * x)
        smoothed_y = int(self.smoothing * self.prev_y + (1 - self.smoothing) * y)
        self.prev_x, self.prev_y = smoothed_x, smoothed_y
        return smoothed_x, smoothed_y

    def run(self):
        while True:
            success, image = self.cap.read()
            if not success:
                continue

            # Flip image horizontally for later mirror effect
            image = cv2.flip(image, 1)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process hand landmarks
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get gesture
                    gesture = self.get_gesture(hand_landmarks)
                    
                    # Get index finger position
                    index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # Convert coordinates
                    camera_x = int(index_tip.x * 640)
                    camera_y = int(index_tip.y * 480)
                    
                    # Map to screen coordinates
                    screen_x = np.interp(camera_x, [0, 640], [0, self.screen_width])
                    screen_y = np.interp(camera_y, [0, 480], [0, self.screen_height])
                    
                    # Apply smoothing
                    smooth_x, smooth_y = self.smooth_movement(screen_x, screen_y)
                    
                    # Handle different gestures
                    if gesture == "MOVE":
                        pyautogui.moveTo(smooth_x, smooth_y)
                        self.clicking_mode = False
                        self.scroll_mode = False
                    
                    elif gesture == "CLICK":
                        if not self.clicking_mode:
                            pyautogui.click()
                            self.clicking_mode = True
                    
                    elif gesture == "SCROLL":
                        if not self.scroll_mode:
                            initial_y = smooth_y
                            self.scroll_mode = True
                        else:
                            scroll_amount = (smooth_y - initial_y) / 50
                            pyautogui.scroll(-int(scroll_amount))
                    
                    # Display gesture on screen
                    cv2.putText(
                        image,
                        f"Gesture: {gesture}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

            # Display the image
            cv2.imshow("Hand Gesture Mouse", image)
            
            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create and run the hand gesture mouse controller
    controller = HandGestureMouse()
    controller.run()
