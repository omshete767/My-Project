import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Screen dimensions
screen_w, screen_h = pyautogui.size()
# Initialize webcam
cap = cv2.VideoCapture(0)

# Constants
PINCH_THRESHOLD = 0.05
FIST_THRESHOLD = 0.03
SCROLL_THRESHOLD = 0.03
MODE_SWITCH_DELAY = 20  # frames

# State variables
left_clicked = False
right_clicked = False
dragging = False
mode = "cursor"  # or "scroll"
mode_counter = 0

def get_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def is_fist(landmarks):
    """Check if hand is closed (fist)"""
    thumb_tip = landmarks[4]
    pinky_tip = landmarks[20]
    return get_distance(thumb_tip, pinky_tip) < FIST_THRESHOLD

def get_scroll_direction(landmarks, prev_y):
    """Calculate vertical scroll direction"""
    middle_tip = landmarks[12]
    current_y = middle_tip.y
    if prev_y is None:
        return 0, current_y
    
    scroll_amount = (prev_y - current_y) * 10  # Sensitivity multiplier
    return scroll_amount, current_y

print("Gesture Controls:")
print("- Index finger: Move cursor")
print("- Thumb+Index pinch: Left click")
print("- Thumb+Middle pinch: Right click")
print("- Closed fist: Drag mode")
print("- Two fingers up/down: Scroll")
print("- Palm open/close 3x: Switch modes")
print("- Press 'q' to quit")

prev_scroll_y = None
frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # Get key points
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            pinky_tip = landmarks[20]
            wrist = landmarks[0]
            
            # Hand center (palm)
            palm_x = int((thumb_tip.x + wrist.x)/2 * frame.shape[1])
            palm_y = int((thumb_tip.y + wrist.y)/2 * frame.shape[0])
            
            # Mode switching (open/close hand 3 times)
            if frame_count % 5 == 0:  # Check every 5 frames
                if is_fist(landmarks):
                    mode_counter += 1
                    if mode_counter == MODE_SWITCH_DELAY:
                        mode = "scroll" if mode == "cursor" else "cursor"
                        mode_counter = 0
                        print(f"Switched to {mode} mode")
            
            if mode == "cursor":
                # Cursor movement
                mouse_x = int(index_tip.x * screen_w)
                mouse_y = int(index_tip.y * screen_h)
                pyautogui.moveTo(mouse_x, mouse_y)
                
                # Left click (thumb-index pinch)
                if get_distance(thumb_tip, index_tip) < PINCH_THRESHOLD:
                    if not left_clicked:
                        pyautogui.click()
                        left_clicked = True
                        cv2.putText(frame, "LEFT CLICK", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    left_clicked = False
                
                # Right click (thumb-middle pinch)
                if get_distance(thumb_tip, middle_tip) < PINCH_THRESHOLD:
                    if not right_clicked:
                        pyautogui.rightClick()
                        right_clicked = True
                        cv2.putText(frame, "RIGHT CLICK", (50, 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    right_clicked = False
                
                # Drag (fist detection)
                if is_fist(landmarks) and not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                    cv2.putText(frame, "DRAGGING", (50, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                elif not is_fist(landmarks) and dragging:
                    pyautogui.mouseUp()
                    dragging = False
            
            elif mode == "scroll":
                # Scroll control
                scroll_amount, prev_scroll_y = get_scroll_direction(landmarks, prev_scroll_y)
                if abs(scroll_amount) > SCROLL_THRESHOLD:
                    pyautogui.scroll(int(scroll_amount * 100))
                    cv2.putText(frame, "SCROLLING", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Visual feedback
            cv2.circle(frame, (palm_x, palm_y), 10, (255, 255, 255), -1)
            cv2.putText(frame, f"Mode: {mode.upper()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display frame
    cv2.imshow("Advanced Gesture Control", frame)
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()