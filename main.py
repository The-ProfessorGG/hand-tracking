import cv2
import mediapipe as mp
import math
import pyautogui

# Open the webcam
cap = cv2.VideoCapture(0)

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Drag state
dragging = False

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read from webcam.")
        break

    # Flip the image so it feels natural
    frame = cv2.flip(frame, 1)

    # Get frame size
    h, w, _ = frame.shape

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Get thumb and index landmarks
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Convert to pixel positions
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Draw circles and line
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255), -1)
            cv2.circle(frame, (index_x, index_y), 10, (255, 0, 255), -1)
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)

            # Distance for pinch detection
            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

            # Show distance
            cv2.putText(
                frame,
                f"Distance: {int(distance)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            # Move mouse using index finger position
            mouse_x = int(index_tip.x * screen_w)
            mouse_y = int(index_tip.y * screen_h)
            pyautogui.moveTo(mouse_x, mouse_y)

            # Pinch = drag
            if distance < 40:
                cv2.putText(
                    frame,
                    "PINCH",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

    cv2.imshow("Hand Tracking Mouse Control", frame)

    # ESC to close
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Safety release
if dragging:
    pyautogui.mouseUp()

cap.release()
cv2.destroyAllWindows()