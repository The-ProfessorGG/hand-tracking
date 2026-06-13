import cv2
import mediapipe as mp
import math
import pyautogui

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

screen_w, screen_h = pyautogui.size()

MARGIN = 0.2

def cam_to_screen(norm_x, norm_y):
    active_w = 1.0 - 2 * MARGIN
    active_h = 1.0 - 2 * MARGIN
    sx = (norm_x - MARGIN) / active_w
    sy = (norm_y - MARGIN) / active_h
    sx = max(0.0, min(1.0, sx))
    sy = max(0.0, min(1.0, sy))
    return int(sx * screen_w), int(sy * screen_h)

box_visible = False
box_x, box_y, box_w, box_h = 200, 150, 200, 150

resizing = False
resize_start_dist = None
resize_start_w = None
resize_start_h = None
resize_start_cx = None
resize_start_cy = None

prev_both_fists = False
dragging = False


def is_fist(landmarks):
    tips  = [8, 12, 16, 20]
    bases = [5,  9, 13, 17]
    for tip_id, base_id in zip(tips, bases):
        tip  = landmarks.landmark[tip_id]
        base = landmarks.landmark[base_id]
        if tip.y < base.y - 0.02:
            return False
    return True


def get_pinch_point(landmarks, frame_w, frame_h):
    thumb = landmarks.landmark[4]
    index = landmarks.landmark[8]
    x = int((thumb.x + index.x) / 2 * frame_w)
    y = int((thumb.y + index.y) / 2 * frame_h)
    return x, y


def pinch_distance(landmarks):
    thumb = landmarks.landmark[4]
    index = landmarks.landmark[8]
    return math.hypot(thumb.x - index.x, thumb.y - index.y)


def draw_active_zone(frame, frame_w, frame_h):
    x1 = int(MARGIN * frame_w)
    y1 = int(MARGIN * frame_h)
    x2 = int((1 - MARGIN) * frame_w)
    y2 = int((1 - MARGIN) * frame_h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
    cv2.putText(frame, "Active zone", (x1 + 4, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    fist_count = 0
    hand_list  = []

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            hand_list.append(lm)
            if is_fist(lm):
                fist_count += 1

    both_fists_now = (fist_count == 2)
    if both_fists_now and not prev_both_fists:
        box_visible = not box_visible
        resizing = False
    prev_both_fists = both_fists_now

    # ── Mouse control ─────────────────────────────────────────────────
    for lm in hand_list:
        if not is_fist(lm):
            index_tip = lm.landmark[8]
            thumb_tip = lm.landmark[4]

            # Midpoint between thumb and index fingertip
            mid_x = (index_tip.x + thumb_tip.x) / 2
            mid_y = (index_tip.y + thumb_tip.y) / 2

            mouse_x, mouse_y = cam_to_screen(mid_x, mid_y)
            pyautogui.moveTo(mouse_x, mouse_y)

            # Pixel positions for drawing
            ix = int(index_tip.x * w)
            iy = int(index_tip.y * h)
            tx = int(thumb_tip.x * w)
            ty = int(thumb_tip.y * h)
            mx = int(mid_x * w)
            my = int(mid_y * h)

            cv2.circle(frame, (ix, iy), 10, (255, 0, 255), -1)   # index — pink
            cv2.circle(frame, (tx, ty), 10, (0, 255, 255), -1)   # thumb — yellow
            cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 0), 2)
            cv2.circle(frame, (mx, my), 8, (255, 255, 255), -1)  # midpoint — white

            dist = math.hypot(ix - tx, iy - ty)
            cv2.putText(frame, f"Distance: {int(dist)}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if dist < 40:
                cv2.putText(frame, "PINCH", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if not dragging:
                    pyautogui.click()
                    dragging = True
            else:
                dragging = False
            break

    if box_visible:
        cv2.rectangle(frame,
                      (box_x, box_y),
                      (box_x + box_w, box_y + box_h),
                      (0, 255, 0), 2)
        cv2.putText(frame, "BOX", (box_x + 5, box_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if box_visible and len(hand_list) == 2:
        pd0 = pinch_distance(hand_list[0])
        pd1 = pinch_distance(hand_list[1])
        PINCH_THRESH = 0.05

        if pd0 < PINCH_THRESH and pd1 < PINCH_THRESH:
            p0 = get_pinch_point(hand_list[0], w, h)
            p1 = get_pinch_point(hand_list[1], w, h)
            left_pt  = p0 if p0[0] < p1[0] else p1
            right_pt = p0 if p0[0] > p1[0] else p1

            EDGE_MARGIN = 60
            near_left  = abs(left_pt[0]  - box_x)           < EDGE_MARGIN
            near_right = abs(right_pt[0] - (box_x + box_w)) < EDGE_MARGIN

            if near_left and near_right:
                current_dist = math.hypot(right_pt[0] - left_pt[0],
                                          right_pt[1] - left_pt[1])
                if not resizing:
                    resizing = True
                    resize_start_dist = current_dist
                    resize_start_w    = box_w
                    resize_start_h    = box_h
                    resize_start_cx   = box_x + box_w // 2
                    resize_start_cy   = box_y + box_h // 2
                else:
                    if resize_start_dist > 0:
                        scale = current_dist / resize_start_dist
                        new_w = max(50, int(resize_start_w * scale))
                        new_h = max(50, int(resize_start_h * scale))
                        box_x = resize_start_cx - new_w // 2
                        box_y = resize_start_cy - new_h // 2
                        box_w = new_w
                        box_h = new_h
            else:
                resizing = False
        else:
            resizing = False

    draw_active_zone(frame, w, h)

    cv2.putText(frame, f"Fists: {fist_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if resizing:
        cv2.putText(frame, "RESIZING", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f"Box: {'ON' if box_visible else 'OFF'}", (w - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()