import cv2
import numpy as np
import tensorflow as tf
import time
import winsound
import mediapipe as mp
from collections import deque

eye_model = tf.keras.models.load_model("Models/eye_model.h5")
yawn_model = tf.keras.models.load_model("Models/yawn_model.h5")


mp_face_mesh = mp.solutions.face_mesh    
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


IMG_SIZE = 64

EYE_THRESHOLD = 0.75
YAWN_THRESHOLD = 0.30

EYE_ALARM_TIME = 2.0
MIN_YAWN_TIME = 0.8
YAWN_LIMIT = 3

PROCESS_EVERY_N_FRAMES = 2

eye_scores = deque(maxlen=5)
yawn_scores = deque(maxlen=5)

eye_close_start = None
yawn_counter = 0
yawn_active = False
yawn_start_time = None
frame_count = 0


LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380]
MOUTH_IDX = [61, 291, 13, 14, 78, 308, 81, 311, 191, 415]


def preprocess(img):
    if img is None or img.size == 0:
        return None

    try:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except:
        return None


def get_landmark_points(face_landmarks, img_w, img_h, indices):
    pts = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * img_w)
        y = int(lm.y * img_h)
        pts.append((x, y))
    return pts


def crop_from_points(frame, points, pad=8):
    if not points:
        return None, None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(frame.shape[1], max(xs) + pad)
    y2 = min(frame.shape[0], max(ys) + pad)

    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

last_result = None

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # Run FaceMesh every few frames for speed
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_result = face_mesh.process(rgb)

    result = last_result

    if result and result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        left_eye_pts = get_landmark_points(face_landmarks, w, h, LEFT_EYE_IDX)
        right_eye_pts = get_landmark_points(face_landmarks, w, h, RIGHT_EYE_IDX)
        mouth_pts = get_landmark_points(face_landmarks, w, h, MOUTH_IDX)

        left_eye, left_box = crop_from_points(frame, left_eye_pts, pad=10)
        right_eye, right_box = crop_from_points(frame, right_eye_pts, pad=10)
        mouth, mouth_box = crop_from_points(frame, mouth_pts, pad=15)

        # Debug rectangles
        if left_box:
            x1, y1, x2, y2 = left_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        if right_box:
            x1, y1, x2, y2 = right_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        if mouth_box:
            x1, y1, x2, y2 = mouth_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)

        eye_score = 0.0
        l_in = preprocess(left_eye)
        r_in = preprocess(right_eye)

        if l_in is not None and r_in is not None:
            l_score = eye_model.predict(l_in, verbose=0)[0][0]
            r_score = eye_model.predict(r_in, verbose=0)[0][0]
            eye_score = (l_score + r_score) / 2.0
            eye_scores.append(eye_score)

        smoothed_eye = sum(eye_scores) / len(eye_scores) if len(eye_scores) > 0 else eye_score
        is_closed = smoothed_eye > EYE_THRESHOLD

        if is_closed:
            if eye_close_start is None:
                eye_close_start = time.time()
            elif time.time() - eye_close_start >= EYE_ALARM_TIME:
                winsound.Beep(1000, 500)
        else:
            eye_close_start = None

        yawn_score = 0.0
        m_in = preprocess(mouth)

        if m_in is not None:
            yawn_score = yawn_model.predict(m_in, verbose=0)[0][0]
            yawn_scores.append(yawn_score)

        smoothed_yawn = sum(yawn_scores) / len(yawn_scores) if len(yawn_scores) > 0 else yawn_score
        is_yawning = smoothed_yawn > YAWN_THRESHOLD

        if is_yawning:
            if not yawn_active:
                yawn_active = True
                yawn_start_time = time.time()
            elif time.time() - yawn_start_time >= MIN_YAWN_TIME:
                yawn_counter += 1
                yawn_active = False
                yawn_start_time = None
        else:
            yawn_active = False
            yawn_start_time = None

        if yawn_counter >= YAWN_LIMIT:
            winsound.Beep(1500, 700)
            yawn_counter = 0

        eye_text = "CLOSED" if is_closed else "OPEN"
        mouth_text = "YAWN" if is_yawning else "NO YAWN"

        eye_color = (0, 0, 255) if is_closed else (0, 255, 0)
        mouth_color = (0, 165, 255) if is_yawning else (255, 255, 0)

        cv2.putText(
            frame,
            f"Eye: {eye_text} ({smoothed_eye:.2f})",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            eye_color,
            2
        )

        cv2.putText(
            frame,
            f"Mouth: {mouth_text} ({smoothed_yawn:.2f})",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            mouth_color,
            2
        )

        cv2.putText(
            frame,
            f"Yawn Count: {yawn_counter}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()