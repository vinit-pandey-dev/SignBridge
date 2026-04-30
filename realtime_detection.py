import cv2
import numpy as np
import os
import queue
import threading
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, BatchNormalization

# ============================================================
# SIGNBRIDGE - REALTIME DETECTION (UPGRADED)
# Fixes Applied:
#   1. Bidirectional LSTM Model
#   2. Prediction Stability (debounce counter)
#   3. Audio Queue (no overlapping speech)
#   4. Confidence display on screen
# ============================================================

# --- 1. SETUP ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
actions = np.array(['Hello', 'Thanks', 'ILoveYou'])

# ---- HELPER FUNCTIONS ----
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# ============================================================
# FIX 1: AUDIO QUEUE - Prevents overlapping speech
# A background worker thread processes one word at a time.
# New words are added to queue and spoken sequentially.
# ============================================================
speech_queue = queue.Queue()

def speech_worker():
    """Background thread: speaks words from queue one by one"""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Slightly slower for clarity
    while True:
        text = speech_queue.get()
        if text is None:  # Sentinel to stop worker
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start the background speech worker
worker_thread = threading.Thread(target=speech_worker, daemon=True)
worker_thread.start()

def speak(text):
    """Add word to speech queue (non-blocking, no overlap)"""
    speech_queue.put(text)

# --- 2. LOAD BIDIRECTIONAL LSTM MODEL ---
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=(30, 1662)),
    BatchNormalization(),
    # Dropout not needed at inference, but layers must match training
    Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
    BatchNormalization(),
    Bidirectional(LSTM(64, return_sequences=False, activation='relu')),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])
model.load_weights('action.h5')
print("Model loaded successfully!")

# --- 3. DETECTION VARIABLES ---
sequence = []
sentence = []
threshold = 0.85           # Confidence threshold (higher = more strict)

# ============================================================
# FIX 2: PREDICTION STABILITY - Debounce Counter
# A word is only added if it's predicted consistently
# for CONSECUTIVE_FRAMES in a row (not just once).
# ============================================================
CONSECUTIVE_FRAMES = 5     # Word must be stable for 5 consecutive predictions
consecutive_count = 0
last_prediction = None

# ============================================================
# FIX 3: COOLDOWN TIMER
# After a word is added, ignore the same word for
# COOLDOWN_FRAMES to prevent repeated detection.
# ============================================================
COOLDOWN_FRAMES = 20
cooldown_counter = 0

# --- 4. REAL-TIME LOOP ---
cap = cv2.VideoCapture(0)
print("Starting real-time detection... Press 'q' to quit, 'c' to clear sentence.")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        # Decrement cooldown
        if cooldown_counter > 0:
            cooldown_counter -= 1

        # --- PREDICTION LOGIC ---
        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predicted_idx = np.argmax(res)
                confidence = res[predicted_idx]
                current_word = actions[predicted_idx]

                # --- DEBOUNCE: Count consecutive same predictions ---
                if current_word == last_prediction:
                    consecutive_count += 1
                else:
                    consecutive_count = 1
                    last_prediction = current_word

                # --- ADD WORD only if: stable + confident + not in cooldown ---
                if (consecutive_count >= CONSECUTIVE_FRAMES and
                    confidence > threshold and
                    cooldown_counter == 0):

                    if len(sentence) == 0 or current_word != sentence[-1]:
                        sentence.append(current_word)
                        speak(current_word)
                        cooldown_counter = COOLDOWN_FRAMES  # Start cooldown
                        consecutive_count = 0

                # --- Visual: Confidence bar ---
                bar_width = int(confidence * 300)
                bar_color = (0, 255, 0) if confidence > threshold else (0, 165, 255)
                cv2.rectangle(image, (0, 60), (bar_width, 80), bar_color, -1)
                cv2.putText(image, f'{current_word}: {confidence*100:.1f}%', (3, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # --- Visual: Stability indicator ---
                stability_pct = min(consecutive_count / CONSECUTIVE_FRAMES, 1.0)
                stab_width = int(stability_pct * 300)
                cv2.rectangle(image, (0, 85), (stab_width, 95), (255, 200, 0), -1)
                cv2.putText(image, f'Stability: {consecutive_count}/{CONSECUTIVE_FRAMES}', (3, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        else:
            # No hands detected - reset sequence and counters
            sequence = []
            consecutive_count = 0
            last_prediction = None
            cv2.putText(image, "Show your hands...", (100, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # --- Visual: Sentence display ---
        if len(sentence) > 5:
            sentence = sentence[-5:]

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Cooldown indicator
        if cooldown_counter > 0:
            cv2.putText(image, f'Cooldown: {cooldown_counter}', (500, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        cv2.imshow('SignBridge RealTime (BiLSTM)', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = []
            print("Sentence cleared.")

cap.release()
cv2.destroyAllWindows()
# Stop the speech worker
speech_queue.put(None)
print("Detection stopped.")