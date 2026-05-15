import streamlit as st
import numpy as np
import os
import queue
import threading
import time

st.set_page_config(layout="wide", page_title="SignBridge AI", page_icon="🌉")

# --- CONFIGURATION ---
ASSETS_PATH = "Assets"
ALPHABET_PATH = os.path.join(ASSETS_PATH, "Alphabet")
VIDEO_FPS_DELAY = 30

# --- LAZY LOAD ---
def get_libraries():
    import cv2
    import mediapipe as mp
    import pyttsx3
    import speech_recognition as sr
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Bidirectional, BatchNormalization
    return cv2, mp, pyttsx3, sr, Sequential, LSTM, Dense, Bidirectional, BatchNormalization

# --- LOAD BIDIRECTIONAL LSTM MODEL ---
@st.cache_resource
def load_model():
    cv2, mp, pyttsx3, sr, Sequential, LSTM, Dense, Bidirectional, BatchNormalization = get_libraries()
    actions = np.array(['Hello', 'Thanks', 'ILoveYou'])
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=(30, 1662)),
        BatchNormalization(),
        Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=False, activation='relu')),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(actions.shape[0], activation='softmax')
    ])
    if os.path.exists('action.h5'):
        model.load_weights('action.h5')
    else:
        st.error("Model file 'action.h5' not found. Please run train_model.py first.")
    return model, actions

# --- MEDIAPIPE HELPERS ---
def mediapipe_detection(image, model):
    import cv2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results, mp_drawing, mp_holistic):
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

# --- AUDIO QUEUE ---
speech_q = queue.Queue()
def speech_worker():
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    while True:
        text = speech_q.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_q.task_done()

if 'speech_worker_started' not in st.session_state:
    t = threading.Thread(target=speech_worker, daemon=True)
    t.start()
    st.session_state.speech_worker_started = True

def speak(text):
    speech_q.put(text)

# ============================================================
# SMART SPEECH-TO-SIGN HELPERS
# ============================================================
def find_word_asset(word):
    """Check if a video exists for the whole word."""
    variants = [word, word.capitalize(), word.upper(), word.lower()]
    extensions = ['.mp4', '.avi', '.mov']
    for variant in variants:
        for ext in extensions:
            path = os.path.join(ASSETS_PATH, f"{variant}{ext}")
            if os.path.exists(path):
                return path
    return None

def find_letter_asset(letter):
    """Check if a video/image exists for a letter."""
    letter = letter.upper()
    extensions = ['.mp4', '.avi', '.png', '.jpg']
    for ext in extensions:
        path = os.path.join(ALPHABET_PATH, f"{letter}{ext}")
        if os.path.exists(path):
            return path
    return None

def build_playlist(sentence_text):
    """
    Smart playlist builder:
      - Word video found? → add word entry
      - Not found? → add each letter as separate entry
    """
    playlist = []
    words = sentence_text.strip().split()
    for word in words:
        clean = ''.join(c for c in word if c.isalpha())
        if not clean:
            continue
        word_path = find_word_asset(clean)
        if word_path:
            playlist.append({'path': word_path, 'label': clean.capitalize(), 'type': 'word'})
        else:
            for letter in clean:
                letter_path = find_letter_asset(letter)
                playlist.append({
                    'path': letter_path,
                    'label': letter.upper(),
                    'type': 'letter' if letter_path else 'missing'
                })
    return playlist

def play_smart_playlist(playlist, stat_col, vid_col, letter_delay, word_delay):
    """Play through the smart playlist with proper delays in Streamlit."""
    import cv2
    import numpy as np

    video_placeholder = vid_col.empty()
    status_placeholder = stat_col.empty()
    progress_bar = stat_col.progress(0)

    total = len(playlist)

    for i, item in enumerate(playlist):
        label = item['label']
        item_type = item['type']
        path = item['path']

        # Update progress
        progress_bar.progress((i + 1) / total, text=f"{i+1}/{total}")

        if item_type == 'word':
            status_placeholder.success(f"▶️ **WORD:** {label}")
        elif item_type == 'letter':
            status_placeholder.info(f"🔤 **LETTER:** {label}")
        else:
            status_placeholder.warning(f"⚠️ No asset for: {label}")
            blank = np.ones((300, 400, 3), dtype='uint8') * 60
            cv2.putText(blank, label, (150, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 100, 255), 8, cv2.LINE_AA)
            video_placeholder.image(blank, channels="BGR", caption=f"Missing: {label}")
            time.sleep(letter_delay)
            continue

        ext = os.path.splitext(path)[1].lower()

        if ext in ['.mp4', '.avi', '.mov']:
            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (45, 45, 45), -1)
                color = (0, 220, 100) if item_type == 'word' else (0, 165, 255)
                tag = "WORD" if item_type == 'word' else "LETTER"
                cv2.putText(frame, f"[{tag}]  {label}", (10, 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                video_placeholder.image(frame, channels="BGR")
                time.sleep(VIDEO_FPS_DELAY / 1000)
            cap.release()

        elif ext in ['.png', '.jpg', '.jpeg']:
            frame = cv2.imread(path)
            if frame is not None:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (45, 45, 45), -1)
                cv2.putText(frame, f"[LETTER]  {label}", (10, 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                video_placeholder.image(frame, channels="BGR", caption=f"Letter: {label}")
                time.sleep(1.0)

        # Delay after each item
        time.sleep(letter_delay if item_type == 'letter' else word_delay)

    status_placeholder.success("✅ Playback Complete!")
    progress_bar.progress(1.0, text="Done!")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("SignBridge 🌉")
st.sidebar.info("Bidirectional LSTM — Real-time Sign Language Translator")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** BiLSTM ✅")
st.sidebar.markdown("**Smart Spell:** Active ✅")
st.sidebar.markdown("**Audio Queue:** Active ✅")
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
letter_delay = st.sidebar.slider("Letter Delay (sec)", 0.2, 1.5, 0.5, 0.1)
word_delay   = st.sidebar.slider("Word Delay (sec)",   0.3, 2.0, 0.8, 0.1)
mode = st.sidebar.radio("Select Mode:", ["🖐️ Sign to Speech", "🗣️ Speech to Sign"])

# ============================================================
# MODE 1: SIGN TO SPEECH
# ============================================================
if mode == "🖐️ Sign to Speech":
    st.title("🖐️ Sign Language → Speech")
    st.caption("Perform 'Hello', 'Thanks', or 'I Love You' — BiLSTM detects in real-time.")

    col1, col2 = st.columns([3, 1])
    with col1:
        st_frame = st.image([])
    with col2:
        st.markdown("### 📝 Detected")
        word_placeholder = st.empty()
        st.markdown("### 📊 Confidence")
        confidence_placeholder = st.empty()
        st.markdown("### ⏱ Stability")
        stability_placeholder = st.empty()

    c1, c2 = st.columns(2)
    start_button = c1.button("▶ Start Camera", use_container_width=True)
    stop_button  = c2.button("⏹ Stop Camera",  use_container_width=True)

    if start_button:
        cv2, mp, pyttsx3, sr, Sequential, LSTM, Dense, Bidirectional, BatchNormalization = get_libraries()
        model, actions = load_model()
        import mediapipe as mp

        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        sequence = []
        sentence = []
        threshold = 0.85
        CONSECUTIVE_FRAMES = 5
        consecutive_count  = 0
        last_prediction    = None
        COOLDOWN_FRAMES    = 20
        cooldown_counter   = 0

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results, mp_drawing, mp_holistic)

                if cooldown_counter > 0:
                    cooldown_counter -= 1

                if results.left_hand_landmarks or results.right_hand_landmarks:
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        res          = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        predicted_idx = np.argmax(res)
                        confidence   = float(res[predicted_idx])
                        current_word = actions[predicted_idx]

                        if current_word == last_prediction:
                            consecutive_count += 1
                        else:
                            consecutive_count = 1
                            last_prediction   = current_word

                        if (consecutive_count >= CONSECUTIVE_FRAMES and
                                confidence > threshold and cooldown_counter == 0):
                            if len(sentence) == 0 or current_word != sentence[-1]:
                                sentence.append(current_word)
                                speak(current_word)
                                cooldown_counter  = COOLDOWN_FRAMES
                                consecutive_count = 0

                        confidence_placeholder.progress(confidence, text=f"{current_word}: {confidence*100:.1f}%")
                        stab = min(consecutive_count / CONSECUTIVE_FRAMES, 1.0)
                        stability_placeholder.progress(stab, text=f"Stable: {consecutive_count}/{CONSECUTIVE_FRAMES}")

                    word_placeholder.markdown(f"## **{' '.join(sentence[-5:])}**")
                else:
                    sequence          = []
                    consecutive_count = 0
                    last_prediction   = None
                    word_placeholder.info("Show your hands...")

                st_frame.image(image, channels="BGR")
                if stop_button:
                    break
        cap.release()

# ============================================================
# MODE 2: SPEECH TO SIGN (SMART)
# ============================================================
elif mode == "🗣️ Speech to Sign":
    st.title("🗣️ Speech → Sign Language")
    st.caption("Speak a sentence — words matched to videos, unknown words are finger-spelled automatically.")

    with st.expander("ℹ️ How Smart Search Works"):
        st.markdown("""
        1. 🎙️ **Voice → Text** using Google Speech Recognition
        2. ✂️ **Split** sentence into individual words
        3. 🔍 **Smart Lookup** for each word:
           - ✅ **Found in `Assets/`?** → Play the word video directly
           - 🔤 **Not found?** → Finger spell using `Assets/Alphabet/` (A–Z)
        4. ⏱️ **Visual delay** between each letter (adjustable in sidebar)
        """)

    input_mode = st.radio("Input method:", ["🎤 Microphone", "⌨️ Type manually"], horizontal=True)
    manual_text = ""
    if input_mode == "⌨️ Type manually":
        manual_text = st.text_input("Type your sentence:", placeholder="e.g. hello my name is john")

    if st.button("▶ Start", use_container_width=False):
        text = None

        if input_mode == "🎤 Microphone":
            cv2, mp, pyttsx3, sr, Sequential, LSTM, Dense, Bidirectional, BatchNormalization = get_libraries()
            r = sr.Recognizer()
            with sr.Microphone() as source:
                with st.spinner("🎙️ Listening... Speak now!"):
                    r.adjust_for_ambient_noise(source)
                    try:
                        audio = r.listen(source, timeout=6, phrase_time_limit=8)
                        text  = r.recognize_google(audio).lower()
                    except Exception as e:
                        st.error(f"Could not understand: {e}")
        else:
            text = manual_text.lower().strip()

        if text:
            st.success(f"📝 Processing: **\"{text}\"**")
            playlist = build_playlist(text)

            # Playlist preview
            st.markdown("### 📋 Playlist Preview")
            cols = st.columns(min(len(playlist), 10))
            for i, item in enumerate(playlist[:10]):
                with cols[i % len(cols)]:
                    if item['type'] == 'word':
                        st.success(f"📹 {item['label']}")
                    elif item['type'] == 'letter':
                        st.info(f"🔤 {item['label']}")
                    else:
                        st.error(f"❌ {item['label']}")
            if len(playlist) > 10:
                st.caption(f"...and {len(playlist) - 10} more items")

            st.markdown("---")
            st.markdown("### 🎬 Playback")
            vid_col, stat_col = st.columns([3, 1])
            stat_col.markdown("**Status**")
            play_smart_playlist(playlist, stat_col, vid_col, letter_delay, word_delay)

        elif input_mode == "⌨️ Type manually" and not manual_text:
            st.warning("Please type a sentence first.")