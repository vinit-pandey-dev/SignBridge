import streamlit as st
import numpy as np
import os
import queue
import threading
import time
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #071a12, #0b2e20, #103b2b);
    color: white;
}

h1 {
    color: #4ade80 !important;
    font-size: 4rem !important;
    font-weight: 800 !important;
    text-shadow: 0px 0px 15px rgba(74,222,128,0.5);
}

h2,h3 {
   color:#4ade80 !important;
}

[data-testid="stSidebar"] {
    background: #04140d;
}

.stButton > button {
    background: linear-gradient(90deg,#16a34a,#22c55e);
    color: white;
    border-radius: 14px;
    border: none;
    font-weight: bold;
    font-size: 18px;
    height: 55px;
}

.stButton > button:hover {
    background: linear-gradient(90deg,#22c55e,#4ade80);
}

.stTextInput input {
    border-radius: 15px;
    border: 2px solid #22c55e;
    font-size: 18px;
}

.card {
    background: rgba(15, 23, 42, 0.4);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid #22c55e;
}
label, p, span, div {
    color: #eafff4 !important;
}

.stRadio label {
    color: #eafff4 !important;
    font-weight: 600 !important;
}

[data-testid="stMarkdownContainer"] {
    color: #eafff4 !important;
}

section[data-testid="stSidebar"] * {
    color: #d1fae5 !important;
}

.stExpander {
    border: 1px solid #22c55e !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
ASSETS_PATH = "Assets"
ALPHABET_PATH = os.path.join(ASSETS_PATH, "Alphabet")
VIDEO_FPS_DELAY = 5

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
    actions = np.array([
    'Yes',
    'Please',
    'Thanks'
])
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
# --- AUDIO QUEUE ---
import pyttsx3

if "speech_q" not in st.session_state:
    st.session_state.speech_q = queue.Queue()

def speech_worker(q):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)

    while True:
        text = q.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        q.task_done()

if "speech_worker_started" not in st.session_state:
    t = threading.Thread(
        target=speech_worker,
        args=(st.session_state.speech_q,),
        daemon=True
    )
    t.start()
    st.session_state.speech_worker_started = True

def speak(text):
    st.session_state.speech_q.put(text)

# ============================================================
# SMART SPEECH-TO-SIGN HELPERS
# ============================================================
def normalize_name(text):
    return ''.join(c.lower() for c in text if c.isalnum())

def find_word_asset(word):
    """Flexible search: matches word video even if case/space differs."""
    clean_word = normalize_name(word)

    extensions = ['.mp4', '.avi', '.mov']

    for filename in os.listdir(ASSETS_PATH):
        file_path = os.path.join(ASSETS_PATH, filename)

        if not os.path.isfile(file_path):
            continue

        name, ext = os.path.splitext(filename)

        if ext.lower() not in extensions:
            continue

        if normalize_name(name) == clean_word:
            return file_path

    return 
   

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
    whole_path = find_word_asset(sentence_text.strip())

    if whole_path:
       return [{
        'path': whole_path,
        'label': sentence_text.strip(),
        'type': 'word'
    }]
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
           video_placeholder.video(path)

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
st.sidebar.image("images.png", width=120)
st.sidebar.success("🌍 Real-Time Sign ↔ Speech Translator")
st.sidebar.success("✅ MediaPipe + BiLSTM")
st.sidebar.info("Real-Time Sign ↔ Speech Translator")
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

    st.image("images.png", width=650)

    st.markdown("""
    <h1 style='text-align:center;color:#4ade80;'>🌉 SignBridge AI</h1>
    <h3 style='text-align:center;color:white;'>Breaking Communication Barriers Through AI</h3>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h1 style='text-align:center;color:#d1fae5;font-size:60px;'>
    🖐️ Sign Language → Speech
    </h1>
    """, unsafe_allow_html=True)

    st.caption("Perform 'Yes', 'Please', or 'Thanks' — BiLSTM detects in real-time.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<h3 style='font-size:26px;'>🎥 Live Camera</h3>", unsafe_allow_html=True)
        st_frame = st.empty()

    with col2:
        st.markdown("<h3 style='color:#4ade80;'>📝 Detected</h3>", unsafe_allow_html=True)
        word_placeholder = st.empty()

        st.markdown("<h3 style='color:#4ade80;'>📊 Confidence</h3>", unsafe_allow_html=True)
        confidence_placeholder = st.empty()

        st.markdown("<h3 style='color:#4ade80;'>⏱ Stability</h3>", unsafe_allow_html=True)
        stability_placeholder = st.empty()

    c1, c2 = st.columns(2)

    start_button = c1.button("🚀 Start Detection", use_container_width=True)
    stop_button = c2.button("🛑 Stop Camera", use_container_width=True)

    if start_button:
        cv2, mp, pyttsx3, sr, Sequential, LSTM, Dense, Bidirectional, BatchNormalization = get_libraries()
        model, actions = load_model()

        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils

        sequence = []
        sentence = []
        threshold = 0.85
        CONSECUTIVE_FRAMES = 5
        consecutive_count = 0
        last_prediction = None
        COOLDOWN_FRAMES = 20
        cooldown_counter = 0

        cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        st.write("Camera Open:", cap.isOpened())

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera frame not received")
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
                        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        predicted_idx = np.argmax(res)
                        confidence = float(res[predicted_idx])
                        current_word = actions[predicted_idx]

                        if current_word == last_prediction:
                            consecutive_count += 1
                        else:
                            consecutive_count = 1
                            last_prediction = current_word

                        if consecutive_count >= CONSECUTIVE_FRAMES and confidence > threshold and cooldown_counter == 0:
                            if len(sentence) == 0 or current_word != sentence[-1]:
                                sentence.append(current_word)
                                st.write("Speaking:", current_word)
                                speak(current_word)

                                cooldown_counter = COOLDOWN_FRAMES
                                consecutive_count = 0

                        confidence_placeholder.progress(confidence, text=f"🎯 {current_word}: {confidence*100:.1f}%")
                        stab = min(consecutive_count / CONSECUTIVE_FRAMES, 1.0)
                        stability_placeholder.progress(stab, text=f"⚡ Stability: {consecutive_count}/{CONSECUTIVE_FRAMES}")

                    word_placeholder.markdown(
                        f"<h2 style='text-align:center;color:#4ade80;'>{' '.join(sentence[-5:])}</h2>",
                        unsafe_allow_html=True
                    )
                else:
                    sequence = []
                    consecutive_count = 0
                    last_prediction = None
                    word_placeholder.info("Show your hands...")

                st_frame.image(image, channels="BGR")

                if stop_button:
                    break

        cap.release()# ============================================================
# MODE 2: SPEECH TO SIGN (SMART)
# ============================================================
if mode == "🗣️ Speech to Sign":

    st.title("🗣️ Speech → Sign Language")
    st.caption("Speak or type a sentence — matched videos will play.")

    input_mode = st.selectbox(
        "Input method:",
        ["🎤 Microphone", "⌨️ Type manually"],
        key="speech_input_method"
    )

    manual_text = ""

    if input_mode == "⌨️ Type manually":
        manual_text = st.text_input(
            "Type your sentence:",
            placeholder="e.g. hello thanks"
        )

    if st.button("▶ Start"):
        text = None

        if input_mode == "🎤 Microphone":
            cv2, mp, pyttsx3, sr, Sequential, LSTM, Dense, Bidirectional, BatchNormalization = get_libraries()
            r = sr.Recognizer()

            try:
                with sr.Microphone() as source:
                    st.info("🎙️ Speak now...")
                    r.adjust_for_ambient_noise(source, duration=2)
                    audio = r.listen(source, timeout=10, phrase_time_limit=10)

                text = r.recognize_google(audio, language="en-US").lower()
                st.success(f"✅ You said: {text}")

            except Exception as e:
                st.error(f"🎤 Error: {e}")

        else:
            text = manual_text.lower().strip()

        if text:
            playlist = build_playlist(text)
            vid_col, stat_col = st.columns([3, 1])
            play_smart_playlist(playlist, stat_col, vid_col, letter_delay, word_delay)
        else:
            st.warning("Please provide input first.")