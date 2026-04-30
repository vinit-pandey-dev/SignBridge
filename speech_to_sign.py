import cv2
import os
import time
import speech_recognition as sr
import numpy as np

# ============================================================
# SIGNBRIDGE - SPEECH TO SIGN (FIXED - No Freezing)
# Fix: cv2.imshow only called AFTER user presses Enter
#      No window open during input() call
# ============================================================

ASSETS_PATH = "Assets"
ALPHABET_PATH = os.path.join(ASSETS_PATH, "Alphabet")
DELAY_BETWEEN_ALPHABETS = 0.5
DELAY_BETWEEN_WORDS = 0.8
VIDEO_FPS_DELAY = 30
WINDOW_NAME = "SignBridge - Speech to Sign"

# ============================================================
# HELPERS
# ============================================================
def find_word_asset(word):
    variants = [word, word.capitalize(), word.upper(), word.lower()]
    extensions = ['.mp4', '.avi', '.mov']
    for variant in variants:
        for ext in extensions:
            path = os.path.join(ASSETS_PATH, f"{variant}{ext}")
            if os.path.exists(path):
                return path
    return None

def find_letter_asset(letter):
    letter = letter.upper()
    extensions = ['.mp4', '.avi', '.png', '.jpg']
    for ext in extensions:
        path = os.path.join(ALPHABET_PATH, f"{letter}{ext}")
        if os.path.exists(path):
            return path
    return None

# ============================================================
# DISPLAY: Draw overlay bar on frame
# ============================================================
def draw_overlay(frame, top_text, bottom_text, mode="word"):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (45, 45, 45), -1)
    cv2.putText(frame, top_text, (15, 33),
               cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, h - 45), (w, h), (45, 45, 45), -1)
    color = (0, 220, 100) if mode == "word" else (0, 165, 255)
    tag = "WORD" if mode == "word" else "LETTER"
    cv2.putText(frame, f"[{tag}]  {bottom_text}", (15, h - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return frame

# ============================================================
# PLAY VIDEO
# ============================================================
def play_video(video_path, label, mode="word"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}")
        return

    print(f"  Playing: {label}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = draw_overlay(frame, f"Signing: {label}", label, mode)
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(VIDEO_FPS_DELAY) & 0xFF == ord('q'):
            cap.release()
            return
    cap.release()

# ============================================================
# SHOW IMAGE
# ============================================================
def show_image(image_path, label, duration=1.0):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"  [ERROR] Cannot load image: {image_path}")
        return
    frame = draw_overlay(frame, f"Letter: {label}", label, "letter")
    cv2.imshow(WINDOW_NAME, frame)
    start = time.time()
    while time.time() - start < duration:
        if cv2.waitKey(30) & 0xFF == ord('q'):
            return

# ============================================================
# PLAY LETTER (video or image)
# ============================================================
def play_letter(letter):
    path = find_letter_asset(letter)
    if path is None:
        print(f"  [SKIP] No asset for letter: {letter}")
        blank = np.ones((400, 500, 3), dtype='uint8') * 50
        cv2.putText(blank, letter.upper(), (170, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 100, 255), 12, cv2.LINE_AA)
        draw_overlay(blank, f"Letter: {letter.upper()} (No video)", letter.upper(), "letter")
        cv2.imshow(WINDOW_NAME, blank)
        cv2.waitKey(int(DELAY_BETWEEN_ALPHABETS * 1000))
        return

    ext = os.path.splitext(path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov']:
        play_video(path, letter.upper(), mode="letter")
    else:
        show_image(path, letter.upper(), duration=1.0)

    time.sleep(DELAY_BETWEEN_ALPHABETS)

# ============================================================
# PROCESS WORD: Smart lookup
# ============================================================
def process_word(word):
    clean = ''.join(c for c in word if c.isalpha())
    if not clean:
        return

    path = find_word_asset(clean)

    if path:
        print(f"\n  [WORD] Video found for '{clean}'")
        play_video(path, clean.capitalize(), mode="word")
        time.sleep(DELAY_BETWEEN_WORDS)
    else:
        print(f"\n  [SPELL] No video for '{clean}' → Finger spelling: {clean.upper()}")

        # Show "Spelling: WORD" intro
        intro = np.ones((400, 600, 3), dtype='uint8') * 40
        cv2.putText(intro, "Spelling:", (160, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(intro, clean.upper(), (int(300 - len(clean) * 28), 270),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 165, 255), 4, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, intro)
        cv2.waitKey(1200)

        for letter in clean:
            print(f"    -> {letter.upper()}")
            play_letter(letter)

        time.sleep(DELAY_BETWEEN_WORDS)

# ============================================================
# PROCESS FULL SENTENCE
# ============================================================
def process_sentence(text):
    print(f"\n{'='*45}")
    print(f"  Sentence: \"{text}\"")
    print(f"{'='*45}")

    words = text.strip().split()

    # Show sentence overview screen
    overview = np.ones((400, 700, 3), dtype='uint8') * 40
    cv2.putText(overview, "You said:", (30, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    display_text = f'"{text}"' if len(text) < 30 else f'"{text[:28]}..."'
    cv2.putText(overview, display_text, (30, 180),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 100), 2, cv2.LINE_AA)
    cv2.putText(overview, f"Words: {len(words)}", (30, 260),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 50), 2)
    cv2.putText(overview, "Starting...", (30, 330),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.imshow(WINDOW_NAME, overview)
    cv2.waitKey(1800)

    for i, word in enumerate(words):
        print(f"\n  Word {i+1}/{len(words)}: '{word}'")
        process_word(word)

    # Done screen
    done = np.ones((400, 600, 3), dtype='uint8') * 40
    cv2.putText(done, "Done!", (200, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 220, 100), 4, cv2.LINE_AA)
    cv2.putText(done, "Press Enter to speak again", (70, 290),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.imshow(WINDOW_NAME, done)
    cv2.waitKey(1500)

# ============================================================
# LISTEN via Microphone
# ============================================================
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n  Listening... Speak now!")
        r.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = r.listen(source, timeout=6, phrase_time_limit=8)
            print("  Processing...")
            text = r.recognize_google(audio)
            print(f"  Heard: '{text}'")
            return text
        except sr.WaitTimeoutError:
            print("  No speech detected.")
            return None
        except sr.UnknownValueError:
            print("  Could not understand audio.")
            return None
        except sr.RequestError:
            print("  Internet required for Google Speech Recognition.")
            return None

# ============================================================
# MAIN
# KEY FIX: Window created ONCE at start
#          input() called in terminal — no conflict with OpenCV
# ============================================================
if __name__ == "__main__":
    os.makedirs(ASSETS_PATH, exist_ok=True)
    os.makedirs(ALPHABET_PATH, exist_ok=True)

    print("=" * 45)
    print("  SignBridge - Speech to Sign")
    print("=" * 45)
    print(f"  Assets      : {ASSETS_PATH}/")
    print(f"  Alphabet    : {ALPHABET_PATH}/")
    print(f"  Letter delay: {DELAY_BETWEEN_ALPHABETS}s")
    print(f"  Word delay  : {DELAY_BETWEEN_WORDS}s")
    print("=" * 45)
    print("  Controls:")
    print("  - Press Enter       -> Mic input")
    print("  - Type a sentence   -> Manual input")
    print("  - Type 'q'          -> Quit")
    print("  - Press 'q' in video window -> Skip")
    print("=" * 45)

    # KEY FIX: Create window ONCE (non-blocking)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 700, 500)

    # Welcome screen
    welcome = np.ones((400, 700, 3), dtype='uint8') * 40
    cv2.putText(welcome, "SignBridge", (180, 160),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 220, 100), 4, cv2.LINE_AA)
    cv2.putText(welcome, "Speech to Sign Translator", (130, 230),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(welcome, "Go to terminal and press Enter!", (120, 310),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    cv2.imshow(WINDOW_NAME, welcome)
    cv2.waitKey(1)  # 1ms only — just to render, not block

    while True:
        print()
        user_input = input("  Press Enter (mic) | Type sentence | 'q' quit: ").strip()

        if user_input.lower() == 'q':
            print("  Goodbye!")
            break
        elif user_input == "":
            text = listen()
            if text:
                process_sentence(text.lower())
        else:
            process_sentence(user_input.lower())

        cv2.waitKey(1)  # Keep window alive

    cv2.destroyAllWindows()