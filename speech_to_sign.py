import cv2
import os
import time
import speech_recognition as sr
import numpy as np

# ============================================================
# SIGNBRIDGE - SPEECH TO SIGN (IMPROVED)
# Changes:
#   1. Alphabet delay 0.5s → 0.2s (faster)
#   2. Word ke baad blank "space" frame
#   3. Video FPS speed up (30ms → 20ms)
#   4. Intro screen removed (faster start)
#   5. Letter highlight animation added
#   6. Word progress bar added
# ============================================================

ASSETS_PATH = "Assets"
ALPHABET_PATH = os.path.join(ASSETS_PATH, "Alphabet")
DELAY_BETWEEN_ALPHABETS = 0.2   # FIX 1: 0.5 → 0.2 (faster letters)
DELAY_BETWEEN_WORDS = 0.5       # Word ke baad gap
SPACE_FRAME_DURATION = 500      # ms — blank frame between words
VIDEO_FPS_DELAY = 20            # FIX 3: 30 → 20ms (faster video)
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
# FIX 2: SPACE FRAME between words
# ============================================================
def show_word_space(word_label):
    """Black frame with '|' separator shown between words"""
    space = np.ones((400, 600, 3), dtype='uint8') * 15
    # Word done indicator
    cv2.putText(space, f"[ {word_label.upper()} ]", (170, 170),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 60), 2, cv2.LINE_AA)
    # Space symbol
    cv2.putText(space, "___", (250, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 80), 2, cv2.LINE_AA)
    cv2.putText(space, "NEXT WORD", (195, 310),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.imshow(WINDOW_NAME, space)
    cv2.waitKey(SPACE_FRAME_DURATION)

# ============================================================
# PLAY VIDEO
# ============================================================
def play_video(video_path, label, mode="word"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}")
        return
    
    # Asli Video FPS nikaalein
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: 
        fps = 30.0
    
    frame_duration = 1.0 / fps  
    print(f"  Playing: {label} (FPS: {fps})")
    
    start_time = time.time()
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        expected_time = start_time + (frame_counter * frame_duration)
        current_time = time.time()
        
        # LAG CONTROL: Agar frame late hai toh skip karo
        if current_time > expected_time + 0.03:
            continue
            
        frame = draw_overlay(frame, f"Signing: {label}", label, mode)
        cv2.imshow(WINDOW_NAME, frame)
        
        time_to_wait = expected_time - time.time()
        if time_to_wait > 0:
            delay = max(1, int(time_to_wait * 1000))
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
# ============================================================
# SHOW IMAGE
# ============================================================
def show_image(image_path, label, duration=0.8):  # FIX: 1.0 → 0.8s
    frame = cv2.imread(image_path)
    if frame is None:
        return
    frame = draw_overlay(frame, f"Letter: {label}", label, "letter")
    cv2.imshow(WINDOW_NAME, frame)
    start = time.time()
    while time.time() - start < duration:
        if cv2.waitKey(30) & 0xFF == ord('q'):
            return

# ============================================================
# PLAY LETTER with highlight animation
# ============================================================
def play_letter(letter, word, letter_index):
    """Play letter with word progress shown at bottom"""
    path = find_letter_asset(letter)

    if path is None:
        # Show text frame if no asset
        blank = np.ones((400, 500, 3), dtype='uint8') * 50
        cv2.putText(blank, letter.upper(), (170, 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 100, 255), 12, cv2.LINE_AA)
        # Word progress at bottom
        _draw_word_progress(blank, word, letter_index)
        cv2.imshow(WINDOW_NAME, blank)
        cv2.waitKey(int(DELAY_BETWEEN_ALPHABETS * 1000))
        return

    ext = os.path.splitext(path)[1].lower()

    if ext in ['.mp4', '.avi', '.mov']:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Draw overlay
            draw_overlay(frame, f"Spelling: {word.upper()}", letter.upper(), "letter")
            # Word progress bar
            _draw_word_progress(frame, word, letter_index)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(VIDEO_FPS_DELAY) & 0xFF == ord('q'):
                cap.release()
                return
        cap.release()
    else:
        frame = cv2.imread(path)
        if frame is not None:
            draw_overlay(frame, f"Spelling: {word.upper()}", letter.upper(), "letter")
            _draw_word_progress(frame, word, letter_index)
            cv2.imshow(WINDOW_NAME, frame)
            start = time.time()
            while time.time() - start < 0.8:
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    return

    time.sleep(DELAY_BETWEEN_ALPHABETS)

def _draw_word_progress(frame, word, current_index):
    """FIX 5: Show word with current letter highlighted at bottom"""
    h, w = frame.shape[:2]
    # Progress bar background
    cv2.rectangle(frame, (0, h - 90), (w, h - 45), (30, 30, 30), -1)

    # Draw each letter — current one highlighted
    total = len(word)
    letter_w = min(40, (w - 40) // max(total, 1))
    start_x = (w - total * letter_w) // 2

    for i, ch in enumerate(word.upper()):
        x = start_x + i * letter_w
        y = h - 55
        if i == current_index:
            # Current letter — bright highlight
            cv2.rectangle(frame, (x - 3, y - 25), (x + letter_w - 5, y + 5),
                         (0, 165, 255), -1)
            cv2.putText(frame, ch, (x + 2, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif i < current_index:
            # Done letters — dim green
            cv2.putText(frame, ch, (x + 2, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 80), 1)
        else:
            # Pending letters — grey
            cv2.putText(frame, ch, (x + 2, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)

# ============================================================
# PROCESS WORD: Smart lookup
# ============================================================
def process_word(word, word_index, total_words):
    clean = ''.join(c for c in word if c.isalpha())
    if not clean:
        return

    path = find_word_asset(clean)

    if path:
        print(f"\n  [WORD] Video found for '{clean}'")
        play_video(path, clean.capitalize(), mode="word")
    else:
        print(f"\n  [SPELL] Finger spelling: {clean.upper()}")

        # FIX 4: Removed long intro — just quick label
        intro = np.ones((400, 600, 3), dtype='uint8') * 40
        cv2.putText(intro, clean.upper(),
                   (int(300 - len(clean) * 20), 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 165, 255), 4, cv2.LINE_AA)
        cv2.putText(intro, "Spelling...", (220, 290),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        cv2.imshow(WINDOW_NAME, intro)
        cv2.waitKey(600)  # FIX 4: 1200 → 600ms

        for i, letter in enumerate(clean):
            print(f"    -> {letter.upper()}")
            play_letter(letter, clean, i)

    # FIX 2: Show space frame between words (not after last word)
    if word_index < total_words - 1:
        show_word_space(clean)

# ============================================================
# PROCESS FULL SENTENCE
# ============================================================
def process_sentence(text):
    print(f"\n{'='*45}")
    print(f"  Sentence: \"{text}\"")
    print(f"{'='*45}")

    words = text.strip().split()
    total = len(words)

    # Quick overview — shorter wait
    overview = np.ones((400, 700, 3), dtype='uint8') * 40
    cv2.putText(overview, "You said:", (30, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    display_text = f'"{text}"' if len(text) < 32 else f'"{text[:30]}..."'
    cv2.putText(overview, display_text, (30, 175),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 100), 2, cv2.LINE_AA)
    cv2.putText(overview, f"{total} word{'s' if total > 1 else ''}", (30, 250),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 50), 2)
    cv2.imshow(WINDOW_NAME, overview)
    cv2.waitKey(1200)  # FIX 4: 1800 → 1200ms

    for i, word in enumerate(words):
        print(f"\n  Word {i+1}/{total}: '{word}'")
        process_word(word, i, total)

    # Done screen
    done = np.ones((400, 600, 3), dtype='uint8') * 40
    cv2.putText(done, "Done!", (195, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 220, 100), 4, cv2.LINE_AA)
    cv2.putText(done, "Press Enter to speak again", (70, 290),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    cv2.imshow(WINDOW_NAME, done)
    cv2.waitKey(1200)

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
            print("  Internet required.")
            return None

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs(ASSETS_PATH, exist_ok=True)
    os.makedirs(ALPHABET_PATH, exist_ok=True)

    print("=" * 45)
    print("  SignBridge - Speech to Sign")
    print("=" * 45)
    print(f"  Letter delay : {DELAY_BETWEEN_ALPHABETS}s")
    print(f"  Word gap     : {SPACE_FRAME_DURATION}ms")
    print(f"  Video speed  : {VIDEO_FPS_DELAY}ms/frame")
    print("=" * 45)
    print("  Enter → Mic | Type → Manual | q → Quit")
    print("=" * 45)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 700, 500)

    # Welcome screen
    welcome = np.ones((400, 700, 3), dtype='uint8') * 40
    cv2.putText(welcome, "SignBridge", (180, 160),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 220, 100), 4, cv2.LINE_AA)
    cv2.putText(welcome, "Speech to Sign Translator", (130, 230),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(welcome, "Go to terminal and press Enter!", (110, 305),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    cv2.imshow(WINDOW_NAME, welcome)
    cv2.waitKey(1)

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

        cv2.waitKey(1)

    cv2.destroyAllWindows()