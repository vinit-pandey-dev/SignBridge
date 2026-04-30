import cv2
import mediapipe as mp

# 1. Setup MediaPipe
mp_holistic = mp.solutions.holistic # Holistic model (Face + Hands + Pose)
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# 2. Define a function to make the detections
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR (OpenCV default) to RGB (MediaPipe needs this)
    image.flags.writeable = False                  # Lock image to improve speed
    results = model.process(image)                 # Make the prediction!
    image.flags.writeable = True                   # Unlock image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back to BGR to show on screen
    return image, results

# 3. Define a function to draw the lines (The "Skeleton")
def draw_styled_landmarks(image, results):
    # Draw Face Connections
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    )
    # Draw Pose Connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )
    # Draw Left Hand
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    # Draw Right Hand
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

# 4. Main Loop
cap = cv2.VideoCapture(0) # 0 is usually the default webcam. Try 1 if 0 fails.

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read() # Read feed

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('SignBridge Camera Test', image)

        # Break loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()