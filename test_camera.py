import cv2

backends = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("DEFAULT", 0)
]

for name, backend in backends:
    print(f"Trying backend: {name}")

    if backend == 0:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0, backend)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"{name}: Camera not opened")
        continue

    for _ in range(20):
        ret, frame = cap.read()

    if not ret or frame is None:
        print(f"{name}: Frame not received")
        cap.release()
        continue

    print(f"{name}: Working")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame lost")
            break

        cv2.imshow(f"Camera Test - {name}", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()

cv2.destroyAllWindows()