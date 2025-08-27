import cv2

cap = cv2.VideoCapture(0)

tm = cv2.TickMeter()
tm.start()

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.Canny(frame, 50, 150)
        inversed_edge = ~frame

        cv2.imshow('Canny Edge', frame)

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)

        tm.stop()
        real_fps = tm.getFPS()
        print(real_fps)
        tm.reset(); tm.start()
        if cv2.waitKey(1) & 0xFF == 27:
            break

else: print("Camera not opened")

cap.release()
cv2.destroyAllWindows()