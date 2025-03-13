import os
import cv2
import numpy as np


def nothing(x):
    pass


cv2.namedWindow("Detected Circles")

cv2.createTrackbar("Param1", "Detected Circles", 78, 200, nothing)
cv2.createTrackbar("Param2", "Detected Circles", 22, 100, nothing)
cv2.createTrackbar("Min Radius", "Detected Circles", 342, 500, nothing)
cv2.createTrackbar("Max Radius", "Detected Circles", 411, 1000, nothing)

current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "25-35.mp4")
cap = cv2.VideoCapture(video_path)

paused = False
original_frame = None

ret, frame = cap.read()
if not ret:
    print("无法读取视频")
    exit()

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        original_frame = frame.copy()

    display_frame = original_frame.copy()

    param1 = cv2.getTrackbarPos("Param1", "Detected Circles")
    param2 = cv2.getTrackbarPos("Param2", "Detected Circles")
    min_radius = cv2.getTrackbarPos("Min Radius", "Detected Circles")
    max_radius = cv2.getTrackbarPos("Max Radius", "Detected Circles")

    gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        20,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(display_frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("Detected Circles", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        paused = not paused
        if paused:
            original_frame = frame.copy()

cap.release()
cv2.destroyAllWindows()
