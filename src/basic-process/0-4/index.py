import os
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(current_dir, "0-4.mp4")
output_path = os.path.join(current_dir, "0-4_output.mp4")

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

switch_interval = 0.5  # unit is second
timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = (
        cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    )  # ms -> s, now everything in second

    if int(timestamp / switch_interval) % 2 == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    out.write(frame)

cap.release()
out.release()

# It is wired that cv convert my vertical video to horizontal video
