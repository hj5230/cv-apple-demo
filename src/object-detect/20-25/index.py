import os
import time
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "20-25.mp4")
output_path = os.path.join(current_dir, f"{int(time.time())}.mp4")

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

total_frames = int(fps * 5.0)
frame_count = 0

while cap.isOpened() and frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    progress = frame_count / total_frames
    if progress < 0.33:
        kernel_size = 3
    elif progress < 0.67:
        kernel_size = 5
    else:
        kernel_size = 7

    frame_count += 1

    # Isolate blue bottle cap
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    blue_only = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(blue_only, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Sobel edge detection
    sobel_h = cv2.Sobel(
        blurred, cv2.CV_64F, 0, 1, ksize=kernel_size
    )  # Horizontal
    sobel_v = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=kernel_size)  # Vertical

    # Convert to absolute values and scale to 8-bit
    abs_sobel_h = cv2.convertScaleAbs(sobel_h)
    abs_sobel_v = cv2.convertScaleAbs(sobel_v)

    # Create colored edge visualization
    colored_edges = np.zeros_like(frame)
    colored_edges[:, :, 0] = abs_sobel_v  # Blue channel for vertical edges
    colored_edges[:, :, 1] = abs_sobel_h  # Green channel for horizontal edges

    cv2.putText(
        colored_edges,
        f"Sobel Kernel: {kernel_size}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    out.write(colored_edges)

cap.release()
out.release()
