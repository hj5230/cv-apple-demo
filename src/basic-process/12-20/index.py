import cv2
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "12-20.mp4")
output_path = os.path.join(current_dir, "12-20_output.mp4")

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
halfway_point = total_frames // 2

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Converting RGB to HSV ranges
hsv_lower = np.array([10, 50, 75])  # Lower bound for tan/brown
hsv_upper = np.array([30, 255, 255])  # Upper bound for tan/brown

# Morphological operation parameters
morph_kernel_size = 5
morph_kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

# Green for highlighting improvements
improvement_color = [0, 255, 0]

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    improved_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)
    improved_mask = cv2.morphologyEx(improved_mask, cv2.MORPH_OPEN, morph_kernel)

    if frame_count < halfway_point:
        binary_output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            binary_output,
            "Binary Mask",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        out.write(binary_output)
    else:
        diff = cv2.subtract(improved_mask, mask)
        improved_binary = cv2.cvtColor(improved_mask, cv2.COLOR_GRAY2BGR)

        diff_colored = np.zeros_like(improved_binary)
        diff_colored[diff > 0] = improvement_color

        result = cv2.addWeighted(improved_binary, 1, diff_colored, 0.5, 0)
        cv2.putText(
            result,
            "Improved Mask (green highlights)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        out.write(result)

    frame_count += 1

cap.release()
out.release()
