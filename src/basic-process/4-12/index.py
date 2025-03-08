import os
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(current_dir, '4-12.mp4')
output_path = os.path.join(current_dir, '4-12_output.mp4')

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

switch_interval = 0.5 # unit is second
timestamp = 0
frame_count = 0

kernel_sizes = [5, 11, 21, 31]

gaussian_subtitle = "Gaussian Filter: Uniform blurring that smooths everything equally, including edges"
bilateral_subtitle = "Bilateral Filter: Edge-preserving smoothing that keeps sharp edges while blurring flat regions"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_count / fps

    cycle_position = int(timestamp / switch_interval) % (len(kernel_sizes) * 2 + 1)

    if cycle_position == 0:
        processed = frame.copy()
        subtitle = "Original (No Filter)"

    elif cycle_position <= len(kernel_sizes):
        k_size = kernel_sizes[cycle_position - 1]
        processed = cv2.GaussianBlur(frame, (k_size, k_size), 0)
        subtitle = f"{gaussian_subtitle} (kernel size = {k_size}x{k_size})"

    else:
        k_idx = cycle_position - len(kernel_sizes) - 1
        d = kernel_sizes[k_idx]
        sigma_color = 75
        sigma_space = 75
        processed = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
        subtitle = f"{bilateral_subtitle} (d = {d}, sigmaColor = {sigma_color}, sigmaSpace = {sigma_space})"

    overlay = processed.copy()

    cv2.rectangle(overlay, (0, height-70), (width, height), (0, 0, 0), -1)

    alpha = 0.7
    processed = cv2.addWeighted(overlay, alpha, processed, 1-alpha, 0)

    cv2.putText(processed, subtitle, (20, height-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out.write(processed)
    frame_count += 1

cap.release()
out.release()
