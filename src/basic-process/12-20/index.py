import os
import numpy as np
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(current_dir, '12-20.mp4')
output_path = os.path.join(current_dir, '12-20_output.mp4')

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))

# HSV color range for red apple (optimized for the shown image)
# hsv_lower_red1 = np.array([0, 70, 50])
# hsv_upper_red1 = np.array([15, 255, 255])
# hsv_lower_red2 = np.array([160, 70, 50])
# hsv_upper_red2 = np.array([179, 255, 255])

# hsv_lower_red1 = np.array([0, 120, 100])
# hsv_upper_red1 = np.array([10, 255, 255])
# hsv_lower_red2 = np.array([160, 120, 100])
# hsv_upper_red2 = np.array([179, 255, 255])

hsv_lower_red1 = np.array([0, 90, 70])
hsv_upper_red1 = np.array([12, 255, 255])
hsv_lower_red2 = np.array([160, 90, 70])
hsv_upper_red2 = np.array([179, 255, 255])

# Morphological operation parameters
# morph_kernel_size = 7
# morph_kernel_size = 5
morph_kernel_size = 6

morph_kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)

improvement_color = [0, 255, 0]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask1 = cv2.inRange(hsv, hsv_lower_red1, hsv_upper_red1)
    mask2 = cv2.inRange(hsv, hsv_lower_red2, hsv_upper_red2)
    mask = mask1 + mask2
    
    binary = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    improved_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)
    improved_mask = cv2.morphologyEx(improved_mask, cv2.MORPH_OPEN, morph_kernel)
    
    improved_binary = cv2.cvtColor(improved_mask, cv2.COLOR_GRAY2BGR)
    
    diff = cv2.subtract(improved_mask, mask)
    diff_colored = np.zeros_like(improved_binary)
    diff_colored[diff > 0] = improvement_color
    
    result = cv2.addWeighted(improved_binary, 1, diff_colored, 0.5, 0)
    
    output_frame = np.hstack((binary, result))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output_frame, "Binary Mask", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(output_frame, "Improved Mask (green highlights)", (width + 10, 30), font, 1, (255, 255, 255), 2)
    
    out.write(output_frame)

cap.release()
out.release()
