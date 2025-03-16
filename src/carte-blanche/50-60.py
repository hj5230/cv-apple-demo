import os
import time
import cv2
import numpy as np

# Setup file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "40-60.mp4")
output_path = os.path.join(current_dir, f"cartoon_effect_{int(time.time())}.mp4")

# Load video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Calculate frame positions
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
first_part_frames = int(fps * 9.0)  # First 9 seconds
second_part_frames = min(
    int(fps * 10.0), total_frames - first_part_frames
)  # Last 10 seconds


# Cartoon effect function
def cartoon_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


# Skip to 9 seconds mark
cap.set(cv2.CAP_PROP_POS_FRAMES, first_part_frames)
frame_count = 0

while cap.isOpened() and frame_count < second_part_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Create display frame
    display_frame = frame.copy()

    # Apple detection using HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red apple HSV ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    apple_mask = cv2.bitwise_or(mask1, mask2)

    # Clean up mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    apple_mask = cv2.morphologyEx(apple_mask, cv2.MORPH_CLOSE, kernel)
    apple_mask = cv2.morphologyEx(apple_mask, cv2.MORPH_OPEN, kernel)
    apple_mask = cv2.dilate(apple_mask, kernel, iterations=2)

    # Apply cartoon effect to entire frame
    cartoon_frame = cartoon_effect(frame)

    # Create blended result
    result = np.copy(frame)

    # Apply cartoon effect only to apple region
    result[apple_mask > 0] = cartoon_frame[apple_mask > 0]

    # Highlight edges for better effect
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200)
    edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8))

    # Apply edges only to apple region
    apple_edges = cv2.bitwise_and(edges_dilated, apple_mask)
    result[apple_edges > 0] = (0, 0, 0)  # Black edges

    # Add title
    cv2.putText(
        result,
        "SELECTIVE CARTOON EFFECT",
        (width // 2 - 180, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    # Add subtitle
    subtitle_y = height - 60
    overlay = result.copy()
    cv2.rectangle(
        overlay, (0, subtitle_y - 10), (width, subtitle_y + 50), (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)

    text = "Applying cartoon effect only to the detected object while preserving background"
    cv2.putText(
        result,
        text,
        (10, subtitle_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Write to output video
    out.write(result)
    frame_count += 1

# Release resources
cap.release()
out.release()

print(f"Part 2 (Cartoon Effect) processing complete. Output saved to {output_path}")
