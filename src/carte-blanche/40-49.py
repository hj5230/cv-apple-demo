import os
import time
import cv2
import numpy as np

# Setup file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "40-60.mp4")
output_path = os.path.join(current_dir, f"rainbow_trails_{int(time.time())}.mp4")

# Load video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process only first 9 seconds
total_frames = int(fps * 9.0)

# Initialize variables for tracking
tracks = []
max_tracks = 30  # Maximum number of positions to remember

# Color palette
colors = [
    (0, 0, 255),  # Red
    (0, 165, 255),  # Orange
    (0, 255, 255),  # Yellow
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    (255, 0, 255),  # Magenta
]

frame_count = 0

while cap.isOpened() and frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Create display frame
    display_frame = frame.copy()

    # Apple detection using HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red apple HSV range (adjust as needed)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find apple center
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Add current position to tracks
        tracks.append((cx, cy))
        # Keep only recent tracks
        if len(tracks) > max_tracks:
            tracks.pop(0)

        # Draw rainbow trail effect
        for i in range(1, len(tracks)):
            # Color changes based on position in trail
            color_idx = i % len(colors)
            color = colors[color_idx]

            thickness = max(1, int(5 * (i / len(tracks))))

            cv2.line(display_frame, tracks[i - 1], tracks[i], color, thickness)

        # Draw current position
        cv2.circle(display_frame, (cx, cy), 10, (0, 0, 255), -1)

    # Add title
    cv2.putText(
        display_frame,
        "MOTION TRACKING WITH RAINBOW TRAILS",
        (width // 2 - 250, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    # Add subtitle
    subtitle_y = height - 60
    overlay = display_frame.copy()
    cv2.rectangle(
        overlay, (0, subtitle_y - 10), (width, subtitle_y + 50), (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

    text = "Tracking object movement with colorful motion trails showing path history"
    cv2.putText(
        display_frame,
        text,
        (10, subtitle_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Write to output video
    out.write(display_frame)
    frame_count += 1

# Release resources
cap.release()
out.release()

print(f"Part 1 (Rainbow Trails) processing complete. Output saved to {output_path}")
