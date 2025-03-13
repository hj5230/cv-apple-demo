import os
import time
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "25-35.mp4")
output_path = os.path.join(current_dir, f"{int(time.time())}.mp4")

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

total_frames = int(fps * 10.0)
frame_count = 0

# Color palette for visualization
colors = [
    (0, 0, 255),  # Red
    (0, 165, 255),  # Orange
    (0, 255, 255),  # Yellow
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    (255, 0, 255),  # Magenta
]

best_params = {"param1": 78, "param2": 22, "minRadius": 342, "maxRadius": 411}

while cap.isOpened() and frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate which section we're in (0-4)
    # 0: 0-2s = Best case
    # 1: 2-4s = Change param1
    # 2: 4-6s = Change param2
    # 3: 6-8s = Change minRadius
    # 4: 8-10s = Change maxRadius
    time_position = frame_count / total_frames * 10
    section = int(time_position / 2)

    section_progress = (time_position % 2) / 2

    param1 = best_params["param1"]
    param2 = best_params["param2"]
    min_radius = best_params["minRadius"]
    max_radius = best_params["maxRadius"]

    # Adjust one parameter based on which section we're in
    if section == 1:  # 2-4s: Change param1 (78 to 30)
        param1 = best_params["param1"] - int(section_progress * 48)
    elif section == 2:  # 4-6s: Change param2 (22 to 5)
        param2 = best_params["param2"] - int(section_progress * 17)
    elif section == 3:  # 6-8s: Lower min_radius (342 to 200)
        min_radius = best_params["minRadius"] - int(section_progress * 142)
    elif section == 4:  # 8-10s: Raise max_radius (411 to 550)
        max_radius = best_params["maxRadius"] + int(section_progress * 139)

    frame_count += 1

    display_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))

        sorted_circles = sorted(circles[0], key=lambda x: x[2], reverse=True)

        for i, circle in enumerate(sorted_circles[:3]):
            x, y, r = circle

            color_idx = (i + frame_count // 5) % len(colors)
            color = colors[color_idx]

            cv2.circle(display_frame, (x, y), r, color, 3)

            cv2.circle(display_frame, (x, y), 5, (255, 255, 255), -1)

            angle = (frame_count * 12) % 360
            rad_angle = np.deg2rad(angle)
            end_x = int(x + r * np.cos(rad_angle))
            end_y = int(y + r * np.sin(rad_angle))
            cv2.line(display_frame, (x, y), (end_x, end_y), color, 2)

    # Display current section and parameters
    section_names = [
        "BEST PARAMETERS",
        "CHANGING param1",
        "CHANGING param2",
        "CHANGING minRadius",
        "CHANGING maxRadius",
    ]

    # Add a title showing which parameter is changing
    cv2.putText(
        display_frame,
        section_names[section],
        (width // 2 - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    y_pos = 70

    # param1
    color = (255, 255, 0) if section == 1 else (255, 255, 255)
    cv2.putText(
        display_frame,
        f"param1: {param1}",
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
    y_pos += 30

    # param2
    color = (255, 255, 0) if section == 2 else (255, 255, 255)
    cv2.putText(
        display_frame,
        f"param2: {param2}",
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
    y_pos += 30

    # minRadius
    color = (255, 255, 0) if section == 3 else (255, 255, 255)
    cv2.putText(
        display_frame,
        f"minRadius: {min_radius}",
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
    y_pos += 30

    # maxRadius
    color = (255, 255, 0) if section == 4 else (255, 255, 255)
    cv2.putText(
        display_frame,
        f"maxRadius: {max_radius}",
        (10, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    impact_descriptions = [
        "These optimal parameters precisely detect the apple's outline. The circle detection is stable and accurate.",
        "param1 defines the high Canny edge threshold. Lower values detect more edges but increase noise and false detections.",
        "param2 is the circle detection threshold. Lower values make detection more permissive but may create false positives.",
        "minRadius sets the smallest detectable circle size. Lower values can detect smaller features but may include unwanted objects.",
        "maxRadius sets the largest detectable circle size. Higher values can detect larger objects but may lose precision.",
    ]

    subtitle_y = height - 60
    overlay = display_frame.copy()
    cv2.rectangle(
        overlay, (0, subtitle_y - 10), (width, subtitle_y + 50), (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

    text = impact_descriptions[section]
    if len(text) > 70:  # Split into two lines if text is long
        split_point = text.rfind(" ", 0, 70)
        line1 = text[:split_point]
        line2 = text[split_point + 1 :]

        cv2.putText(
            display_frame,
            line1,
            (10, subtitle_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            line2,
            (10, subtitle_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    else:
        cv2.putText(
            display_frame,
            text,
            (10, subtitle_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    out.write(display_frame)

cap.release()
out.release()
