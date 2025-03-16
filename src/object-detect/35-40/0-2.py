import os
import time
import cv2
import numpy as np

# Setup file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "35-40.mp4")
output_path = os.path.join(current_dir, f"rectangle_{int(time.time())}.mp4")

# Load video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Load multiple templates
templates = []
template_masks = []
for i in range(1, 5):
    template_path = os.path.join(current_dir, f"template-{i}.png")
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

    if template is None:
        print(f"Warning: Could not load template-{i}.png")
        continue

    if template.shape[2] == 4:  # Check for alpha channel
        template_masks.append(template[:, :, 3])
        templates.append(template[:, :, 0:3])
    else:
        template_masks.append(None)
        templates.append(template)

# Process only first 2 seconds
total_frames = int(fps * 2.0)

# Color palette
colors = [
    (0, 0, 255),    # Red
    (0, 165, 255),  # Orange
    (0, 255, 255),  # Yellow
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (255, 0, 255),  # Magenta
]


# Non-maximum suppression to filter overlapping rectangles
def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    # Convert to [x1, y1, x2, y2, score] format
    boxes_array = np.array(
        [
            [
                b["top_left"][0],
                b["top_left"][1],
                b["bottom_right"][0],
                b["bottom_right"][1],
                b["match_val"],
            ]
            for b in boxes
        ]
    )

    # Get coordinates
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]
    scores = boxes_array[:, 4]

    # Calculate area
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort by score
    idxs = np.argsort(scores)

    # Final boxes
    pick = []

    while len(idxs) > 0:
        # Get highest scoring box
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Calculate overlap
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Calculate overlap width and height
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Calculate overlap ratio
        overlap = (w * h) / area[idxs[:last]]

        # Remove boxes with high overlap
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )

    return [boxes[i] for i in pick]


# Downscale templates to speed up matching
downscale_factor = 1
downscaled_templates = []
downscaled_masks = []

for i, tmpl in enumerate(templates):
    # Only downscale large templates
    if tmpl.shape[0] > 100 or tmpl.shape[1] > 100:
        new_h = int(tmpl.shape[0] / downscale_factor)
        new_w = int(tmpl.shape[1] / downscale_factor)
        downscaled_tmpl = cv2.resize(tmpl, (new_w, new_h))
        downscaled_templates.append(downscaled_tmpl)

        if template_masks[i] is not None:
            downscaled_mask = cv2.resize(template_masks[i], (new_w, new_h))
            downscaled_masks.append(downscaled_mask)
        else:
            downscaled_masks.append(None)
    else:
        downscaled_templates.append(tmpl)
        downscaled_masks.append(template_masks[i])

frame_count = 0

while cap.isOpened() and frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    all_matches = []

    # Optionally downscale frame
    if downscale_factor > 1:
        processed_frame = cv2.resize(
            frame, (int(width / downscale_factor), int(height / downscale_factor))
        )
    else:
        processed_frame = frame

    for template_idx, tmpl in enumerate(downscaled_templates):
        h, w = tmpl.shape[:2]

        # Template matching
        if downscaled_masks[template_idx] is not None:
            result = cv2.matchTemplate(
                processed_frame,
                tmpl,
                cv2.TM_CCORR_NORMED,
                mask=downscaled_masks[template_idx],
            )
        else:
            result = cv2.matchTemplate(processed_frame, tmpl, cv2.TM_CCORR_NORMED)

        # Keep top 5 matches
        num_matches = 5
        for _ in range(min(num_matches, result.size)):
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val < 0.7:  # Minimum threshold
                break

            # Scale coordinates back if downscaled
            if downscale_factor > 1:
                top_left = (
                    max_loc[0] * downscale_factor,
                    max_loc[1] * downscale_factor,
                )
                orig_w = templates[template_idx].shape[1]
                orig_h = templates[template_idx].shape[0]
            else:
                top_left = max_loc
                orig_w = w
                orig_h = h

            bottom_right = (top_left[0] + orig_w, top_left[1] + orig_h)
            center = (top_left[0] + orig_w // 2, top_left[1] + orig_h // 2)

            all_matches.append(
                {
                    "template_idx": template_idx,
                    "match_val": max_val,
                    "top_left": top_left,
                    "bottom_right": bottom_right,
                    "center": center,
                    "width": orig_w,
                    "height": orig_h,
                }
            )

            # Set matched area to low value to avoid duplicates
            y, x = max_loc
            result[
                max(0, y - h // 4) : min(result.shape[0], y + h // 4),
                max(0, x - w // 4) : min(result.shape[1], x + w // 4),
            ] = 0

    # Apply NMS to filter overlapping boxes
    filtered_matches = non_max_suppression(all_matches, 0.3)

    # Limit to top 3 matches
    best_matches = filtered_matches[:3]

    # Draw flashing rectangles
    for i, match in enumerate(best_matches):
        color_idx = (i + frame_count // 5) % len(colors)
        color = colors[color_idx]

        # Flashing effect
        if frame_count % 10 < 5:
            cv2.rectangle(
                display_frame, match["top_left"], match["bottom_right"], color, 3
            )
            cv2.circle(display_frame, match["center"], 5, (255, 255, 255), -1)

            # Add match value text
            text = f"Match: {match['match_val']:.2f}"
            text_pos = (match["top_left"][0], match["top_left"][1] - 10)
            cv2.putText(
                display_frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

    # Add title
    cv2.putText(
        display_frame,
        "OBJECT DETECTION - TEMPLATE MATCHING",
        (width // 2 - 250, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    # Add description
    subtitle_y = height - 60
    overlay = display_frame.copy()
    cv2.rectangle(
        overlay, (0, subtitle_y - 10), (width, subtitle_y + 50), (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

    text = (
        "Using template matching to locate objects. Rectangles show the best matches."
    )
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

print(f"Processing complete. Output saved to {output_path}")
