import os
import time
import cv2
import numpy as np

# Set file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "35-40.mp4")
output_path = os.path.join(current_dir, f"grayscale_overlay_{int(time.time())}.mp4")
first_frames_path = os.path.join(current_dir, f"first_5frames_{int(time.time())}.mp4")

# Load video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
first_frames_out = cv2.VideoWriter(first_frames_path, fourcc, fps, (width, height))

# Load multiple templates
templates = []
template_masks = []
for i in range(1, 5):
    template_path = os.path.join(current_dir, f"template-{i}.png")
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

    if template is None:
        print(f"Warning: Could not load template-{i}.png")
        continue

    if template.shape[2] == 4:  # Check for transparency channel
        template_masks.append(template[:, :, 3])
        templates.append(template[:, :, 0:3])
    else:
        template_masks.append(None)
        templates.append(template)

# Skip first 2 seconds
frames_to_skip = int(fps * 2.0)
for _ in range(frames_to_skip):
    ret, _ = cap.read()
    if not ret:
        print("Error: Could not skip frames")
        cap.release()
        exit(1)

# Process the next 3 seconds
total_frames = int(fps * 3.0)
frame_count = 0

while cap.isOpened() and frame_count < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Create a zero matrix for storing likelihood
    likelihood_map = np.zeros((height, width), dtype=np.float32)

    # Calculate similarity map for each template
    for template_idx, tmpl in enumerate(templates):
        # Convert to grayscale for better matching
        gray_tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = gray_tmpl.shape

        # Template matching
        if template_masks[template_idx] is not None:
            result = cv2.matchTemplate(
                gray_frame,
                gray_tmpl,
                cv2.TM_CCOEFF_NORMED,
                mask=template_masks[template_idx],
            )
        else:
            result = cv2.matchTemplate(gray_frame, gray_tmpl, cv2.TM_CCOEFF_NORMED)

        # Create a full-size temporary map
        temp_map = np.zeros((height, width), dtype=np.float32)

        # Get result dimensions
        res_h, res_w = result.shape

        # Map each result point to center position
        for y in range(res_h):
            for x in range(res_w):
                # Only process matches above threshold
                if result[y, x] > 0.7:
                    # Calculate center position
                    center_y = y + h // 2
                    center_x = x + w // 2

                    # Ensure within image bounds
                    if 0 <= center_y < height and 0 <= center_x < width:
                        # Update likelihood map (take max value)
                        temp_map[center_y, center_x] = max(
                            temp_map[center_y, center_x], result[y, x]
                        )

        # Apply Gaussian blur for smoother distribution
        if np.max(temp_map) > 0:
            temp_map = cv2.GaussianBlur(temp_map, (31, 31), 0)

            # Update total likelihood map (take max value)
            likelihood_map = np.maximum(likelihood_map, temp_map)

    # Normalize to 0-255 range
    if np.max(likelihood_map) > 0:
        likelihood_map = cv2.normalize(likelihood_map, None, 0, 255, cv2.NORM_MINMAX)

    # Create final grayscale map
    final_map = likelihood_map.astype(np.uint8)

    # Convert grayscale to color heatmap
    gray_heatmap = cv2.applyColorMap(final_map, cv2.COLORMAP_JET)

    # Create blue border
    blue_border = np.zeros((height, width, 3), dtype=np.uint8)
    blue_border[:, :] = (255, 0, 0)  # Blue (BGR format)

    # Create a mask with 1 at border, 0 inside
    border_width = 4
    border_mask = np.ones((height, width), dtype=np.uint8)
    border_mask[border_width:-border_width, border_width:-border_width] = 0

    # Create composite image starting with original frame
    composite = frame.copy()

    # Add heatmap overlay with transparency
    alpha_heatmap = 0.7
    mask = final_map > 30  # Only show heatmap where values exceed threshold

    # Apply blending in valid areas
    composite[mask] = cv2.addWeighted(
        composite[mask], 1 - alpha_heatmap, gray_heatmap[mask], alpha_heatmap, 0
    )

    # Apply blue border
    composite[border_mask == 1] = blue_border[border_mask == 1]

    # Add title
    cv2.putText(
        composite,
        "OBJECT DETECTION - PROBABILITY MAP",
        (width // 2 - 250, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    # Add explanation text
    subtitle_y = height - 60
    overlay = composite.copy()
    cv2.rectangle(
        overlay, (0, subtitle_y - 10), (width, subtitle_y + 50), (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, 0.7, composite, 0.3, 0, composite)

    text = "The probability map shows likelihood of object presence. Brighter areas indicate higher probability."
    cv2.putText(
        composite,
        text,
        (10, subtitle_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Add color bar legend
    bar_width = 200
    bar_height = 20
    bar_x = width - bar_width - 20
    bar_y = height - 100

    # Create gradient color bar
    gradient = np.linspace(0, 255, bar_width).astype(np.uint8)
    gradient = np.tile(gradient, (bar_height, 1))
    gradient_color = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

    # Add color bar to frame
    composite[bar_y : bar_y + bar_height, bar_x : bar_x + bar_width] = gradient_color

    # Add labels to color bar
    cv2.putText(
        composite,
        "Low",
        (bar_x, bar_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        composite,
        "High",
        (bar_x + bar_width - 30, bar_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # Write to main output video
    out.write(composite)

    # Save first 5 frames to separate video
    if frame_count < 5:
        first_frames_out.write(composite)

    frame_count += 1

    print(f"Processed frame {frame_count}/{total_frames}")

# Release resources
cap.release()
out.release()
first_frames_out.release()

print(f"Grayscale overlay complete. Output saved to {output_path}")
print(f"First 5 frames saved to {first_frames_path}")
