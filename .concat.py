import cv2
import os
import sys
import math


def get_video_properties(video_path):
    """Get video properties like fps, width, height"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return fps, width, height


def concatenate_videos(video_files, output_file, target_fps=30.0):
    """Concatenate the list of video files into a single output file with normalized frame rate"""
    if not video_files:
        print("No video files to concatenate.")
        return False

    # Get dimensions from the first video (we'll use these for all videos)
    _, width, height = get_video_properties(video_files[0])
    if width is None:
        print(f"Could not open video file: {video_files[0]}")
        return False

    # Create video writer with target FPS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, target_fps, (width, height))

    # Process each video in sequence
    for i, video_file in enumerate(video_files):
        print(f"Processing video {i+1}/{len(video_files)}: {video_file}")
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print(f"Could not open video file: {video_file}")
            out.release()
            return False

        # Get the source video's frame rate
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  - Source FPS: {source_fps}, Target FPS: {target_fps}")

        # Calculate frame skipping ratio
        if source_fps > target_fps:
            # We need to skip frames
            frame_skip_ratio = source_fps / target_fps
            print(f"  - Frame skip ratio: {frame_skip_ratio}")
        else:
            # No skipping needed (we'll use all frames)
            frame_skip_ratio = 1.0

        frame_count = 0
        frames_to_write = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize if dimensions don't match
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            # Determine if we should keep this frame based on our ratio
            frames_to_write += 1.0 / frame_skip_ratio

            if math.floor(frames_to_write) > frame_count:
                out.write(frame)
                frame_count += 1

        cap.release()
        print(f"  - Frames processed: {frame_count}")

    # Release resources
    out.release()
    print(f"Concatenation completed! Output file: {output_file}")
    return True


def main():
    print("Video Concatenation Tool")
    print("------------------------")
    print(
        "Enter file paths one by one. Type 'done' when finished to start concatenation."
    )

    video_files = []

    while True:
        file_input = input("Enter video file path (or 'done' to finish): ").strip()

        if file_input.lower() == "done":
            if not video_files:
                print("No files added. Please add at least one file.")
                continue
            break

        if not os.path.exists(file_input):
            print(f"File not found: {file_input}")
            continue

        # Check if it's a valid video file
        fps, width, height = get_video_properties(file_input)
        if fps is None:
            print(f"Could not open as a video file: {file_input}")
            continue

        video_files.append(file_input)
        print(
            f"Added: {file_input} (FPS: {fps}, Resolution: {width}x{height}, Total files: {len(video_files)})"
        )

    # Ask for output file
    output_file = input("Enter output file path (e.g., output.mp4): ").strip()
    if not output_file:
        output_file = "output.mp4"

    # Ask for target frame rate
    while True:
        try:
            target_fps_input = input("Enter target FPS (default is 30): ").strip()
            if not target_fps_input:
                target_fps = 30.0
                break
            target_fps = float(target_fps_input)
            if target_fps <= 0:
                print("FPS must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    # Ask for confirmation
    print("\nReady to concatenate the following files:")
    for i, file in enumerate(video_files):
        fps, _, _ = get_video_properties(file)
        print(f"{i+1}. {file} (FPS: {fps})")
    print(f"Output will be saved to: {output_file} with {target_fps} FPS")

    confirm = input("\nStart concatenation? (y/n): ").strip().lower()
    if confirm != "y":
        print("Concatenation cancelled.")
        return

    # Start concatenation
    success = concatenate_videos(video_files, output_file, target_fps)

    if success:
        print(f"Videos successfully concatenated to {output_file}")
    else:
        print("Concatenation failed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)
