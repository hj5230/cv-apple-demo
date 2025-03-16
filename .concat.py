import cv2
import os
import sys


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


def concatenate_videos(video_files, output_file):
    """Concatenate the list of video files into a single output file"""
    if not video_files:
        print("No video files to concatenate.")
        return False

    # Get properties from the first video
    fps, width, height = get_video_properties(video_files[0])
    if fps is None:
        print(f"Could not open video file: {video_files[0]}")
        return False

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Process each video in sequence
    for i, video_file in enumerate(video_files):
        print(f"Processing video {i+1}/{len(video_files)}: {video_file}")
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print(f"Could not open video file: {video_file}")
            out.release()
            return False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()

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
        print(f"Added: {file_input} (Total files: {len(video_files)})")

    # Ask for output file
    output_file = input("Enter output file path (e.g., output.mp4): ").strip()
    if not output_file:
        output_file = "output.mp4"

    # Ask for confirmation
    print("\nReady to concatenate the following files:")
    for i, file in enumerate(video_files):
        print(f"{i+1}. {file}")
    print(f"Output will be saved to: {output_file}")

    confirm = input("\nStart concatenation? (y/n): ").strip().lower()
    if confirm != "y":
        print("Concatenation cancelled.")
        return

    # Start concatenation
    success = concatenate_videos(video_files, output_file)

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
