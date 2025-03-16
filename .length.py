import cv2
import os
import argparse
import sys
from tqdm import tqdm


def accelerate_video(input_file, output_file, target_duration=None, speed_factor=None):
    """
    Accelerate a video by either specifying a target duration or a speed factor.

    Parameters:
        input_file: Path to the input video file
        output_file: Path to the output video file
        target_duration: Target duration in seconds (e.g., 60 for 1 minute)
        speed_factor: Speed multiplication factor (e.g., 1.5 for 50% faster)
    """
    # Verify the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False

    # Open the input video
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: Could not open video '{input_file}'.")
        return False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_duration = frame_count / fps

    # Calculate the new FPS based on the acceleration method
    if target_duration is not None:
        if target_duration <= 0:
            print("Error: Target duration must be positive.")
            cap.release()
            return False

        speed_factor = original_duration / target_duration
        new_fps = fps * speed_factor
    elif speed_factor is not None:
        if speed_factor <= 0:
            print("Error: Speed factor must be positive.")
            cap.release()
            return False

        new_fps = fps * speed_factor
        target_duration = original_duration / speed_factor
    else:
        print("Error: Either target_duration or speed_factor must be specified.")
        cap.release()
        return False

    # Create the output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can change codec as needed
    out = cv2.VideoWriter(output_file, fourcc, new_fps, (width, height))

    # Calculate how many frames to skip
    # We'll use a frame selection approach rather than changing the speed
    if speed_factor <= 1.0:
        # For slowing down (not our goal here, but including for completeness)
        frame_step = 1.0 / speed_factor
        print("Warning: You're slowing down the video, not accelerating it.")
    else:
        # For speeding up
        frame_step = 1.0

    # Initialize frame counter
    current_position = 0.0
    frame_index = 0

    # Process frames with progress bar
    print(f"Processing video: {os.path.basename(input_file)}")
    print(
        f"Original duration: {original_duration:.2f}s, Target duration: {target_duration:.2f}s"
    )
    print(f"Speed factor: {speed_factor:.2f}x, New FPS: {new_fps:.2f}")

    pbar = tqdm(total=frame_count, unit="frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only write frames at the calculated step rate
        if int(current_position) == frame_index:
            out.write(frame)

        current_position += frame_step
        frame_index += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"Acceleration completed! Output saved to: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Accelerate a video to a target duration or by a speed factor"
    )
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("output", help="Output video file path")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--duration",
        type=float,
        help="Target duration in seconds (e.g., 60 for 1 minute)",
    )
    group.add_argument(
        "--speed", type=float, help="Speed factor (e.g., 1.5 for 50%% faster)"
    )

    args = parser.parse_args()

    # Call the acceleration function
    accelerate_video(args.input, args.output, args.duration, args.speed)


if __name__ == "__main__":
    try:
        # Check if running with arguments
        if len(sys.argv) > 1:
            main()
        else:
            # Interactive mode
            print("Video Acceleration Tool")
            print("----------------------")

            input_file = input("Enter input video file path: ").strip()
            output_file = input("Enter output video file path: ").strip()

            acceleration_type = (
                input("Accelerate by [d]uration or [s]peed factor? (d/s): ")
                .strip()
                .lower()
            )

            if acceleration_type == "d":
                target_duration = float(
                    input("Enter target duration in seconds: ").strip()
                )
                accelerate_video(
                    input_file, output_file, target_duration=target_duration
                )
            elif acceleration_type == "s":
                speed_factor = float(
                    input("Enter speed factor (e.g., 1.5 for 50% faster): ").strip()
                )
                accelerate_video(input_file, output_file, speed_factor=speed_factor)
            else:
                print(
                    "Invalid option. Please choose either 'd' for duration or 's' for speed factor."
                )

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
