import cv2
import os
import sys
import subprocess
import math
import shutil
from datetime import datetime


def get_video_size(file_path):
    """Get the file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return None


def get_video_duration(file_path):
    """Get video duration in seconds using OpenCV"""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    return duration


def format_size(size_bytes):
    """Format size in bytes to human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.2f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.2f} GB"


def parse_size(size_str):
    """Parse size string like '10MB' to bytes"""
    size_str = size_str.strip().upper()

    # Extract number and unit
    if size_str.endswith("KB"):
        multiplier = 1024
        number = float(size_str[:-2])
    elif size_str.endswith("MB"):
        multiplier = 1024 * 1024
        number = float(size_str[:-2])
    elif size_str.endswith("GB"):
        multiplier = 1024 * 1024 * 1024
        number = float(size_str[:-2])
    elif size_str.endswith("B"):
        multiplier = 1
        number = float(size_str[:-1])
    else:
        try:
            # Assume it's just a number in bytes
            return int(size_str)
        except ValueError:
            return None

    return int(number * multiplier)


def reduce_video_quality(input_file, output_file, target_size, max_attempts=10):
    """
    Reduce video quality to meet the target file size

    Uses FFmpeg to encode the video with decreasing quality until the target size is met
    """
    # Check if FFmpeg is available
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        print(
            "Error: FFmpeg is not installed or not in the PATH. Please install FFmpeg."
        )
        return False

    original_size = get_video_size(input_file)
    if original_size is None:
        print(f"Error: Could not get size of {input_file}")
        return False

    if original_size <= target_size:
        print(f"Video already smaller than target size. Copying file...")
        shutil.copy2(input_file, output_file)
        return True

    # Get video duration
    duration = get_video_duration(input_file)
    if duration is None:
        print(f"Error: Could not get duration of {input_file}")
        return False

    # Calculate target bitrate
    # Leave some margin for container overhead (5%)
    margin = 0.95
    target_bitrate_total = (target_size * 8 * margin) / duration

    # Allocate 85% to video, 15% to audio
    video_bitrate = int(target_bitrate_total * 0.85)
    audio_bitrate = int(target_bitrate_total * 0.15)

    # Ensure minimum bitrates
    video_bitrate = max(video_bitrate, 100000)  # At least 100 kbps for video
    audio_bitrate = max(audio_bitrate, 32000)  # At least 32 kbps for audio

    print(f"Original size: {format_size(original_size)}")
    print(f"Target size: {format_size(target_size)}")
    print(f"Video duration: {duration:.2f} seconds")
    print(f"Calculated video bitrate: {video_bitrate/1000:.2f} kbps")
    print(f"Calculated audio bitrate: {audio_bitrate/1000:.2f} kbps")

    # Start with a relatively high CRF (lower quality)
    min_crf = 18  # Higher quality bound
    max_crf = 35  # Lower quality bound

    # Binary search to find appropriate quality
    attempt = 1
    best_result = None

    while attempt <= max_attempts:
        # Use middle CRF value
        current_crf = (min_crf + max_crf) / 2
        temp_output = f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"

        print(f"\nAttempt {attempt}/{max_attempts} with CRF={current_crf:.1f}")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-c:v",
            "libx264",
            "-crf",
            f"{current_crf:.1f}",
            "-preset",
            "medium",
            "-c:a",
            "aac",
            "-b:a",
            f"{audio_bitrate}",
            "-movflags",
            "+faststart",
            temp_output,
        ]

        try:
            subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )
            result_size = get_video_size(temp_output)

            if result_size is None:
                print(f"Error: Failed to get size of output file")
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                attempt += 1
                continue

            print(
                f"Result size: {format_size(result_size)} ({result_size/target_size*100:.1f}% of target)"
            )

            # Keep track of the best result so far
            if result_size <= target_size and (
                best_result is None or result_size > best_result[0]
            ):
                if best_result and os.path.exists(best_result[1]):
                    os.remove(best_result[1])
                best_result = (result_size, temp_output, current_crf)
            elif best_result is None and attempt == max_attempts:
                # Last attempt and still no valid result, keep this one
                best_result = (result_size, temp_output, current_crf)
            else:
                # Remove this temp file
                if os.path.exists(temp_output):
                    os.remove(temp_output)

            # Adjust CRF based on result
            if abs(result_size - target_size) / target_size < 0.05:
                # Within 5% of target, good enough
                print(f"Result within 5% of target size, stopping.")
                if best_result is None:
                    best_result = (result_size, temp_output, current_crf)
                break

            if result_size > target_size:
                # Too big, increase CRF (reduce quality)
                min_crf = current_crf
            else:
                # Too small, decrease CRF (increase quality)
                max_crf = current_crf

            # If we're converging
            if max_crf - min_crf < 0.5:
                print("Converged on best CRF value.")
                break

        except subprocess.SubprocessError as e:
            print(f"Error during encoding: {e}")
            if os.path.exists(temp_output):
                os.remove(temp_output)

        attempt += 1

    # Use the best result we found
    if best_result:
        result_size, temp_file, used_crf = best_result
        print(f"\nBest result: {format_size(result_size)} with CRF={used_crf:.1f}")
        print(f"Target size: {format_size(target_size)}")
        print(
            f"Difference: {(result_size-target_size)/target_size*100:.1f}% from target"
        )

        # Rename to final output
        try:
            shutil.move(temp_file, output_file)
            return True
        except OSError as e:
            print(f"Error saving final file: {e}")
            return False
    else:
        print("Failed to meet target size requirements.")
        return False


def main():
    print("Video Size Reducer")
    print("------------------")

    # Get input video file
    while True:
        input_file = input("Enter path to video file: ").strip()
        if not input_file:
            continue

        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            continue

        # Check if it's a valid video file
        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            print(f"Could not open as a video file: {input_file}")
            cap.release()
            continue

        cap.release()
        break

    # Get current file size
    current_size = get_video_size(input_file)
    print(f"Current file size: {format_size(current_size)}")

    # Get target size
    while True:
        target_size_str = input("Enter target size (e.g., 10MB): ").strip()
        if not target_size_str:
            continue

        target_size = parse_size(target_size_str)
        if target_size is None:
            print("Invalid size format. Use formats like 500KB, 10MB, 1GB, or bytes.")
            continue

        if target_size >= current_size:
            print("Target size is larger than current size. No reduction needed.")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != "y":
                continue

        break

    # Get output file
    output_file = input("Enter output file path (e.g., output.mp4): ").strip()
    if not output_file:
        # Generate default output name
        base_name, ext = os.path.splitext(input_file)
        output_file = f"{base_name}_reduced{ext}"

    # Ask for confirmation
    print("\nReady to reduce video:")
    print(f"Input: {input_file} ({format_size(current_size)})")
    print(f"Target size: {format_size(target_size)}")
    print(f"Output: {output_file}")

    confirm = input("\nStart reduction? (y/n): ").strip().lower()
    if confirm != "y":
        print("Operation cancelled.")
        return

    # Start reduction
    print("\nReducing video quality to meet target size...")
    success = reduce_video_quality(input_file, output_file, target_size)

    if success:
        final_size = get_video_size(output_file)
        print(f"\nVideo successfully reduced!")
        print(f"Final size: {format_size(final_size)}")
        print(f"Target size: {format_size(target_size)}")
        print(
            f"Difference: {(final_size-target_size)/target_size*100:.1f}% from target"
        )
    else:
        print("\nVideo reduction failed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)
