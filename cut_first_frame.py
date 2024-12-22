import cv2
import argparse

def extract_first_frame(video_path, output_path):
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Đọc frame đầu tiên
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Cannot read the first frame of video {video_path}")
        cap.release()
        return

    # Lưu frame đầu tiên
    cv2.imwrite(output_path, frame)
    print(f"First frame extracted and saved to: {output_path}")

    # Đóng video
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first frame of a video and save it as an image.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    parser.add_argument("--output", type=str, required=True, help="Path to save the first frame image")

    args = parser.parse_args()

    extract_first_frame(args.video, args.output)
