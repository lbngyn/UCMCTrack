import cv2
import pandas as pd
import random

def get_resolution(input_path): 
    if '01' in input_path: 
        return (1920,1080)
    if '02' in input_path: 
        return (1920,1080)
    if '03' in input_path: 
        return (1920,1080)
    if '04' in input_path: 
        return (1920,1080)
    if '05' in input_path: 
        return (640,480)
    if '06' in input_path: 
        return (640,480)
    if '07' in input_path: 
        return (1920,1080)
    if '08' in input_path: 
        return (1920,1080)
    if '09' in input_path: 
        return (1920,1080)
    if '10' in input_path: 
        return (1920,1080)
    if '11' in input_path: 
        return (1920,1080)
    if '12' in input_path: 
        return (1920,1080)
    if '13' in input_path: 
        return (1920,1080)
    if '14' in input_path: 
        return (1920,1080)

def generate_colors(num_colors):
    """Sinh màu ngẫu nhiên cho các object_id"""
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]

def draw_bounding_boxes(input_txt, input_video, output_video):
    # Đọc dữ liệu từ file TXT
    data = pd.read_csv(input_txt, header=None)
    data.columns = [
        'frame_id', 'object_id', 'x', 'y', 'width', 'height', 
        'confidence', 'class_id', 'visibility_ratio', 'something'
    ]

    # Lấy danh sách object_id duy nhất
    unique_objects = data['object_id'].unique()
    colors = generate_colors(len(unique_objects))  # Sinh màu cho từng object_id
    object_colors = {obj_id: colors[i] for i, obj_id in enumerate(unique_objects)}

    # Mở video đầu vào
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Không thể mở video đầu vào!")
        return

    # Thông tin video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = get_resolution(input_video)
    w_scale = resolution[0] / frame_width
    h_scale = resolution[1] / frame_height

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tạo video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec cho video đầu ra
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    print("Đang xử lý video...")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Kết thúc video

        # Lấy các bounding box tương ứng với frame hiện tại
        frame_data = data[data['frame_id'] == frame_idx + 1]
        for _, row in frame_data.iterrows():
            x, y, w, h = int(row['x']//w_scale), int(row['y']//h_scale), int(row['width']//w_scale), int(row['height']//h_scale)
            confidence = row['confidence']
            object_id = row['object_id']

            # Lấy màu sắc cho object_id
            color = object_colors[object_id]

            # Vẽ bounding box
            cv2.rectangle(
                frame, 
                (x,y), 
                (x+w, y+h), 
                color, 2
            )

            # Thêm thông tin Confidence và Object ID
            label = f"ID: {object_id}"
            cv2.putText(
                frame, label, (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        # Ghi frame đã xử lý vào video đầu ra
        out.write(frame)
        frame_idx += 1

        # Hiển thị tiến trình
        print(f"Đã xử lý frame {frame_idx}/{total_frames}", end="\r")

    # Đóng video
    cap.release()
    out.release()
    print("\nHoàn tất! Video đầu ra đã được lưu.")

# Thay đổi đường dẫn file nếu cần
input_txt = r"det_results\mot17\yolov10\MOT17-02-SDP.txt"  # File chứa dữ liệu bounding box (TXT)
input_video = r"videos\MOT17-02-FRCNN-raw.mp4"   # Video gốc
output_video = r"check_video-02.mp4"  # Video đầu ra

draw_bounding_boxes(input_txt, input_video, output_video)
