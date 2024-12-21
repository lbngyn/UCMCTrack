import cv2
import torch
from torchvision.transforms import functional as F
import os
import argparse
import numpy as np
from ultralytics import YOLO
import os,cv2
from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper

# 定义一个Detection类，包含id,bb_left,bb_top,bb_width,bb_height,conf,det_class
class Detection:

    def __init__(self, id, bb_left = 0, bb_top = 0, bb_width = 0, bb_height = 0, conf = 0, det_class = 0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)


    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2,self.bb_top+self.bb_height,self.y[0,0],self.y[1,0])

    def __repr__(self):
        return self.__str__()


# Detector类，用于从Yolo检测器获取目标检测的结果
class Detector:
    def __init__(self, model_path = 'pretrained/yolov10x.pt'):
        self.seq_length = 0
        self.gmc = None
        self.model_path = model_path

    def load(self,cam_para_file):
        self.mapper = Mapper(cam_para_file,"MOT17")
        self.model = YOLO(self.model_path)

    def get_dets(self, img,conf_thresh = 0.5,det_classes = [0]):
        
        dets = []

        # 将帧从 BGR 转换为 RGB（因为 OpenCV 使用 BGR 格式）  
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        # 使用 RTDETR 进行推理  
        results = self.model(frame,imgsz = 1088)

        det_id = 0
        for box in results[0].boxes:
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id  = box.cls.cpu().numpy()[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
                continue

            # 新建一个Detection对象
            det = Detection(det_id)
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.y,det.R = self.mapper.mapto([det.bb_left,det.bb_top,det.bb_width,det.bb_height])
            det_id += 1

            dets.append(det)

        return dets

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

def process_video(input_path, output_path, model_path, conf_thresh, cam_para_dir):
    # Mở video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        return
    
    # Tạo file kết quả
    filename = os.path.basename(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = get_resolution(input_path)
    w_scale = resolution[0]/frame_width
    h_scale = resolution[1]/frame_height

    # Loại bỏ phần mở rộng .mp4
    name_without_extension = os.path.splitext(filename)[0]
    print(name_without_extension)
    os.makedirs(output_path, exist_ok=True)
    result_path = os.path.join(output_path, name_without_extension + '-SDP.txt')
    cam_para_path = os.path.join(cam_para_dir, name_without_extension + '-SDP.txt')
    print(result_path)

    detector = Detector(model_path)
    detector.load(cam_para_path)

    result_file = open(result_path, 'w')
    frame_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(frame.shape) 
        print(type(frame))
        # Lưu kích thước ảnh gốc

        # Dự đoán bounding boxes và scale về kích thước gốc
        dets = detector.get_dets(frame,conf_thresh)
        
        # Ghi dữ liệu vào file theo format
        for det in dets:
            detid = det.id 
            x = det.bb_left 
            y = det.bb_top  
            w = det.bb_width  
            h = det.bb_height  
            conf = det.conf 
            cls = det.det_class 

            result_file.write(f"{frame_id},{detid},{(x*w_scale):.1f},{(y*h_scale):.1f},{(w*w_scale):.1f},{(h*h_scale):.1f},{conf:.2f},-1,-1,-1\n")
     
        frame_id += 1

    # Đóng file và giải phóng tài nguyên
    result_file.close()
    cap.release()
    print(f"Detection results saved to {result_path}")


def process_videos_in_folder(input_folder, output_path, model_path, conf_thresh, cam_para_dir):
    # Duyệt qua tất cả các file trong thư mục input
    for filename in os.listdir(input_folder):
        if filename.endswith(('.mp4', '.avi', '.mov')):  # Chỉ xử lý các video
            input_video_path = os.path.join(input_folder, filename)
            print(f"Processing {input_video_path}...")
            process_video(input_video_path, output_path, model_path, conf_thresh, cam_para_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv10x on video frames and save detections.")
    parser.add_argument('--input', type=str, required=True, help="Path to folder containing videos.")
    parser.add_argument('--output', type=str, required=True, help="Path to save detection results (txt files).")
    parser.add_argument('--model', type=str, required=True, help="Path to YOLOv10x model weights.")
    parser.add_argument('--conf_thresh', required=False, type=float, default=0.01, help="Confidence threshold for detections.")
    parser.add_argument('--cam_para_dir', required=False, type=str, default = "cam_para/MOT17", help='camera parameter file name')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_videos_in_folder(args.input, args.output, args.model, args.conf_thresh, args.cam_para_dir)
