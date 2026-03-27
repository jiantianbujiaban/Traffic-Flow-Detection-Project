# yolov5_master/video1_detect.py

import sys
from pathlib import Path

# 添加 yolov5_master 到 sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5_master
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import csv
import os
import time
import cv2
import torch
import numpy as np
import pandas as pd

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from sort import Sort

# ... 后续逻辑不变


def log(msg):
    print(f"[LOG] {msg}")

def detect(
    weights,
    source,
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    device='',
    view_img=False,
    save_csv=True,
    save_dir='static/runs/fixed_video_output'
):
    os.makedirs(save_dir, exist_ok=True)
    video_save_path = os.path.join(save_dir, 'output.mp4')
    csv_save_path = os.path.join(save_dir, 'results.csv')
    excel_save_path = os.path.join(save_dir, 'results.xlsx')

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    model.warmup(imgsz=(1, 3, *imgsz))
    names = model.names

    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), f"Failed to open {source}"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    log(f"Video {source} opened, FPS: {fps}, Resolution: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

    tracker = Sort()
    last_positions, last_times, static_start_times = {}, {}, {}

    if save_csv:
        csv_file = open(csv_save_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Frame", "ID", "Speed_kmh"])

    frame_id = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_id += 1
        img = cv2.resize(frame, tuple(imgsz))
        img_tensor = torch.from_numpy(img).to(device).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

        det = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes(img_tensor.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in pred:
                x1, y1, x2, y2 = map(int, xyxy)
                if x2 <= x1 or y2 <= y1:
                    continue
                det.append([x1, y1, x2, y2, float(conf)])

        tracks = tracker.update(np.array(det))
        current_time = time.time()

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if track_id in last_positions:
                dx = cx - last_positions[track_id][0]
                dy = cy - last_positions[track_id][1]
                dt = current_time - last_times[track_id]
                speed = (np.sqrt(dx ** 2 + dy ** 2) / dt) * 3.6 if dt > 0 else 0

                if speed < 1:
                    if track_id not in static_start_times:
                        static_start_times[track_id] = current_time
                    elif current_time - static_start_times[track_id] > 0.5:
                        continue
                else:
                    static_start_times.pop(track_id, None)

                if save_csv:
                    csv_writer.writerow([frame_id, track_id, round(speed, 2)])
                cv2.putText(frame, f'ID:{track_id} Speed:{round(speed, 1)} km/h', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            last_positions[track_id] = (cx, cy)
            last_times[track_id] = current_time
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        video_writer.write(frame)

    cap.release()
    video_writer.release()
    if save_csv:
        csv_file.close()
        df = pd.read_csv(csv_save_path)
        df.to_excel(excel_save_path, index=False)
        os.remove(csv_save_path)

    cv2.destroyAllWindows()
    return video_save_path, excel_save_path
