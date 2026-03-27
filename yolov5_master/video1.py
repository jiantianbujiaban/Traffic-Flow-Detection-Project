import argparse
import csv
import os
import sys
import time
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from sort import Sort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


# 日志输出函数
def log(msg):
    print(f"[LOG] {msg}")


def detect(opt):
    source = opt.source
    weights = opt.weights
    imgsz = opt.imgsz
    conf_thres = opt.conf_thres
    iou_thres = opt.iou_thres
    device = select_device(opt.device)

    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)
    video_save_path = os.path.join(save_dir, 'output.mp4')
    csv_save_path = os.path.join(save_dir, 'results.csv')
    excel_save_path = os.path.join(save_dir, 'results.xlsx')

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

    save_csv = opt.save_csv
    view_img = opt.view_img

    tracker = Sort()

    last_positions = {}
    last_times = {}
    static_start_times = {}  # 静止时间记录

    if save_csv:
        csv_file = open(csv_save_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Frame", "ID", "Speed_kmh"])

    frame_id = 0
    while True:
        success, frame = cap.read()
        if not success:
            log("Video ended or failed to read frame.")
            break

        frame_id += 1
        log(f"Processing frame {frame_id}")

        img = cv2.resize(frame, tuple(imgsz))
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

        det = []
        if pred is not None and len(pred):
            log(f"Detections: {len(pred)}")
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
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if track_id in last_positions:
                dx = cx - last_positions[track_id][0]
                dy = cy - last_positions[track_id][1]
                dt = current_time - last_times[track_id]
                speed = (np.sqrt(dx ** 2 + dy ** 2) / dt) * 3.6 if dt > 0 else 0

                # 判断是否静止
                if speed < 1:
                    if track_id not in static_start_times:
                        static_start_times[track_id] = current_time
                    elif current_time - static_start_times[track_id] > 0.5:
                        log(f"Track {track_id} static > 0.5s, skipped drawing and speed output.")
                        continue  # 跳过该目标
                else:
                    static_start_times.pop(track_id, None)

                # 正常处理
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

        if view_img:
            cv2.imshow('result', frame)
            if cv2.waitKey(1) == ord('q'):
                log("User pressed 'q', exiting.")
                break

        video_writer.write(frame)

    cap.release()
    video_writer.release()
    if save_csv:
        csv_file.close()
        log(f"CSV file saved as {csv_save_path}")

        # 转成 Excel 文件并删除 CSV
        df = pd.read_csv(csv_save_path)
        df.to_excel(excel_save_path, index=False)
        os.remove(csv_save_path)
        log(f"Excel file saved as {excel_save_path}")

    cv2.destroyAllWindows()
    log(f"Video saved as {video_save_path}")
    log("Program finished.")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='model path')
    parser.add_argument('--source', type=str, default='aaa.mp4', help='video path')
    parser.add_argument('--imgsz', nargs=2, type=int, default=[640, 640], help='image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-csv', action='store_true', help='save results to CSV and Excel')
    parser.add_argument('--save-dir', type=str, default='static/runs/fixed_video_output', help='directory to save results')
    return parser.parse_args()


def main():
    opt = parse_opt()
    log(f"Using weights: {opt.weights}")
    log(f"Video source: {opt.source}")
    log(f"Save directory: {opt.save_dir}")
    detect(opt)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += [
            '--weights', 'best.pt',
            '--source', 'aaa.mp4',
            '--view-img',
            '--save-csv',
            '--save-dir', 'static/runs/fixed_video_output'
        ]
    main()
