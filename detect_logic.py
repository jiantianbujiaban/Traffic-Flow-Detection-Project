import os
import uuid
import shutil
import pandas as pd
from collections import Counter
from yolov5_master.video import run

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/runs'
FIXED_OUTPUT_NAME = 'fixed_video_output'
FIXED_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, FIXED_OUTPUT_NAME)

CLASS_NAMES = ['person', 'car', 'bus', 'truck']

def setup_folders():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FIXED_OUTPUT_FOLDER, exist_ok=True)

def create_dataset(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

def lstm_predict(series, n_steps=10):
    scaler = MinMaxScaler(feature_range=(0,1))
    series_scaled = scaler.fit_transform(series.reshape(-1,1))

    X, y = create_dataset(series_scaled, n_steps)
    if len(X) == 0:
        return []

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=30, verbose=0)

    last_seq = series_scaled[-n_steps:]
    n_forecast = len(series_scaled)

    preds = []
    input_seq = last_seq.copy()
    for _ in range(n_forecast):
        pred = model.predict(input_seq[np.newaxis,:,:])[0,0]
        preds.append(pred)
        input_seq = np.append(input_seq[1:], pred)
        input_seq = input_seq.reshape(-1,1)

    forecast = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    return forecast.tolist()

def handle_detection(file_storage):
    filename = str(uuid.uuid4()) + os.path.splitext(file_storage.filename)[1]
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file_storage.save(input_path)

    if os.path.exists(FIXED_OUTPUT_FOLDER):
        shutil.rmtree(FIXED_OUTPUT_FOLDER)
    os.makedirs(FIXED_OUTPUT_FOLDER, exist_ok=True)

    run(
        weights='weights/Best_Accuracy_Dhaka_AI_Yolov5l_By_Autobot_BS_8.pt',
        source=input_path,
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        project=OUTPUT_FOLDER,
        name=FIXED_OUTPUT_NAME,
        exist_ok=True,
        save_csv=True,
        save_txt=True,
        nosave=False,
        view_img=False,
    )

    ext = filename.lower().split('.')[-1]

    count_excel_path = os.path.join(FIXED_OUTPUT_FOLDER, 'detection_counts.xlsx')
    counts_data = None
    chart_labels = []
    chart_values = []
    lstm_forecast = []

    if os.path.exists(count_excel_path):
        df_counts = pd.read_excel(count_excel_path)

        # 提取第一列字符串中最后一个'_'后的数字，作为排序键
        def extract_num(val):
            try:
                return int(str(val).rsplit('_', 1)[-1])
            except:
                return -1  # 出错时放前面

        df_counts['sort_num'] = df_counts.iloc[:,0].apply(extract_num)
        df_counts = df_counts.sort_values('sort_num').reset_index(drop=True)
        df_counts = df_counts.drop(columns=['sort_num'])

        # 汇总统计所有帧每类物体总数（去除第一列）
        counts_data = df_counts.drop(columns=[df_counts.columns[0]], errors='ignore').sum().to_dict()

        if df_counts.shape[1] >= 2:
            chart_labels = df_counts.iloc[:, 0].tolist()
            if 'car' in df_counts.columns:
                car_series = df_counts['car'].fillna(0).values.astype(float)
                chart_values = car_series.tolist()
                lstm_forecast = lstm_predict(car_series)
            else:
                chart_values = df_counts.iloc[:, 1].tolist()

    return {
        'filename': filename,
        'input_path': input_path,
        'ext': ext,
        'counts_data': counts_data,
        'chart_labels': chart_labels,
        'chart_values': chart_values,
        'lstm_forecast': lstm_forecast,
        'fixed_output_folder': FIXED_OUTPUT_FOLDER
    }

def get_image_counts(filename):
    label_path = os.path.join(FIXED_OUTPUT_FOLDER, 'labels', filename.rsplit('.', 1)[0] + '.txt')
    counts = Counter()
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                if 0 <= class_id < len(CLASS_NAMES):
                    counts[CLASS_NAMES[class_id]] += 1
    return dict(counts)

def convert_video_format(detected_video):
    converted_video = os.path.join(FIXED_OUTPUT_FOLDER, 'converted_' + os.path.basename(detected_video))
    if not os.path.exists(converted_video):
        import subprocess
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', detected_video,
                '-vcodec', 'libx264',
                '-acodec', 'aac',
                converted_video
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"视频转换失败: {e}")
            return detected_video
    return converted_video
