# Traffic Flow Detection Project

基于 YOLO 与 OpenCV 的智能交通流量检测系统，实现车辆检测、车辆计数、交通流量统计与视频分析等功能。

---

## 项目简介

本项目是一个基于计算机视觉的交通流量检测系统，能够对道路视频中的车辆进行实时检测与统计分析。

系统主要使用：

- YOLO 目标检测模型
- OpenCV 视频处理
- Python 深度学习框架

实现以下功能：

- 车辆目标检测
- 实时车辆计数
- 交通流量统计
- 视频流分析
- 检测结果可视化

---

## 项目功能

### 1. 车辆检测

支持对视频中的车辆进行自动识别，包括：

- 小汽车
- 卡车
- 公交车
- 摩托车

---

### 2. 实时计数

系统能够对经过指定区域的车辆进行实时统计。

功能包括：

- 单方向计数
- 多车辆统计
- 流量变化分析

---

### 3. 视频分析

支持：

- 本地视频文件检测
- 摄像头实时检测
- 检测结果视频保存

---

### 4. 可视化展示

检测结果将以可视化形式展示：

- 检测框
- 车辆类别
- 实时数量
- FPS 信息

---

## 项目结构

```bash
Traffic-Flow-Detection-Project/
│
├── data/                # 数据集
├── model/               # 模型文件
├── output/              # 输出结果
├── videos/              # 测试视频
├── main.py              # 主程序
├── detect.py            # 检测模块
├── requirements.txt     # 依赖库
└── README.md
```

---

## 环境配置

### 1. 克隆项目

```bash
git clone https://github.com/jiantianbujiaban/Traffic-Flow-Detection-Project.git
```

---

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 运行项目

### 使用视频检测

```bash
python main.py
```

---

### 使用摄像头实时检测

修改输入源后运行：

```bash
python main.py
```

---

## 技术栈

- Python
- OpenCV
- YOLO
- NumPy
- Deep Learning

---

## 应用场景

- 智慧城市
- 智能交通系统
- 道路监控
- 车辆流量统计
- AI 视频分析

---

## 后续优化方向

未来可以继续完善：

- 多目标跟踪
- 车速检测
- 车牌识别
- 红绿灯识别
- Web 可视化平台
- GPU 加速部署

---

## License

This project is licensed under the MIT License.

---

## 作者

GitHub: https://github.com/jiantianbujiaban
