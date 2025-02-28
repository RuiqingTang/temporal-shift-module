#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   websocket_api.py
@Time    :   2025/02/24 16:02:46
@Author  :   Ruiqing Tang 
@Contact :   tangruiqing123@gmail.com
'''

import os
import time
import numpy as np
import cv2
from typing import Tuple
from PIL import Image
import onnx
import onnxruntime as ort
import websocket
import json

def resource_path(relative_path):
    """获取资源文件的绝对路径。
    当在开发环境中时，返回当前目录的路径；
    打包后返回临时目录的路径（sys._MEIPASS）。
    """
    try:
        base_path = sys._MEIPASS  # PyInstaller 打包后临时文件夹
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 配置参数
WEBSOCKET_ENABLED = True  # 是否启用 WebSocket
WEBSOCKET_HOST = "localhost"  # WebSocket 服务器地址
WEBSOCKET_PORT = 8888         # WebSocket 端口
WEBSOCKET_URL = f"ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}"

# 图像预处理相关定义
# =============================================================================
def resize_shorter(image, size):
    """
    按照较短边将图像缩放到指定尺寸，同时保持长宽比。
    image: PIL.Image 对象
    size: 较短边目标尺寸
    """
    w, h = image.size
    if w < h:
        new_w = size
        new_h = int(size * h / w)
    else:
        new_h = size
        new_w = int(size * w / h)
    return image.resize((new_w, new_h), Image.BILINEAR)

def center_crop(image, crop_size):
    """
    对 PIL.Image 进行中心裁剪，裁剪区域大小为 (crop_size, crop_size)
    """
    w, h = image.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))

def transform_image(image):
    """
    对输入的 PIL.Image（RGB）执行预处理：
      1. 将短边缩放到 256；
      2. 中心裁剪 224×224；
      3. 转换为 NumPy 数组并归一化到 [0, 1]；
      4. 调整通道顺序（HWC -> CHW）；
      5. 归一化（减去均值除以标准差）；
      6. 增加 batch 维度，得到形状 (1, 3, 224, 224) 的数组。
    注意：所有数值均确保为 np.float32 类型。
    """
    # 1. 缩放和中心裁剪
    image = resize_shorter(image, 256)
    image = center_crop(image, 224)
    # 2. 转换为 NumPy 数组，并归一化到 [0, 1]
    np_img = np.array(image).astype(np.float32) / np.float32(255.0)
    # 3. 调整通道顺序 (H, W, C) -> (C, H, W)
    np_img = np.transpose(np_img, (2, 0, 1))
    # 4. 均值和标准差归一化，确保均值和 std 为 float32
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    np_img = (np_img - mean) / std
    # 5. 增加 batch 维度 -> (1, 3, 224, 224)
    np_img = np.expand_dims(np_img, axis=0)
    return np_img

def get_transform():
    """
    返回图像预处理函数
    """
    return transform_image

catigories = [
    "Doing other things",      # 0
    "Drumming Fingers",        # 1
    "No gesture",              # 2
    "Pulling Hand In",         # 3
    "Pulling Two Fingers In",  # 4
    "Pushing Hand Away",       # 5
    "Pushing Two Fingers Away",# 6
    "Rolling Hand Backward",   # 7
    "Rolling Hand Forward",    # 8
    "Shaking Hand",            # 9
    "Sliding Two Fingers Down",# 10
    "Sliding Two Fingers Left",# 11
    "Sliding Two Fingers Right",# 12
    "Sliding Two Fingers Up",   # 13
    "Stop Sign",                # 14
    "Swiping Down",             # 15
    "Swiping Left",             # 16
    "Swiping Right",            # 17
    "Swiping Up",               # 18
    "Thumb Down",               # 19
    "Thumb Up",                 # 20
    "Turning Hand Clockwise",   # 21
    "Turning Hand Counterclockwise",# 22
    "Zooming In With Full Hand",    # 23
    "Zooming In With Two Fingers",   # 24
    "Zooming Out With Full Hand",    # 25
    "Zooming Out With Two Fingers"    # 26
]

SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True

def process_output(idx_, history):
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # 历史缓冲区最大长度

    # 屏蔽不合法的动作（若当前预测属于非法类别，则保持上一帧的预测）
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # 将类别 0 转换为 “No gesture”（类别 2）
    if idx_ == 0:
        idx_ = 2

    # 历史平滑
    if len(history) > 1 and idx_ != history[-1]:
        if not (history[-1] == history[-2]):
            idx_ = history[-1]

    history.append(idx_)
    history = history[-max_hist_len:]
    return history[-1], history

# =============================================================================
# 模型转换相关：将 PyTorch 模型转换为 ONNX 格式
# =============================================================================

def get_onnx_session(onnx_path="mobilenet_v2_tsm.onnx"):
    onnx_path = resource_path(onnx_path)  # 修改：获取资源的正确路径
    if not os.path.exists(onnx_path):
        print("错误：找不到 ONNX 模型文件：", onnx_path)
        exit(1)
    session = ort.InferenceSession(onnx_path)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print("ONNX 模型加载成功！")
    return session, input_names, output_names

WINDOW_NAME = 'Video Gesture Recognition'


def init_websocket():
    try:
        ws = websocket.WebSocket()
        ws.connect(WEBSOCKET_URL)
        print("WebSocket 连接成功！")
        return ws
    except Exception as e:
        print(f"WebSocket 连接失败: {e}")
        return None

def main():
    print("打开摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    # 设置较低分辨率以提高处理速度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    print("构建预处理器...")
    transform = get_transform()

    print("加载 ONNX 模型...")
    session, input_names, output_names = get_onnx_session()

    # 初始化 buffer（状态变量），均为全 0 数组，与模型转换时的形状保持一致
    buffer = [
        np.zeros((1, 3, 56, 56), dtype=np.float32),
        np.zeros((1, 4, 28, 28), dtype=np.float32),
        np.zeros((1, 4, 28, 28), dtype=np.float32),
        np.zeros((1, 8, 14, 14), dtype=np.float32),
        np.zeros((1, 8, 14, 14), dtype=np.float32),
        np.zeros((1, 8, 14, 14), dtype=np.float32),
        np.zeros((1, 12, 14, 14), dtype=np.float32),
        np.zeros((1, 12, 14, 14), dtype=np.float32),
        np.zeros((1, 20, 7, 7), dtype=np.float32),
        np.zeros((1, 20, 7, 7), dtype=np.float32)
    ]
    idx = 0
    history = [2]
    history_logit = []
    i_frame = -1

    current_time = 0.1  # 防止除 0
    last_ws_send_time = 0
    min_send_interval = 1.0 / 70  # 每秒最多发送70次，即最小间隔约14.3毫秒

    ws = None
    if WEBSOCKET_ENABLED:
        ws = init_websocket()

    print("准备就绪！")
    while True:
        i_frame += 1
        ret, img = cap.read()
        if not ret:
            break

        # 为了降低帧率，每隔一帧处理一次
        if i_frame % 2 == 0:
            t1 = time.time()
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # 预处理，得到形状 (1, 3, 224, 224) 的 np.float32 数组
            input_np = transform(img_pil)
            
            # 构造 onnxruntime 输入字典
            # 第一个输入为图像，后续 10 个为 buffer 状态
            feed_dict = {}
            feed_dict[input_names[0]] = input_np
            for i, buf in enumerate(buffer):
                feed_dict[input_names[i+1]] = buf

            # 运行推理，获得多个输出
            outputs = session.run(None, feed_dict)
            # 第一个输出为分类 logits，其余输出为更新后的 buffer 状态
            feat = outputs[0]
            buffer = outputs[1:]
            
            # 后处理：使用 softmax（可选）和历史平滑
            if SOFTMAX_THRES > 0:
                feat_flat = feat.reshape(-1)
                feat_flat = feat_flat - np.max(feat_flat)
                softmax = np.exp(feat_flat) / np.sum(np.exp(feat_flat))
                if np.max(softmax) > SOFTMAX_THRES:
                    idx_ = np.argmax(feat, axis=1)[0]
                else:
                    idx_ = idx
            else:
                idx_ = np.argmax(feat, axis=1)[0]

            if HISTORY_LOGIT:
                history_logit.append(feat)
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = process_output(idx_, history)
            t2 = time.time()
            current_time = t2 - t1
            print(f"帧 {i_frame}: {catigories[idx]}  ({1/current_time:.1f} Vid/s)")

            # 如果启用了 WebSocket，发送识别结果
            if WEBSOCKET_ENABLED and ws:
                try:
                    # 检查是否可以发送（是否已经过了最小间隔时间）
                    current_time_ms = time.time()
                    if current_time_ms - last_ws_send_time >= min_send_interval:
                        message = {
                            "type": "gesture",
                            "timestamp": current_time_ms,
                            "category": catigories[idx],
                            "fps": 1 / current_time if current_time > 0 else 0
                        }
                        ws.send(json.dumps(message))
                        last_ws_send_time = current_time_ms  # 更新最后发送时间
                except Exception as e:
                    print(f"WebSocket 发送失败: {e}")

            print(f"帧 {i_frame}: {catigories[idx]}  ({1/current_time:.1f} Vid/s)")

        # 在显示图像上叠加预测结果和帧率
        disp_img = cv2.resize(img, (640, 480))
        # 水平翻转，获得镜像效果
        disp_img = disp_img[:, ::-1]
        h, w, _ = disp_img.shape
        label = np.full((h // 10, w, 3), 255, dtype=np.uint8)
        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(h / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1/current_time if current_time > 0 else 0),
                    (w - 170, int(h / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        disp_img = np.concatenate((disp_img, label), axis=0)
        cv2.imshow(WINDOW_NAME, disp_img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # 按 q 或 ESC 退出
            break
        elif key in [ord('F'), ord('f')]:
            full_screen = not full_screen
            if full_screen:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

