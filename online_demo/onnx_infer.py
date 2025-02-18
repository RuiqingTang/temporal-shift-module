#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   my_main.py
@Time    :   2025/02/05 14:32:10
@Author  :   Ruiqing Tang 
@Contact :   tangruiqing123@gmail.com
'''

import os
import time
import numpy as np
import cv2
from typing import Tuple
from PIL import Image, ImageOps
import torch
import torchvision
import onnx
import onnxruntime as ort

# 导入模型（请确保 mobilenet_v2_tsm.py 在当前目录或 Python 路径中）
from mobilenet_v2_tsm import MobileNetV2

# 配置参数
SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True

# =============================================================================
# 图像预处理相关定义
# =============================================================================
class GroupScale(object):
    """将输入的 PIL.Image 按照短边缩放到给定大小。"""
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupCenterCrop(object):
    """中心裁剪"""
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class Stack(object):
    """将图片列表按通道拼接"""
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate([np.array(x) for x in img_group], axis=2)

class ToTorchFormatTensor(object):
    """
    将 PIL.Image (RGB) 或 numpy.ndarray (H x W x C) 转换为 torch.FloatTensor，
    并将通道维度置于首位，同时将数值归一化到 [0, 1]。
    """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy 数组
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class GroupNormalize(object):
    """对 tensor 进行归一化"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)
        return tensor

def get_transform():
    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
    ])
    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

# =============================================================================
# 手势类别
# =============================================================================
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

# =============================================================================
# 输出后处理函数，用于平滑预测
# =============================================================================
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
def convert_model_to_onnx(model, onnx_path="mobilenet_v2_tsm.onnx"):
    """
    使用一组 dummy inputs 完成模型导出。模型包含 11 个输入：
      - 第 1 个为图像输入，形状 (1, 3, 224, 224)
      - 后续 10 个为 buffer 状态，形状分别为：
            (1, 3, 56, 56),
            (1, 4, 28, 28),
            (1, 4, 28, 28),
            (1, 8, 14, 14),
            (1, 8, 14, 14),
            (1, 8, 14, 14),
            (1, 12, 14, 14),
            (1, 12, 14, 14),
            (1, 20, 7, 7),
            (1, 20, 7, 7)
    """
    model.eval()
    dummy_inputs = (
        torch.randn(1, 3, 224, 224, requires_grad=False),
        torch.zeros(1, 3, 56, 56),
        torch.zeros(1, 4, 28, 28),
        torch.zeros(1, 4, 28, 28),
        torch.zeros(1, 8, 14, 14),
        torch.zeros(1, 8, 14, 14),
        torch.zeros(1, 8, 14, 14),
        torch.zeros(1, 12, 14, 14),
        torch.zeros(1, 12, 14, 14),
        torch.zeros(1, 20, 7, 7),
        torch.zeros(1, 20, 7, 7)
    )
    input_names = ["input{}".format(i) for i in range(len(dummy_inputs))]
    # 假设模型输出 11 个张量：第 1 个为分类 logits，其余为更新后的 buffer 状态
    output_names = ["output{}".format(i) for i in range(11)]
    print("导出 ONNX 模型到文件：", onnx_path)
    torch.onnx.export(model, dummy_inputs, onnx_path,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=11)
    print("转换完成！")

def get_onnx_session(onnx_path="mobilenet_v2_tsm.onnx"):
    """
    若 ONNX 文件不存在，则先加载 PyTorch 模型并转换；
    否则直接加载 ONNX 模型，并返回 onnxruntime.InferenceSession 及输入/输出名称列表。
    """
    if not os.path.exists(onnx_path):
        print("ONNX 文件不存在，开始转换...")
        model = MobileNetV2(n_class=27)
        if not os.path.exists("mobilenetv2_jester_online.pth.tar"):
            print('Downloading PyTorch checkpoint...')
            import urllib.request
            url = 'https://hanlab18.mit.edu/projects/tsm/models/mobilenetv2_jester_online.pth.tar'
            urllib.request.urlretrieve(url, './mobilenetv2_jester_online.pth.tar')
        checkpoint = torch.load("mobilenetv2_jester_online.pth.tar", map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        convert_model_to_onnx(model, onnx_path)
    session = ort.InferenceSession(onnx_path)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print("ONNX 模型加载成功！")
    return session, input_names, output_names

# =============================================================================
# 主函数：使用 OpenCV 采集摄像头视频，并使用 onnxruntime 进行推理
# =============================================================================
WINDOW_NAME = 'Video Gesture Recognition'
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

    print("准备就绪！")
    while True:
        i_frame += 1
        ret, img = cap.read()
        if not ret:
            break

        # 为了降低帧率，每隔一帧处理一次
        if i_frame % 2 == 0:
            t1 = time.time()
            # 预处理：将 BGR 格式图像转换为 PIL Image，并转换为 RGB
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # transform 接受一个列表，返回一个 tensor，形状为 (3, 224, 224)
            img_tran = transform([img_pil])
            input_tensor = img_tran.unsqueeze(0)  # (1, 3, 224, 224)
            # 转为 numpy 数组（必须为 float32）
            input_np = input_tensor.cpu().numpy().astype(np.float32)

            # 构造 onnxruntime 的输入字典：
            # 第一个输入为图像，后续 10 个为 buffer 状态
            feed_dict = {}
            # 注意：这里按照转换时设置的 input_names 顺序填入数据
            feed_dict[input_names[0]] = input_np
            for i, buf in enumerate(buffer):
                feed_dict[input_names[i+1]] = buf

            # 运行推理，获得 11 个输出
            outputs = session.run(None, feed_dict)
            # 第 1 个输出为分类 logits，其余输出为更新后的 buffer 状态
            feat = outputs[0]
            buffer = outputs[1:]  # 更新 buffer

            # 后处理：若设置 SOFTMAX_THRES，则进行 softmax 判断，否则直接取 argmax
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
