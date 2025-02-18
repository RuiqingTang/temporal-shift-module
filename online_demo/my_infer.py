#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   my_main.py
@Time    :   2025/02/05 14:32:10
@Author  :   Ruiqing Tang 
@Contact :   tangruiqing123@gmail.com
'''

import numpy as np
import cv2
import os
import time
from typing import Tuple
from PIL import Image, ImageOps
import torch
import torchvision

# 导入模型（请确保 mobilenet_v2_tsm.py 文件在当前目录或 Python 路径中）
from mobilenet_v2_tsm import MobileNetV2

SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True

# =============================================================================
# 以下为图像预处理相关定义
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
    "Doing other things",  # 0
    "Drumming Fingers",    # 1
    "No gesture",          # 2
    "Pulling Hand In",     # 3
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
    # 若不启用 refine 输出，则直接返回当前预测
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # 历史缓冲区最大长度

    # 屏蔽不合法的动作（若当前预测属于非法类别，则保持上一帧的预测）
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # 将“Doing other things”（类别 0）替换为单一的“no gesture”（类别 2）
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
# 模型加载函数（直接使用 PyTorch 推理，不进行转换）
# =============================================================================
def get_model():
    # 创建模型并加载预训练权重
    model = MobileNetV2(n_class=27)
    if not os.path.exists("mobilenetv2_jester_online.pth.tar"):
        print('Downloading PyTorch checkpoint...')
        import urllib.request
        url = 'https://hanlab18.mit.edu/projects/tsm/models/mobilenetv2_jester_online.pth.tar'
        urllib.request.urlretrieve(url, './mobilenetv2_jester_online.pth.tar')
    print("Loading model...")
    checkpoint = torch.load("mobilenetv2_jester_online.pth.tar", map_location='cpu')
    # 如果 checkpoint 中含有 'state_dict' 键，则加载该字典，否则直接加载整个 checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 自动选择 GPU（如果可用）或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)
    model.to(device)

    # 初始化 buffer（状态变量），与原始模型输入对应，形状参考原始代码：
    # 第一个输入为图像，其余 10 个为状态。这里只初始化状态为零张量。
    buffer = (
        torch.zeros(1, 3, 56, 56, device=device),
        torch.zeros(1, 4, 28, 28, device=device),
        torch.zeros(1, 4, 28, 28, device=device),
        torch.zeros(1, 8, 14, 14, device=device),
        torch.zeros(1, 8, 14, 14, device=device),
        torch.zeros(1, 8, 14, 14, device=device),
        torch.zeros(1, 12, 14, 14, device=device),
        torch.zeros(1, 12, 14, 14, device=device),
        torch.zeros(1, 20, 7, 7, device=device),
        torch.zeros(1, 20, 7, 7, device=device)
    )
    return model, buffer, device

# =============================================================================
# 主函数：使用 OpenCV 采集摄像头视频并进行实时手势识别
# =============================================================================
WINDOW_NAME = 'Video Gesture Recognition'
def main():
    print("Open camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    # 设置较低分辨率以加快处理速度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    index = 0
    print("Build transformer...")
    transform = get_transform()
    print("Loading Torch model...")
    model, buffer, device = get_model()

    idx = 0
    history = [2]
    history_logit = []
    i_frame = -1

    print("Ready!")
    while True:
        i_frame += 1
        ret, img = cap.read()
        if not ret:
            break

        # 为降低帧率，每隔一帧处理一次
        # if i_frame % 2 == 0:
        t1 = time.time()
        # 使用 PIL 打开图像并转换为 RGB 格式
        img_pil = Image.fromarray(img).convert('RGB')
        # 批量传入（这里只传入一帧）进行预处理，输出 tensor 形状为 (C, H, W)
        img_tran = transform([img_pil])
        input_tensor = img_tran.unsqueeze(0).to(device)  # (1, 3, 224, 224)
        with torch.no_grad():
            # 模型前向传播：第一个输入为图像，其余为状态变量（buffer）
            outputs = model(input_tensor, *buffer)
        # 输出第一个元素为分类结果，其余为更新后的状态（buffer）
        feat = outputs[0]
        buffer = outputs[1:]
        # 将 logits 转换为 numpy 数组，便于后续处理
        feat_np = feat.cpu().numpy()
        if SOFTMAX_THRES > 0:
            feat_np_flat = feat_np.reshape(-1)
            feat_np_flat -= feat_np_flat.max()
            softmax = np.exp(feat_np_flat) / np.sum(np.exp(feat_np_flat))
            print("最大 softmax 值：", max(softmax))
            if max(softmax) > SOFTMAX_THRES:
                idx_ = np.argmax(feat_np, axis=1)[0]
            else:
                idx_ = idx
        else:
            idx_ = np.argmax(feat_np, axis=1)[0]

        if HISTORY_LOGIT:
            history_logit.append(feat_np)
            history_logit = history_logit[-12:]
            avg_logit = sum(history_logit)
            idx_ = np.argmax(avg_logit, axis=1)[0]

        idx, history = process_output(idx_, history)

        t2 = time.time()
        current_time = t2 - t1
        print(f"{index} {catigories[idx]}  ({1/current_time:.1f} Vid/s)")
        index += 1

        # 显示视频，添加预测信息
        disp_img = cv2.resize(img, (640, 480))
        disp_img = disp_img[:, ::-1]  # 水平翻转（镜像效果）
        height, width, _ = disp_img.shape
        label = np.full((height // 10, width, 3), 255, dtype='uint8')
        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1/current_time if current_time > 0 else 0),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        disp_img = np.concatenate((disp_img, label), axis=0)
        cv2.imshow(WINDOW_NAME, disp_img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # 按 q 或 ESC 退出
            break
        elif key in [ord('F'), ord('f')]:  # 全屏切换
            print('Changing full screen option!')
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
