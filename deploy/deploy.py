#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   deploy.py
@Time    :   2025/01/25 23:50:20
@Author  :   Ruiqing Tang 
@Contact :   tangruiqing123@gmail.com
'''


from typing import Tuple
import numpy as np
import cv2

import os
os.environ["TVM_HOME"] = "D:/Projects/Python_projects/pose_estimation/temporal-shift-module/third_party/tvm/python/tvm"
# os.environ["TVM_CXX_COMPILER"] = "D:/Software/Anaconda/envs/tsm/Library/bin/clang++.exe"

import tvm
import tvm.relay
import time

SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True

catigories = [
    "Doing other things",  # 0
    "Drumming Fingers",  # 1
    "No gesture",  # 2
    "Pulling Hand In",  # 3
    "Pulling Two Fingers In",  # 4
    "Pushing Hand Away",  # 5
    "Pushing Two Fingers Away",  # 6
    "Rolling Hand Backward",  # 7
    "Rolling Hand Forward",  # 8
    "Shaking Hand",  # 9
    "Sliding Two Fingers Down",  # 10
    "Sliding Two Fingers Left",  # 11
    "Sliding Two Fingers Right",  # 12
    "Sliding Two Fingers Up",  # 13
    "Stop Sign",  # 14
    "Swiping Down",  # 15
    "Swiping Left",  # 16
    "Swiping Right",  # 17
    "Swiping Up",  # 18
    "Thumb Down",  # 19
    "Thumb Up",  # 20
    "Turning Hand Clockwise",  # 21
    "Turning Hand Counterclockwise",  # 22
    "Zooming In With Full Hand",  # 23
    "Zooming In With Two Fingers",  # 24
    "Zooming Out With Full Hand",  # 25
    "Zooming Out With Two Fingers"  # 26
]


def transform(frame: np.ndarray):
    # OpenCV处理图像
    frame = cv2.resize(frame, (224, 224))  # (224, 224, 3) 0 ~ 255
    frame = frame / 255.0  # (224, 224, 3) 0 ~ 1.0
    frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
    frame = np.expand_dims(frame, axis=0)  # (1, 3, 224, 224) 0 ~ 1.0
    # 归一化
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    frame = (frame - mean) / std
    return frame.astype(np.float32)

def process_output(idx_, history):
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # use only single no action class
    if idx_ == 0:
        idx_ = 2
    
    # history smoothing
    if len(history) > 1 and idx_ != history[-1]:
        if not (history[-1] == history[-2]):
            idx_ = history[-1]
    
    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history

WINDOW_NAME = 'Video Gesture Recognition'

def main():
    print("Open camera...")
    cap = cv2.VideoCapture(0)
    
    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    print("Build Executor...")
    # 直接用转换后的模型
    lib_fname = 'mobilenet_tsm_tvm_llvm.tar'
    graph_fname = 'mobilenet_tsm_tvm_llvm.json'
    params_fname = 'mobilenet_tsm_tvm_llvm.params'
    
    with open(graph_fname, 'rt') as f:
        graph = f.read()
    tvm_module = tvm.runtime.load_module(lib_fname)
    params = tvm.runtime.load_param_dict(bytearray(open(params_fname, 'rb').read()))
    
    device = tvm.cpu()
    graph_module = tvm.contrib.graph_executor.create(graph, tvm_module, device)

    for pname, pvalue in params.items():
        graph_module.set_input(pname, pvalue)

    def executor(inputs: Tuple[tvm.nd.NDArray]):
        for index, value in enumerate(inputs):
            graph_module.set_input(index, value)
        graph_module.run()
        return tuple(graph_module.get_output(index) for index in range(len(inputs)))

    buffer = (
        tvm.nd.empty((1, 3, 56, 56), device=device),
        tvm.nd.empty((1, 4, 28, 28), device=device),
        tvm.nd.empty((1, 4, 28, 28), device=device),
        tvm.nd.empty((1, 8, 14, 14), device=device),
        tvm.nd.empty((1, 8, 14, 14), device=device),
        tvm.nd.empty((1, 8, 14, 14), device=device),
        tvm.nd.empty((1, 12, 14, 14), device=device),
        tvm.nd.empty((1, 12, 14, 14), device=device),
        tvm.nd.empty((1, 20, 7, 7), device=device),
        tvm.nd.empty((1, 20, 7, 7), device=device)
    )

    idx = 0
    history = [2]
    history_logit = []
    i_frame = -1

    print("Ready!")
    while True:
        i_frame += 1
        _, img = cap.read()
        if i_frame % 2 == 0:
            t1 = time.time()
            img_tran = transform(img)
            img_nd = tvm.nd.array(img_tran, device=device)
            inputs = (img_nd,) + buffer
            outputs = executor(inputs)
            feat, buffer = outputs[0], outputs[1:]
            
            if SOFTMAX_THRES > 0:
                feat_np = feat.asnumpy().reshape(-1)
                feat_np -= feat_np.max()
                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))
                if max(softmax) > SOFTMAX_THRES:
                    idx_ = np.argmax(feat.asnumpy(), axis=1)[0]
                else:
                    idx_ = idx
            else:
                idx_ = np.argmax(feat.asnumpy(), axis=1)[0]

            if HISTORY_LOGIT:
                history_logit.append(feat.asnumpy())
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = process_output(idx_, history)
            t2 = time.time()
            current_time = t2 - t1

        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            full_screen = not full_screen
            if full_screen:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()