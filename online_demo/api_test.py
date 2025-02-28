#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   api_test.py
@Time    :   2025/02/24 16:07:17
@Author  :   Ruiqing Tang 
@Contact :   tangruiqing123@gmail.com
'''


import websocket
import json

# WebSocket 服务器的地址和端口
WEBSOCKET_URL = "ws://localhost:8888"

def on_message(ws, message):
    """处理接收到的消息"""
    try:
        data = json.loads(message)  # 解析 JSON 数据
        if data.get("type") == "gesture":
            category = data.get("category")
            fps = data.get("fps")
            print(f"Gesture: {category} | FPS: {fps:.1f}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

def on_error(ws, error):
    """处理错误"""
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    """处理连接关闭"""
    print("WebSocket connection closed")

def on_open(ws):
    """连接成功后执行的操作"""
    print("Connected to WebSocket server")

# 初始化 WebSocket 客户端
if __name__ == "__main__":
    ws = websocket.WebSocketApp(WEBSOCKET_URL,
                            on_open=on_open,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)

    print(f"Connecting to WebSocket server at {WEBSOCKET_URL}...")
    ws.run_forever()  # 启动 WebSocket 客户端