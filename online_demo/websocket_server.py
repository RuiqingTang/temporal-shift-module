#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   websocket_server.py
@Time    :   2025/02/24 16:10:00
@Author  :   Ruiqing Tang 
@Contact :   tangruiqing123@gmail.com
'''

import asyncio
import websockets
import json

# WebSocket 服务器配置
HOST = 'localhost'
PORT = 8888

# 存储所有连接的客户端
CLIENTS = set()

async def handle_connection(websocket):
    """处理新的 WebSocket 连接"""
    print(f"新客户端连接: {websocket.remote_address}")
    CLIENTS.add(websocket)
    try:
        async for message in websocket:
            # 将收到的消息广播给所有其他客户端
            for client in CLIENTS:
                if client != websocket:
                    try:
                        await client.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        CLIENTS.remove(client)
    except websockets.exceptions.ConnectionClosed:
        print(f"客户端断开连接: {websocket.remote_address}")
    finally:
        CLIENTS.remove(websocket)

async def main():
    """启动 WebSocket 服务器"""
    async with websockets.serve(handle_connection, HOST, PORT):
        print(f"WebSocket 服务器运行在 ws://{HOST}:{PORT}")
        await asyncio.Future()  # 保持服务器运行

if __name__ == "__main__":
    asyncio.run(main())