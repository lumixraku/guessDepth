#!/usr/bin/env python3
"""
MoGe Model HTTP API Server
通过 main.py 启动 FastAPI 服务
"""

import uvicorn
import sys
import os

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入 FastAPI 应用
from moge_api import app

def main():
    """启动 FastAPI 服务"""
    print("正在启动 MoGe Model HTTP API 服务...")
    print("访问 http://localhost:8000 查看 API 文档")
    
    # 使用 uvicorn 启动应用
    uvicorn.run(
        "moge_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # 生产环境中关闭重载
    )

if __name__ == "__main__":
    main()