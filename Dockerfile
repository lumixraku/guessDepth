# 使用官方 Python 运行时作为基础镜像
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装 uv
RUN pip install uv

# 使用 uv 安装项目依赖
RUN uv sync

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV PYTHONPATH=/app

# 启动服务
CMD ["uv", "run", "python", "moge_api.py"]