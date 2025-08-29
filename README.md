# MoGe Model HTTP API

这个项目提供了一个 HTTP API 服务，用于通过 ONNX 模型运行 MoGe（Monocular Geometric Model）推理。

## 功能

- 通过 HTTP 接口访问 MoGe 模型
- 自动下载模型文件（首次运行时）
- 支持图像上传和处理
- 返回处理后的法线图结果

## 安装依赖

```bash
uv sync
```

## 启动服务

### 直接运行
```bash
uv run python main.py
```

### 使用 Docker
```bash
docker build -t moge-api .
docker run -p 8000:8000 moge-api
```

### 使用 Docker Compose
```bash
docker-compose up
```

服务将在 `http://localhost:8000` 上运行。

## API 端点

### `GET /`

返回 API 信息和可用端点。

### `POST /process`

处理上传的图像。

**参数:**
- `image`: 上传的图像文件（必需）
- `num_tokens`: num_tokens 参数值（可选，默认为 1024）

**返回:**
- 处理后的 PNG 图像文件

## 使用示例

### 使用 cURL

```bash
curl -X POST -F "image=@./assets/source.jpg" -o ./output/result.png http://localhost:8000/process
```

### 使用 Python 客户端

```bash
python moge_client.py http://localhost:8000 ./assets/source.jpg ./output/result.png
```

### 使用原始命令行方式

```bash
uv run python run_moge_model.py ./assets/source.jpg
```

## Docker 部署

### 构建 Docker 镜像
```bash
docker build -t moge-api .
```

### 运行 Docker 容器
```bash
docker run -p 8000:8000 -v ./models:/app/models -v ./output:/app/output moge-api
```

### 使用 Docker Compose
```bash
docker-compose up
```

Docker 部署会自动：
1. 安装所有依赖
2. 在容器启动时下载模型（首次运行时）
3. 挂载模型和输出目录以实现持久化存储

## 项目结构

- `main.py`: 项目入口点，启动 FastAPI 服务
- `moge_api.py`: FastAPI 服务实现
- `run_moge_model.py`: 原始命令行版本
- `moge_client.py`: Python 客户端示例
- `Dockerfile`: Docker 镜像构建文件
- `docker-compose.yml`: Docker Compose 配置文件
- `models/`: 模型文件存储目录
- `assets/`: 示例图像
- `output/`: 输出结果目录

## 依赖

- FastAPI
- Uvicorn
- ONNX Runtime
- NumPy
- Pillow
- Requests

## 注意事项

1. 首次运行时会自动从 Hugging Face 下载模型文件
2. 模型文件存储在 `./models/moge_model.onnx`
3. 服务启动时会预加载模型以提高响应速度
4. 支持的图像格式取决于 Pillow 库的支持
5. Docker 部署会将模型和输出目录挂载到宿主机，以实现持久化存储