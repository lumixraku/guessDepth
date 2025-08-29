# MoGe Model HTTP API

This project provides an HTTP API service for running MoGe (Monocular Geometric Model) inference through an ONNX model.

## Features

- Access MoGe model through HTTP interface
- Automatic model file download (on first run)
- Support for image upload and processing
- Returns processed normal map results

## Install Dependencies

```bash
uv sync
```

## Start Service

### Direct Run
```bash
uv run python main.py
```

### Using Docker
```bash
docker build -t moge-api .
docker run -p 8000:8000 moge-api
```

### Using Docker Compose
```bash
docker-compose up
```

The service will run on `http://localhost:8000`.

## API Endpoints

### `GET /`

Returns API information and available endpoints.

### `POST /process`

Processes uploaded images.

**Parameters:**
- `image`: Uploaded image file (required)
- `num_tokens`: num_tokens parameter value (optional, default is 1024)

**Returns:**
- Processed PNG image file

## Usage Examples

### Using cURL

```bash
curl -X POST -F "image=@./assets/source.jpg" -o ./output/result.png http://localhost:8000/process
```

### Using Python Client

```bash
python moge_client.py http://localhost:8000 ./assets/source.jpg ./output/result.png
```

### Using Original Command Line Version

```bash
uv run python run_moge_model.py ./assets/source.jpg
```

## Docker Deployment

### Build Docker Image
```bash
docker build -t moge-api .
```

### Run Docker Container
```bash
docker run -p 8000:8000 -v ./models:/app/models -v ./output:/app/output moge-api
```

### Using Docker Compose
```bash
docker-compose up
```

Docker deployment will automatically:
1. Install all dependencies
2. Download model on container startup (on first run)
3. Mount model and output directories for persistent storage

## Project Structure

- `main.py`: Project entry point, starts FastAPI service
- `moge_api.py`: FastAPI service implementation
- `run_moge_model.py`: Original command line version
- `moge_client.py`: Python client example
- `Dockerfile`: Docker image build file
- `docker-compose.yml`: Docker Compose configuration file
- `models/`: Model file storage directory
- `assets/`: Sample images
- `output/`: Output results directory

## Dependencies

- FastAPI
- Uvicorn
- ONNX Runtime
- NumPy
- Pillow
- Requests

## Notes

1. Model files will be automatically downloaded from Hugging Face on first run
2. Model files are stored in `./models/moge_model.onnx`
3. Model is pre-loaded on service startup for improved response speed
4. Supported image formats depend on Pillow library support
5. Docker deployment mounts model and output directories to host for persistent storage