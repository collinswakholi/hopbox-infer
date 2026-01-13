# HopBox Docker Usage Guide

Complete guide for running the HopBox inference Docker container with all available options.

## Quick Start

### Basic Run (Auto-detect CPU/GPU)

```bash
docker run --rm \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  collins137/hopboxinfer:latest
```

**Windows PowerShell:**
```powershell
docker run --rm `
  -v "C:\Path\To\Images:/data/input" `
  -v "C:\Path\To\Results:/data/output" `
  collins137/hopboxinfer:latest
```

## All Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `/data/input` | Path to folder containing image subfolders |
| `OUTPUT_DIR` | `/data/output` | Path to save results |
| `MODEL_PATH` | `/app/models/best.pt` | Path to YOLOv8 model (built-in by default) |
| `BATCH_SIZE` | `4` | Number of images to process per batch |
| `CONF` | `0.58` | Confidence threshold (0.0-1.0) |
| `IOU` | `0.75` | NMS IoU threshold (0.0-1.0) |
| `IMGSZ` | `2048` | Input image size for inference |
| `DEVICE` | `auto` | Device to use: `0` (GPU), `cpu`, or empty for auto-detect |

### Complete Example with All Options

```bash
docker run --rm \
  --gpus all \
  -e INPUT_DIR="/data/input" \
  -e OUTPUT_DIR="/data/output" \
  -e BATCH_SIZE=8 \
  -e CONF=0.4 \
  -e IOU=0.75 \
  -e IMGSZ=1024 \
  -e DEVICE="0" \
  -v "/home/user/hopbox_images:/data/input" \
  -v "/home/user/hopbox_results:/data/output" \
  collins137/hopboxinfer:latest
```

**Windows PowerShell:**
```powershell
docker run --rm `
  --gpus all `
  -e INPUT_DIR="/data/input" `
  -e OUTPUT_DIR="/data/output" `
  -e BATCH_SIZE=8 `
  -e CONF=0.4 `
  -e IOU=0.75 `
  -e IMGSZ=1024 `
  -e DEVICE="0" `
  -v "C:\Users\You\HopBox\Images:/data/input" `
  -v "C:\Users\You\HopBox\Results:/data/output" `
  collins137/hopboxinfer:latest
```

**Note:** The model is built into the container at `/app/models/best.pt`. You only need to specify `MODEL_PATH` if using a custom model.

## Usage Scenarios

### 1. CPU-Only Inference

```bash
docker run --rm \
  -e DEVICE="cpu" \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  collins137/hopboxinfer:latest
```

### 2. GPU Inference (Linux/WSL2 with NVIDIA GPU)

```bash
docker run --rm \
  --gpus all \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  collins137/hopboxinfer:latest
```

### 3. High Confidence Filtering

For stricter detections, increase confidence threshold:

```bash
docker run --rm \
  -e CONF=0.75 \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  collins137/hopboxinfer:latest
```

### 4. Faster Processing (Smaller Image Size, Larger Batches)

Trade accuracy for speed:

```bash
docker run --rm \
  -e BATCH_SIZE=16 \
  -e IMGSZ=1024 \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  collins137/hopboxinfer:latest
```

### 5. Maximum Quality (Slower)

For best accuracy:

```bash
docker run --rm \
  -e BATCH_SIZE=2 \
  -e IMGSZ=2048 \
  -e CONF=0.5 \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  collins137/hopboxinfer:latest
```

### 6. Custom Model

Use your own trained model:

```bash
docker run --rm \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  -v "/path/to/models:/custom_models" \
  -e MODEL_PATH="/custom_models/my_model.pt" \
  collins137/hopboxinfer:latest
```

### 7. Multiple GPU Devices

Specify which GPU to use:

```bash
# Use GPU 0
docker run --rm --gpus '"device=0"' \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  collins137/hopboxinfer:latest

# Use GPUs 0 and 1
docker run --rm --gpus '"device=0,1"' \
  -e DEVICE="0,1" \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  collins137/hopboxinfer:latest
```

## Input Folder Structure

Your input folder should contain subfolders with images:

```
/data/input/
├── Batch_1/
│   ├── IMG_0001.JPG
│   ├── IMG_0002.JPG
│   └── ...
├── Batch_2/
│   ├── IMG_0001.JPG
│   └── ...
└── ...
```

## Output Structure

Results will be organized by input folder:

```
/data/output/
├── Batch_1/
│   ├── Features.csv
│   ├── Image_Predictions/
│   │   ├── IMG_0001.JPG
│   │   └── ...
│   └── Results_Batch_1/
│       ├── crops/
│       └── ...
└── Batch_2/
    └── ...
```

## Performance Tuning

### GPU Memory Issues

If you get "CUDA out of memory" errors, reduce batch size:

```bash
docker run --rm --gpus all \
  -e BATCH_SIZE=2 \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  collins137/hopboxinfer:latest
```

### Speed vs Quality Trade-offs

| Setting | Speed | Quality | Memory |
|---------|-------|---------|--------|
| `BATCH_SIZE=16, IMGSZ=1024` | Fast | Lower | Low |
| `BATCH_SIZE=4, IMGSZ=2048` | Balanced | Good | Medium |
| `BATCH_SIZE=1, IMGSZ=2048` | Slow | Best | High |

### Recommended Settings by Hardware

**High-end GPU (RTX 3090, A100):**
```bash
-e BATCH_SIZE=8 -e IMGSZ=2048
```

**Mid-range GPU (RTX 3060, RTX 4060):**
```bash
-e BATCH_SIZE=4 -e IMGSZ=2048
```

**Low-end GPU or CPU:**
```bash
-e BATCH_SIZE=2 -e IMGSZ=1024
```

## Troubleshooting

### Container Exits Immediately

Check the logs:
```bash
docker logs $(docker ps -lq)
```

### Permission Errors

Ensure output directory has write permissions or run with user mapping:
```bash
docker run --rm \
  --user $(id -u):$(id -g) \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  collins137/hopboxinfer:latest
```

### No GPU Detected

Verify NVIDIA Container Toolkit is installed:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Advanced Usage

### Run Interactively (Debug Mode)

```bash
docker run -it \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  --entrypoint /bin/bash \
  collins137/hopboxinfer:latest
```

Then inside the container:
```bash
python inference_final.py \
  --input-dir /data/input \
  --output-dir /data/output \
  --batch-size 4
```

### Keep Container Running

```bash
docker run -d \
  --name hopbox-inference \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  collins137/hopboxinfer:latest

# View logs
docker logs -f hopbox-inference

# Stop when done
docker stop hopbox-inference
docker rm hopbox-inference
```

## Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  hopbox:
    image: collins137/hopboxinfer:latest
    volumes:
      - ./images:/data/input
      - ./results:/data/output
    environment:
      - BATCH_SIZE=8
      - CONF=0.58
      - IMGSZ=2048
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Run with:
```bash
docker-compose up
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/collinswakholi/hopbox-infer/issues
- Docker Hub: https://hub.docker.com/r/collins137/hopboxinfer
