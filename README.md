# HopBox Inference

Automated hop instance segmentation and feature extraction using YOLOv8.

## Overview

This repository contains an inference pipeline for analyzing hop images using a trained YOLOv8 segmentation model. It performs:
- Object detection and segmentation (ColorCard, Hops, QR codes)
- Color correction using reference color cards
- QR code reading and OCR fallback
- Feature extraction from detected hops
- Batch processing of multiple image folders

## Quick Start with Docker

### Pull and Run

```bash
# Pull the pre-built image from Docker Hub
docker pull collins137/hopboxinfer:latest

# Run inference (CPU)
docker run --rm \
  -v "/path/to/your/images:/data/input" \
  -v "/path/to/save/results:/data/output" \
  collins137/hopboxinfer:latest

# Run inference (GPU - Linux/WSL2 with NVIDIA GPU)
docker run --rm --gpus all \
  -v "/path/to/your/images:/data/input" \
  -v "/path/to/save/results:/data/output" \
  collins137/hopboxinfer:latest
```

### Customize Parameters

```bash
docker run --rm --gpus all \
  -e BATCH_SIZE=8 \
  -e CONF=0.4 \
  -e IMGSZ=1024 \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  collins137/hopboxinfer:latest
```

**Available Environment Variables:**
- `BATCH_SIZE` (Default: 4) - Number of images processed per batch
- `CONF` (Default: 0.58) - Confidence threshold for detections
- `IMGSZ` (Default: 2048) - Input image size
- `DEVICE` (Default: auto) - Device to use (`0`, `cpu`, etc.)

## Local Development

### Requirements
- Python 3.11+
- CUDA 12.1+ (for GPU support)

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python inference_final.py \
  --input-dir "/path/to/images" \
  --output-dir "/path/to/results" \
  --batch-size 4 \
  --conf 0.58
```

## Repository Structure

```
.
├── inference_final.py      # Main inference script
├── functions.py            # Helper functions (color correction, QR reading, feature extraction)
├── requirements.txt        # Python dependencies
├── models/
│   └── best.pt            # Trained YOLOv8 model
├── Dockerfile             # Docker image definition
└── README_DOCKER.md       # Detailed Docker instructions
```

## Output

The script generates:
- `Features.csv` - Extracted features for all detected hops
- `Image_Predictions/` - Cropped images of detected objects
- `Results_*/` - YOLO prediction visualizations

## CI/CD

This repository uses GitHub Actions to automatically build and push Docker images to Docker Hub on every push to the main branch.

## Docker Hub

Pre-built image available at: `collins137/hopboxinfer:latest`

## License

[Add your license here]
