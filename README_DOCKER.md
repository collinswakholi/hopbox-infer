# Docker Instructions for HopBox Inference

This guide explains how to build the Docker image and run the inference script on any machine (Local PC, Cloud VM, etc.) using Docker.

## Prerequisites

1.  **Install Docker:**
    *   **Windows/Mac:** Install Docker Desktop.
    *   **Linux:** Install Docker Engine.

2.  **NVIDIA GPU Support (Linux/WSL2):**
    *   Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed to allow Docker to access your GPU.

## 1. Build the Docker Image

Run the following command in the directory containing the `Dockerfile`:

```bash
docker build -t hopbox-inference:v1 .
```

*   `-t hopbox-inference:v1`: Tags the image with a name and version.
*   `.`: Specifies the current directory as the build context.

## 2. Run Inference

You can run the inference container by mounting your local data folders into the container.

### Basic Usage (CPU or standard Docker)

Assume your images are in `C:\Path\To\Images` (Windows) or `/home/user/images` (Linux).

**Windows (PowerShell):**
```powershell
docker run --rm `
  -v "C:\Path\To\Images:/data/input" `
  -v "C:\Path\To\Results:/data/output" `
  hopbox-inference:v1
```

**Linux/Mac:**
```bash
docker run --rm \
  -v "/home/user/images:/data/input" \
  -v "/home/user/results:/data/output" \
  hopbox-inference:v1
```

*   `--rm`: Automatically removes the container after it stops.
*   `-v ...:/data/input`: Maps your local image folder to the container's input directory.
*   `-v ...:/data/output`: Maps your local result folder to the container's output directory.

### Running with GPU Support (NVIDIA)

Add the `--gpus all` flag:

```bash
docker run --rm --gpus all \
  -v "/home/user/images:/data/input" \
  -v "/home/user/results:/data/output" \
  hopbox-inference:v1
```

### Customizing Parameters

You can change inference parameters by setting environment variables using `-e`:

```bash
docker run --rm --gpus all \
  -e BATCH_SIZE=8 \
  -e CONF=0.4 \
  -e IMGSZ=1024 \
  -v "/home/user/images:/data/input" \
  -v "/home/user/results:/data/output" \
  hopbox-inference:v1
```

**Available Environment Variables:**
*   `BATCH_SIZE` (Default: 4)
*   `CONF` (Default: 0.58)
*   `IMGSZ` (Default: 2048)
*   `MODEL_PATH` (Default: `models/best.pt` - Use internal model)
*   `DEVICE` (Default: Auto-detect - `0`, `cpu`, etc.)

### Using a Different Model

If you have a new model file (e.g., `new_model.pt`) on your host machine, you can mount it and point the container to it:

```bash
docker run --rm \
  -v "/path/to/images:/data/input" \
  -v "/path/to/results:/data/output" \
  -v "/path/to/my_models:/models_mount" \
  -e MODEL_PATH="/models_mount/new_model.pt" \
  hopbox-inference:v1
```

## 3. Push to Docker Hub

To share this image or verify it, you can push it to your Docker Hub account.

1.  **Login to Docker Hub** (if not already logged in):
    ```bash
    docker login
    ```
    Enter your username (`collins137`) and password.

2.  **Tag the image:**
    If you built it as `hopbox-inference:v1`, tag it for your repository:
    ```bash
    docker tag hopbox-inference:v1 collins137/hopboxinfer:latest
    ```

3.  **Push the image:**
    ```bash
    docker push collins137/hopboxinfer:latest
    ```

4.  **Pull and Run on another machine:**
    ```bash
    docker run --rm --gpus all \
      -v "/path/to/local/images:/data/input" \
      -v "/path/to/local/results:/data/output" \
      collins137/hopboxinfer:latest
    ```

## Troubleshooting

*   **"No compatible GPUs found":** Ensure `--gpus all` is passed (Linux/WSL2 only) and NVIDIA drivers are installed on the host.
*   **Permissions:** If output files are locked, check user permissions on the mapped volume folders.
