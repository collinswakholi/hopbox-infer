# Use an official PyTorch runtime as a parent image
# We use a tag that includes CUDA for GPU support. 
# If running on CPU only, this image will still work but is larger than necessary.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and other tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
# This includes inference_final.py, functions.py, and models/ folder
COPY . /app

# Make the inference script executable (optional)
RUN chmod +x inference_final.py

# Define environment variables with defaults (can be overridden at runtime)
# These map to the arguments in inference_final.py
ENV INPUT_DIR="/data/input"
ENV OUTPUT_DIR="/data/output"
ENV MODEL_PATH="models/best.pt"
ENV BATCH_SIZE=4
ENV CONF=0.58
ENV IMGSZ=2048
ENV DEVICE=""

# Create directories for mounting volumes
RUN mkdir -p /data/input /data/output

# Entrypoint script effectively runs the python command with arguments mapped from env vars
ENTRYPOINT ["/bin/bash", "-c", "python inference_final.py --input-dir \"${INPUT_DIR}\" --output-dir \"${OUTPUT_DIR}\" --model \"${MODEL_PATH}\" --batch-size \"${BATCH_SIZE}\" --conf \"${CONF}\" --imgsz \"${IMGSZ}\" --device \"${DEVICE}\""]
