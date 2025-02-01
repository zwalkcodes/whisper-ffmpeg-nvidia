# Use the NVIDIA base image
FROM jrottenberg/ffmpeg:4.1-nvidia

RUN echo "Acquire::http::Pipeline-Depth 0;" > /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::http::No-Cache true;" >> /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::BrokenProxy    true;" >> /etc/apt/apt.conf.d/99custom

RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean
RUN apt-get update -o Acquire::CompressionTypes::Order::=gz

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Whisper and other Python packages
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir \
    torch \
    torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir \
    openai-whisper \
    ffmpeg-python \
    boto3 \
    awscli \
    vim

# Set the working directory
WORKDIR /app

# Copy your application code
COPY app/ .

# Make the script executable
RUN chmod +x transcode_and_transcribe.py

# Define the command to run your application
ENTRYPOINT ["/bin/bash", "-c"]