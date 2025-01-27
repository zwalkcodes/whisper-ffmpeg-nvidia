# Whisper FFmpeg NVIDIA Container

> ⚠️ **Work in Progress**: This project is under active development. Features and documentation may be incomplete or subject to change.

GPU-accelerated video transcoding and transcription service using Whisper and FFmpeg.

## Overview

This container provides a GPU-accelerated solution for:
- Video transcoding with FFmpeg
- Audio transcription using OpenAI's Whisper
- Support for NVIDIA GPU acceleration
- AWS S3 integration for input/output handling

## Features

- GPU-accelerated video transcoding
- Multiple output formats and resolutions (UHD to 540p)
- H.264 and H.265 encoding support
- Automatic audio extraction and transcription
- DASH and HLS manifest generation
- Progress tracking and status updates
- S3 integration for file handling

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- AWS credentials configured for S3 access

## Building the Image

```bash
docker build -t whisper-ffmpeg-nvidia .
```

## Running the Container

```bash
docker run --gpus all whisper-ffmpeg-nvidia
```

## Output Formats

The container generates multiple output formats:
- UHD (3840x2160) - H.265
- 1080p - H.265 and H.264
- 720p - H.264
- 540p - H.264
- High-quality AAC audio (256k)
- Word-level transcription JSON

## Project Structure

```
whisper-ffmpeg-nvidia/
├── .github/
│   └── workflows/
│       └── deploy.yaml    # GitHub Actions workflow
├── app/
│   └── transcode_and_transcribe.py
└── Dockerfile
```

## CI/CD

The project includes GitHub Actions workflow for:
- Building the Docker image
- Pushing to Amazon ECR
- Automated deployments on main branch updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

