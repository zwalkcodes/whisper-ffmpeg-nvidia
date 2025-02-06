# Whisper FFmpeg NVIDIA Container

> ⚠️ **Work in Progress**: This project is under active development. Features and documentation may be incomplete or subject to change. Currently works in production but it needs H.265 support. Terraform infrastructure is stored in a different private repository. Please ask if you'd like me to share.

GPU-accelerated video transcoding and transcription service using Whisper and FFmpeg.

## Overview

This container provides a GPU-accelerated solution for:
- Video transcoding with FFmpeg
- Audio transcription using OpenAI's Whisper
- Support for NVIDIA GPU acceleration
- AWS S3 integration for input/output handling
- SQS queue integration for task management

## Features

- GPU-accelerated video transcoding
- Multiple output formats and resolutions (UHD to 540p)
- H.264 encoding support
- Automatic audio extraction and transcription
- HLS manifest generation
- Progress tracking and status updates
- S3 integration for file handling
- SQS queue integration for processing tasks

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- AWS credentials configured for S3 and SQS access

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
- UHD (3840x2160) - H.264
- 1080p - H.264
- 720p - H.264
- 540p - H.264
- AAC audio (128k)
- Word-level transcription JSON

## SQS Queue Integration

The transcode script listens for messages from an SQS queue to initiate processing tasks. Each message should contain details about the video file to be processed, such as the S3 bucket and key.

### Message Format

Messages sent to the SQS queue should be in JSON format, including:
- `bucket`: The S3 bucket name where the video is stored.
- `key`: The S3 key (path) to the video file.
- `output_formats`: Desired output formats and resolutions.

### Processing Workflow

1. **Receive Message**: The script polls the SQS queue for new messages.
2. **Download Video**: Downloads the video file from S3 based on the message details.
3. **Transcode and Transcribe**: Uses FFmpeg and Whisper to process the video.
4. **Upload Results**: Uploads the transcoded video and transcription back to S3.
5. **Update Status**: Sends a status update back to the queue or logs the result.

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

