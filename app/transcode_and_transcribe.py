import os
import subprocess
import boto3
import json
from datetime import datetime
import whisper
import logging

# Initialize S3 client
s3_client = boto3.client('s3')

# Set up logging
logging.basicConfig(
    filename='/app/whisper_transcription.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def download_from_s3(s3_path):
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])
    local_path = f"/tmp/{os.path.basename(key)}"
    
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, local_path)
    return local_path

def upload_to_s3(local_path, s3_path):
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])
    
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, key)

def send_status_event(task_arn, file_name, status, progress=None):
    events = boto3.client('events', region_name=os.environ['AWS_REGION'])
    
    event = {
        'version': '0',
        'id': task_arn.split('/')[-1],
        'detail-type': 'ECS Task State Change',
        'source': 'aws.ecs',
        'account': os.environ['AWS_ACCOUNT_ID'],
        'time': datetime.utcnow().isoformat(),
        'region': os.environ['AWS_REGION'],
        'detail': {
            'taskArn': task_arn,
            'fileName': file_name,
            'status': status
        }
    }
    
    if progress is not None:
        event['detail']['progress'] = progress
    
    events.put_events(Entries=[{
        'Source': 'aws.ecs',
        'DetailType': 'ECS Task State Change',
        'Detail': json.dumps(event),
        'EventBusName': 'default'
    }])

def get_video_metadata(input_path):
    # Use ffprobe to get video metadata
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,display_aspect_ratio',
        '-of', 'json', input_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    metadata = json.loads(result.stdout)
    print("Available metadata:", json.dumps(metadata, indent=2))
    return metadata['streams'][0]

def transcribe_audio(local_file_path, output_file):
    """Use Whisper to transcribe the audio file."""
    try:
        logging.info(f"Loading Whisper model and starting transcription for {local_file_path}")
        model = whisper.load_model("large", weights_only=True).to("cuda")

        # Add word_timestamps=True to get word-level timestamps in the result
        result = model.transcribe(local_file_path, word_timestamps=True)

        # Save the transcription as a JSON file
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        logging.info(f"Saved transcription to {output_file}")
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise

def process_video():
    # Get environment variables
    bucket_name = os.environ['S3_BUCKET']  # Will be passed from ECS task
    input_key = os.environ['INPUT_KEY']    # From SQS/Lambda
    
    # Define paths within the bucket
    input_path = f"s3://{bucket_name}/uploads/{input_key}"
    base_name = os.path.splitext(os.path.basename(input_key))[0]
    transcoding_path = f"s3://{bucket_name}/transcoding_samples/{base_name}/"
    transcription_path = f"s3://{bucket_name}/transcriptions/{base_name}.json"
    
    task_arn = os.environ['ECS_CONTAINER_METADATA_URI_V4'].split('/')[-2]
    
    local_input = download_from_s3(input_path)
    
    # Define progress steps
    progress_steps = {
        'DOWNLOAD_COMPLETED': 10,
        'AUDIO_PROCESSING_COMPLETED': 20,
        'VIDEO_PROCESSING_COMPLETED': 70,
        'PLAYLIST_CREATION_COMPLETED': 80,
        'DASH_MANIFEST_CREATION_COMPLETED': 90,
        'UPLOAD_COMPLETED': 100
    }
    
    # Get video metadata
    metadata = get_video_metadata(local_input)

    # Debug metadata
    print("Available metadata:", json.dumps(metadata, indent=2))
    
    # Calculate aspect ratio
    width = int(metadata["width"])
    height = int(metadata["height"])
    actual_ratio = width / height
    target_ratio = 16 / 9
    
    # Allow small floating point differences
    if abs(actual_ratio - target_ratio) > 0.01:
        raise ValueError(f"Input video is not in 16:9 aspect ratio. Got {actual_ratio:.3f}, expected {target_ratio:.3f}")
    
    print(f"Aspect ratio check passed: {width}x{height} = {actual_ratio:.3f} ≈ 16:9")

    # Create temporary working directory
    work_dir = f"/tmp/{base_name}"
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        # Send status event: Download completed
        send_status_event(task_arn, base_name, 'DOWNLOAD_COMPLETED', progress_steps['DOWNLOAD_COMPLETED'])

        # Process audio separately (high quality)
        audio_output = f"{work_dir}/{base_name}-WAV.mp4"
        audio_cmd = f'ffmpeg -i {local_input} -vn -c:a aac -b:a 256k {audio_output}'
        subprocess.run(audio_cmd, shell=True, check=True)

        # Transcribe the audio file
        transcription_output = f"{work_dir}/{base_name}.json"
        transcribe_audio(audio_output, transcription_output)

        # Upload transcription to S3
        upload_to_s3(transcription_output, transcription_path)

        # Send status event: Audio processing completed
        send_status_event(task_arn, base_name, 'AUDIO_PROCESSING_COMPLETED', progress_steps['AUDIO_PROCESSING_COMPLETED'])
        
        # Define video variants
        variants = [
            {
                "name": "UHD",
                "size": "3840x2160",
                "video_codec": "hevc_nvenc",
                "video_opts": "-preset p7 -rc vbr -cq 23 -b:v 12M -maxrate 12M -bufsize 24M",
                "bitrate": "12000k",
                "audio_opts": "-c:a aac -b:a 192k"
            },
            {
                "name": "1080P-H265",
                "size": "1920x1080",
                "video_codec": "hevc_nvenc",
                "video_opts": "-preset p7 -rc vbr -cq 23 -b:v 6M -maxrate 6M -bufsize 12M",
                "bitrate": "6000k",
                "audio_opts": "-c:a aac -b:a 192k"
            },
            {
                "name": "1080P-H264",
                "size": "1920x1080",
                "video_codec": "h264_nvenc",
                "video_opts": "-preset p7 -rc vbr -cq 23 -b:v 6M -maxrate 6M -bufsize 12M",
                "bitrate": "6000k",
                "audio_opts": "-c:a aac -b:a 192k"
            },
            {
                "name": "720P-H264",
                "size": "1280x720",
                "video_codec": "h264_nvenc",
                "video_opts": "-preset p7 -rc vbr -cq 23 -b:v 4M -maxrate 4M -bufsize 8M",
                "bitrate": "4000k",
                "audio_opts": "-c:a aac -b:a 192k"
            },
            {
                "name": "540P-H264",
                "size": "960x540",
                "video_codec": "h264_nvenc",
                "video_opts": "-preset p7 -rc vbr -cq 23 -b:v 2M -maxrate 2M -bufsize 4M",
                "bitrate": "2500k",
                "audio_opts": "-c:a aac -b:a 192k"
            }
        ]

        # Filter variants based on input resolution
        filtered_variants = [v for v in variants if int(v['size'].split('x')[0]) <= width and int(v['size'].split('x')[1]) <= height]

        # Create fragmented MP4s for each variant
        for variant in filtered_variants:
            output_file = f"{work_dir}/{base_name}-{variant['name']}.mp4"
            cmd = (
                f'ffmpeg -i {local_input} '
                f'-c:v {variant["video_codec"]} {variant["video_opts"]} '
                f'-vf scale={variant["size"]} '
                f'{variant["audio_opts"]} '
                f'-g 48 -keyint_min 48 '  # GOP size of 2 seconds at 24fps
                f'-sc_threshold 0 '  # Disable scene change detection
                f'-movflags +frag_keyframe+empty_moov+default_base_moof '  # Fragmented MP4
                f'-segment_time 6 '  # 6-second segments
                f'{output_file}'
            )
            subprocess.run(cmd, shell=True, check=True)

        # Send status event: Video processing completed
        send_status_event(task_arn, base_name, 'VIDEO_PROCESSING_COMPLETED', progress_steps['VIDEO_PROCESSING_COMPLETED'])

        # Create HLS master playlist
        with open(f"{work_dir}/{base_name}.m3u8", "w") as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:3\n")
            
            # Add subtitle tracks (only once at the top)
            f.write(f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="English",LANGUAGE="en",AUTOSELECT=YES,DEFAULT=YES,URI="subtitles/{base_name}_en.vtt"\n')
            f.write(f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",NAME="Spanish",LANGUAGE="es",AUTOSELECT=NO,DEFAULT=NO,URI="subtitles/{base_name}_es.vtt"\n')

            # Add video streams
            for variant in filtered_variants:
                f.write(f'#EXT-X-STREAM-INF:BANDWIDTH={variant["bitrate"].replace("k","000")},RESOLUTION={variant["size"]},CODECS="avc1.4d401f,mp4a.40.2",SUBTITLES="subs"\n')
                f.write(f'{base_name}-{variant["name"]}.m3u8\n')

        # Send status event: Playlist creation completed
        send_status_event(task_arn, base_name, 'PLAYLIST_CREATION_COMPLETED', progress_steps['PLAYLIST_CREATION_COMPLETED'])

        # Generate DASH manifest and segments
        dash_cmd = (
            f'ffmpeg -i {work_dir}/{base_name}-1080P-H265.mp4 '
            f'-i {work_dir}/{base_name}-1080P-H264.mp4 '
            f'-i {work_dir}/{base_name}-720P-H264.mp4 '
            f'-i {work_dir}/{base_name}-540P-H264.mp4 '
            f'-map 0 -map 1 -map 2 -map 3 '
            f'-c copy '
            f'-f dash '
            f'-seg_duration 6 '
            f'-frag_duration 2 '
            f'-dash_segment_type mp4 '
            f'-media_seg_name "chunk-$RepresentationID$-$Number%05d$.m4s" '
            f'-init_seg_name "init-$RepresentationID$.mp4" '
            f'{work_dir}/{base_name}.mpd'
        )
        subprocess.run(dash_cmd, shell=True, check=True)

        # Generate HLS playlists and segments
        for variant in filtered_variants:
            hls_cmd = (
                f'ffmpeg -i {work_dir}/{base_name}-{variant["name"]}.mp4 '
                f'-c copy '
                f'-f hls '
                f'-hls_time 6 '
                f'-hls_segment_type fmp4 '
                f'-hls_playlist_type vod '
                f'-hls_segment_filename "{work_dir}/{variant["name"]}-%d.m4s" '
                f'{work_dir}/{variant["name"]}.m3u8'
            )
            subprocess.run(hls_cmd, shell=True, check=True)

        # Upload all segments and manifests to S3
        for root, _, files in os.walk(work_dir):
            for file in files:
                local_file = os.path.join(root, file)
                s3_key = f"{transcoding_path}{file}"
                upload_to_s3(local_file, s3_key)

        # Send status event: Upload completed
        send_status_event(task_arn, base_name, 'UPLOAD_COMPLETED', progress_steps['UPLOAD_COMPLETED'])

        # Clean up
        subprocess.run(f'rm -rf {work_dir}', shell=True)
        os.remove(local_input)
        
        # Send success event
        send_status_event(task_arn, base_name, 'COMPLETED', 100)
        
    except Exception as e:
        # Send failure event
        send_status_event(task_arn, base_name, 'FAILED', 0)
        raise e

if __name__ == "__main__":
    process_video()