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
    filename='/home/ubuntu/whisper_transcription.log',
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
    events = boto3.client('events')
    
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
    return metadata['streams'][0]

def transcribe_audio(local_file_path, output_file):
    """Use Whisper to transcribe the audio file."""
    try:
        logging.info(f"Loading Whisper model and starting transcription for {local_file_path}")
        model = whisper.load_model("large")

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
    input_path = os.environ['INPUT_FILE']
    output_path = os.environ['OUTPUT_PATH']
    task_arn = os.environ['ECS_CONTAINER_METADATA_URI_V4'].split('/')[-2]
    
    # Derive base_name from input_path
    base_name = os.path.splitext(os.path.basename(input_path))[0]
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
    width = metadata['width']
    height = metadata['height']
    aspect_ratio = metadata['display_aspect_ratio']

    # Check if the video is 16:9
    if aspect_ratio != '16:9':
        raise ValueError("Input video is not in 16:9 aspect ratio")

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

        # Send status event: Audio processing completed
        send_status_event(task_arn, base_name, 'AUDIO_PROCESSING_COMPLETED', progress_steps['AUDIO_PROCESSING_COMPLETED'])
        
        # Define video variants
        variants = [
            {
                "name": "UHD",
                "size": "3840x2160",
                "video_codec": "libx265",
                "video_opts": "-preset slow -crf 23 -x265-params vbv-maxrate=12000:vbv-bufsize=24000",
                "bitrate": "12000k",
                "audio_opts": "-c:a aac -b:a 192k"
            },
            {
                "name": "1080P-H265",
                "size": "1920x1080",
                "video_codec": "libx265",
                "video_opts": "-preset slow -crf 23 -x265-params vbv-maxrate=6000:vbv-bufsize=12000",
                "bitrate": "6000k",
                "audio_opts": "-c:a aac -b:a 192k"
            },
            {
                "name": "1080P-H264",
                "size": "1920x1080",
                "video_codec": "libx264",
                "video_opts": "-preset slow -crf 23 -maxrate 6000k -bufsize 12000k",
                "bitrate": "6000k",
                "audio_opts": "-c:a aac -b:a 192k"
            },
            {
                "name": "720P-H264",
                "size": "1280x720",
                "video_codec": "libx264",
                "video_opts": "-preset slow -crf 23 -maxrate 4000k -bufsize 8000k",
                "bitrate": "4000k",
                "audio_opts": "-c:a aac -b:a 192k"
            },
            {
                "name": "540P-H264",
                "size": "960x540",
                "video_codec": "libx264",
                "video_opts": "-preset slow -crf 23 -maxrate 2500k -bufsize 5000k",
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
                s3_key = f"{output_path}{base_name}/{file}"
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