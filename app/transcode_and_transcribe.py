import os
import subprocess
import boto3
import json
from datetime import datetime
import whisper
import logging
import requests
import torch

region = os.getenv('AWS_REGION', 'us-west-1')  # Default to 'us-west-1' if not set

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=region)
ec2 = boto3.client('ec2', region_name=region)
sqs = boto3.client('sqs', region_name=region)
dynamodb = boto3.client('dynamodb', region_name=region)

INSTANCE_ID = requests.get('http://169.254.169.254/latest/meta-data/instance-id').text
QUEUE_URL = os.environ.get("SQS_QUEUE_URL")

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/app/whisper_transcription.log'),
        logging.StreamHandler()  # This sends logs to console
    ]
)

def optimize_gpu():
    if torch.cuda.is_available():
        # Enable TF32 for faster processing on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        torch.cuda.empty_cache()

def set_termination_protection(enabled):
    """
    Sets termination protection for this EC2 instance.
    """
    try:
        ec2.modify_instance_attribute(
            InstanceId=INSTANCE_ID,
            DisableApiTermination={
                'Value': enabled
            }
        )
        logging.info(f"Set termination protection: {enabled}")
    except Exception as e:
        logging.error(f"Failed to set termination protection: {e}")

def poll_queue():
    """
    Poll messages from SQS. If no messages are found after a few attempts, shut down.
    """
    max_empty_attempts = 3
    wait_time = 20  # seconds to wait for each poll
    visibility_timeout = 43200  # 12 hours - maximum allowed by SQS

    for attempt in range(max_empty_attempts):
        logging.info(f"Polling attempt {attempt + 1}/{max_empty_attempts}")
        
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=wait_time,
            VisibilityTimeout=visibility_timeout,
            AttributeNames=['All']
        )

        if 'Messages' in response:
            for message in response['Messages']:
                try:
                    logging.info(f"Processing message: {message['MessageId']}")
                    set_termination_protection(True)
                    process_message(message['Body'])

                    # Delete message after successful processing
                    sqs.delete_message(
                        QueueUrl=QUEUE_URL,
                        ReceiptHandle=message['ReceiptHandle']
                    )
                    logging.info(f"Successfully processed and deleted message: {message['MessageId']}")
                except Exception as e:
                    logging.error(f"Error processing message: {e}")
                    
                    # Delete the failed message to prevent infinite retries
                    try:
                        sqs.delete_message(
                            QueueUrl=QUEUE_URL,
                            ReceiptHandle=message['ReceiptHandle']
                        )
                        logging.info(f"Deleted failed message: {message['MessageId']}")
                    except Exception as delete_error:
                        logging.error(f"Failed to delete failed message: {delete_error}")
                        
                finally:
                    set_termination_protection(False)
        else:
            logging.info(f"No messages found (attempt {attempt + 1}/{max_empty_attempts})")
            if attempt < max_empty_attempts - 1:
                logging.info(f"Waiting {wait_time} seconds before next attempt...")

    logging.info("No more messages to process. Shutting down...")
    shutdown_instance()

def shutdown_instance():
    """
    Safely shuts down the EC2 instance after work is done.
    """
    logging.info("Shutting down EC2 instance.")
    try:
        ec2.terminate_instances(InstanceIds=[INSTANCE_ID])
        logging.info(f"Successfully initiated termination of instance {INSTANCE_ID}")
    except Exception as e:
        logging.error(f"Failed to terminate instance: {e}")

def process_message(message_body):
    """
    Process the message and call process_video with extracted parameters.
    """
    try:
        message_data = json.loads(message_body)

        # Log the entire message_data
        logging.info(f"Received message data: {json.dumps(message_data, indent=4)}")
        
        # Set default value for UHD_ENABLED if not present
        uhd_enabled = message_data.get('UHD_ENABLED', False)

        logging.info(f"UHD_ENABLED: {uhd_enabled}")
        
        try:
            # Add required fields except UHD_ENABLED
            required_fields = ['S3_BUCKET', 'INPUT_PATH', 'VIDEO_TABLE']
            
            # Validate required fields
            for field in required_fields:
                if field not in message_data:
                    raise ValueError(f"Missing required field: {field}")
            
            process_video(
                message_data['S3_BUCKET'],
                message_data['INPUT_PATH'],
                message_data['VIDEO_TABLE'],  # Pass VIDEO_TABLE to process_video
                uhd_enabled,  # Pass UHD_ENABLED to process_video
                message_data.get('INCLUDE_DOWNLOAD', False)  # Pass INCLUDE_DOWNLOAD to process_video
            )     
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode message body: {e}")
            raise
    except Exception as e:
        logging.error(f"Error processing message: {e}")
        raise

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
    """Use Whisper to transcribe the audio file with optimized settings."""
    try:
        logging.info(f"Loading Whisper model and starting transcription for {local_file_path}")
        
        # Use medium model for initial fast pass
        # The medium model is ~4x faster than large with ~1% lower accuracy
        model = whisper.load_model("medium").cuda()
        
        # Optimize model for inference
        model.eval()
        with torch.inference_mode():
            # Use efficient attention
            model.encoder.use_flash_attention = True if hasattr(model.encoder, 'use_flash_attention') else False
            
            # Optimize batch size and chunk size for GPU memory
            result = model.transcribe(
                local_file_path,
                word_timestamps=True,
                batch_size=16,        # Increase batch size for faster processing
                compute_type="float16" # Use half precision for faster processing
            )

        # Save the transcription
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        logging.info(f"Saved transcription to {output_file}")

    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise

def get_frame_rate(input_file):
    # Use ffprobe to extract the frame rate
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
        input_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get frame rate: {result.stderr}")
    
    # Parse the frame rate (e.g., "30000/1001" or "25")
    frame_rate_str = result.stdout.strip()
    if '/' in frame_rate_str:
        num, denom = map(int, frame_rate_str.split('/'))
        frame_rate = num / denom
    else:
        frame_rate = float(frame_rate_str)
    
    return frame_rate

def process_video(s3_bucket, input_path, video_table, uhd_enabled, include_download=False):
    """
    Process the video using the provided parameters.
    """
    input_key = os.path.basename(input_path)
    update_progress(input_key, 0, video_table)  # Starting progress

    try:
        # Use the parameters as needed in your process_video logic
        logging.info(f"Starting video processing for bucket: {s3_bucket}, key: {input_key}")

        # Define paths within the bucket
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        transcoding_path = f"s3://{s3_bucket}/transcoding_samples/"
        transcription_path = f"s3://{s3_bucket}/transcriptions/{base_name}.json"
        download_path = f"s3://{s3_bucket}/downloads/{os.path.basename(input_path)}"

        local_input = download_from_s3(input_path)
        update_progress(input_key, 10, video_table)

        # Get video metadata
        metadata = get_video_metadata(local_input)

        # Debug metadata
        print("Available metadata:", json.dumps(metadata, indent=2))
        
        # Calculate aspect ratio
        width = int(metadata["width"])
        height = int(metadata["height"])
        actual_ratio = width / height
        target_ratio = 16 / 9
        
        if abs(actual_ratio - target_ratio) > 0.01:
            raise ValueError(f"Input video is not in 16:9 aspect ratio.")
        
        print(f"Aspect ratio check passed: {width}x{height} = {actual_ratio:.3f} ≈ 16:9")

        # Create temporary working directory
        work_dir = f"/tmp"
        os.makedirs(work_dir, exist_ok=True)
        
        try:
            # Process audio separately (high quality)
            audio_output = f"{work_dir}/{base_name}-WAV.mp4"

            # Check if audio file exists and has content
            if os.path.exists(audio_output) and os.path.getsize(audio_output) > 0:
                logging.info(f"Skipping audio extraction, file already exists: {audio_output}")
            else:
                logging.info(f"Starting audio extraction to: {audio_output}")
                audio_cmd = (
                    f'ffmpeg -y -hwaccel cuda -c:v h264_cuvid '
                    f'-i {local_input} '
                    f'-vn '
                    f'-c:a aac -b:a 256k '
                    f'{audio_output}'
                )
                try:
                    subprocess.run(audio_cmd, shell=True, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    logging.error(f"FFmpeg command failed:\nOutput: {e.output}\nError: {e.stderr}")
                    raise
                logging.info("Audio extraction completed")

            # Transcribe the audio file
            transcription_output = f"{work_dir}/{base_name}.json"
            if os.path.exists(transcription_output) and os.path.getsize(transcription_output) > 0:
                logging.info(f"Skipping transcription, file already exists: {transcription_output}")
            else:
                logging.info("Starting transcription")
                transcribe_audio(audio_output, transcription_output)

            try:
                 # Upload transcription to S3
                upload_to_s3(transcription_output, transcription_path)
                # Delete the audio output file after successful upload
                if os.path.exists(audio_output):
                    os.remove(audio_output)
                    os.remove(transcription_output)
                    logging.info("Deleted audio output file: %s", audio_output)
            except Exception as e:
                logging.error("S3 upload failed: %s", e)
                raise e
         
            update_progress(input_key, 20, video_table)
            
            # Define video variants with vbr_hq for good quality and bandwidth
            variants = [
                {
                    "name": "UHD-H264",
                    "size": "3840x2160",
                    "video_codec": "h264_nvenc",
                    "video_opts": "-preset slow -rc vbr_hq -b:v 12M -cq:v 20 -qmin 19 -qmax 26 -maxrate 15M -bufsize 30M -profile:v main",
                    "bitrate": "12M",  # Estimated bandwidth
                    "segment_duration": 10,
                    "codec": "avc1.640028",
                    "audio_opts": "-c:a aac -b:a 256k"
                },
                {
                    "name": "1080P-H264",
                    "size": "1920x1080",
                    "video_codec": "h264_nvenc",
                    "video_opts": "-preset slow -rc vbr_hq -b:v 8M -cq:v 21 -qmin 19 -qmax 28 -maxrate 10M -bufsize 20M -profile:v main",
                    "bitrate": "8M",  # Estimated bandwidth 
                    "segment_duration": 8,
                    "codec": "avc1.640028",
                    "audio_opts": "-c:a aac -b:a 256k"
                },
                {
                    "name": "720P-H264",
                    "size": "1280x720",
                    "video_codec": "h264_nvenc",
                    "video_opts": "-preset slow -rc vbr_hq -b:v 4M -cq:v 23 -qmin 20 -qmax 30 -maxrate 5M -bufsize 10M -profile:v main",
                    "bitrate": "4M",  # Estimated bandwidth
                    "segment_duration": 6,
                    "codec": "avc1.64001f",
                    "audio_opts": "-c:a aac -b:a 128k"
                },
                {
                    "name": "540P-H264",
                    "size": "960x540",
                    "video_codec": "h264_nvenc",
                    "video_opts": "-preset slow -rc vbr_hq -b:v 2M -cq:v 24 -qmin 21 -qmax 32 -maxrate 3M -bufsize 6M -profile:v main",
                    "bitrate": "2M",  # Estimated bandwidth
                    "segment_duration": 6,
                    "codec": "avc1.64001f",
                    "audio_opts": "-c:a aac -b:a 128k"
                }
            ]

            # Filter variants based on input resolution and H.265 support
            filtered_variants = [
                v for v in variants
                if int(v['size'].split('x')[0]) <= width and int(v['size'].split('x')[1]) <= height
            ]

            # If UHD is not enabled, remove UHD variants
            if not uhd_enabled:
                filtered_variants = [v for v in filtered_variants if "UHD" not in v['name']]

            # Get the frame rate of the input video
            frame_rate = get_frame_rate(local_input)
            keyframe_interval = int(frame_rate * 2)  # 2-second interval
            playlist_file = os.path.join(work_dir, f"{base_name}.m3u8") # Create the master playlist file

            # Create HLS segments and playlists for each variant
            variant_playlists = []
            m3u8_playlists = []
            total_variants = len(filtered_variants)
            for idx, variant in enumerate(filtered_variants):
                variant_playlist = f"{work_dir}/{base_name}-{variant['name']}.m3u8"
                variant_playlist_m3u8 = f"{base_name}-{variant['name']}.m3u8"
                variant_playlists.append(variant_playlist)
                m3u8_playlists.append(variant_playlist_m3u8)
                
                # Calculate progress for this variant
                variant_progress = 20 + \
                                 ((90 - 20) * 
                                  (idx + 1) / total_variants)
                
                logging.info(f"Processing variant {idx + 1}/{total_variants}: {variant['name']}")
                cmd = (
                    f'ffmpeg -y -i {local_input} '
                    f'-c:v {variant["video_codec"]} {variant["video_opts"]} '
                    f'-vf scale={variant["size"].replace("x", ":")} '
                    f'{variant["audio_opts"]} '
                    f'-g {keyframe_interval} -keyint_min {keyframe_interval} '
                    f'-sc_threshold 40 '
                    f'-f hls '
                    f'-hls_time {variant["segment_duration"]} '
                    f'-hls_playlist_type vod '
                    f'-hls_segment_filename "{work_dir}/{base_name}-{variant["name"]}-%03d.ts" '
                    f'{variant_playlist}'
                )
                try:
                    subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    logging.error(f"FFmpeg command failed:\nOutput: {e.output}\nError: {e.stderr}")
                    raise
                
                # Update progress after each variant is processed
                update_progress(input_key, int(variant_progress), video_table)
                logging.info(f"Completed variant {variant['name']} ({idx + 1}/{total_variants})")

            # Create the master playlist
            master_playlist_file = f"{work_dir}/{base_name}.m3u8"
            non_uhd_variants = [v for v in filtered_variants if "UHD" not in v['name']]
            create_master_playlist(master_playlist_file, non_uhd_variants, m3u8_playlists, frame_rate, base_name)

            # Create an additional UHD playlist if UHD_ENABLED is true
            if uhd_enabled:
                uhd_variants = [v for v in filtered_variants if "UHD" in v['name']]
                uhd_playlist_file = f"{work_dir}/{base_name}-UHD.m3u8"
                create_master_playlist(uhd_playlist_file, uhd_variants + non_uhd_variants, m3u8_playlists, frame_rate, base_name)

            # Send status event: Playlist creation completed
            update_progress(input_key, 90, video_table)
       
            # Upload or delete the local input based on INCLUDE_DOWNLOAD
            if include_download:
                logging.info(f"Uploading local input to downloads folder: {download_path}")
                upload_to_s3(local_input, download_path)
            else:
                logging.info(f"Deleting local input: {local_input}")
                os.remove(local_input)

            # Upload all segments and manifests to S3
            for root, _, files in os.walk(work_dir):
                for file in files:
                    local_file = os.path.join(root, file)
                    s3_key = f"{transcoding_path}{file}"
                    upload_to_s3(local_file, s3_key)

            # Send status event: Upload completed
            update_progress(input_key, 100, video_table)

            # Clean up
            subprocess.run(f'rm -rf {work_dir}/*', shell=True)

            # Send success event
            update_progress(input_key, 100, video_table)
            
        except Exception as e:
            error_message = str(e)
            update_progress(input_key, 0, video_table, error_message=error_message)
            logging.error(f"Failed to process video {input_key}: {e}")
            raise

    except Exception as e:
        error_message = str(e)
        update_progress(input_key, 0, video_table, error_message=error_message)
        logging.error(f"Failed to process video {input_key}: {e}")
        raise

def update_progress(video_id, percent_complete, table_name, error_message=None):
    """
    Update video processing progress in DynamoDB.
    Creates item if it doesn't exist, updates if it does.
    """
    try:
        current_time = datetime.utcnow().isoformat()
        
        # Try to get the existing item first
        try:
            response = dynamodb.get_item(
                TableName=table_name,
                Key={
                    'FileName': {'S': video_id}
                }
            )
            
            # If item exists, update it
            if 'Item' in response:
                update_expression = "SET ProcessingStatus = :status, UpdatedAt = :time"
                expression_values = {
                    ':status': {'S': str(percent_complete) if not error_message else "ERROR"},
                    ':time': {'S': current_time}
                }
                
                if error_message:
                    update_expression += ", ErrorMessage = :error"
                    expression_values[':error'] = {'S': error_message}

                dynamodb.update_item(
                    TableName=table_name,
                    Key={
                        'FileName': {'S': video_id}
                    },
                    UpdateExpression=update_expression,
                    ExpressionAttributeValues=expression_values
                )
            else:
                # Item doesn't exist, create new item
                item = {
                    'FileName': {'S': video_id},
                    'ProcessingStatus': {'S': "ERROR" if error_message else str(percent_complete)},
                    'CreatedDate': {'S': current_time},
                    'UpdatedAt': {'S': current_time}
                }
                if error_message:
                    item['ErrorMessage'] = {'S': error_message}
                
                dynamodb.put_item(
                    TableName=table_name,
                    Item=item
                )

        except dynamodb.exceptions.ResourceNotFoundException:
            logging.error(f"Table {table_name} not found")
            raise
            
        logging.info(f"Updated progress for {video_id}: {percent_complete}%")
        if error_message:
            logging.info(f"Logged error for {video_id}: {error_message}")
            
    except Exception as e:
        logging.error(f"Failed to update progress: {e}")

def create_master_playlist(file_path, variants, m3u8_playlists, frame_rate, base_name):
    with open(file_path, 'w') as f:
        f.write("#EXTM3U\n")
        f.write("#EXT-X-VERSION:6\n")
        f.write("#EXT-X-INDEPENDENT-SEGMENTS\n")
        
        for variant, variant_playlist_m3u8 in zip(variants, m3u8_playlists):
            numeric_bitrate = variant["bitrate"].replace("M", "000")
            average_bandwidth = int(numeric_bitrate) // 2
            combined_codecs = f'{variant["codec"]},mp4a.40.2'

            f.write(f'#EXT-X-STREAM-INF:BANDWIDTH={numeric_bitrate},AVERAGE-BANDWIDTH={average_bandwidth},RESOLUTION={variant["size"]},CODECS="{combined_codecs}",FRAME-RATE={frame_rate},CLOSED-CAPTIONS=NONE\n')
            f.write(f'{variant_playlist_m3u8}\n')

def str2bool(val):
    return str(val).lower() in ("yes", "true", "1")

def log_gpu_usage():
    try:
        result = subprocess.run('nvidia-smi', shell=True, capture_output=True, text=True)
        logging.info(f"GPU Usage:\n{result.stdout}")
    except Exception as e:
        logging.error(f"Failed to get GPU usage: {e}")

if __name__ == "__main__":
    optimize_gpu()
    s3_bucket = os.environ["S3_BUCKET"]
    input_path = os.environ["INPUT_PATH"]
    video_table = os.environ["VIDEO_TABLE"]
    uhd_enabled = str2bool(os.environ.get("UHD_ENABLED", "false"))
    include_download = str2bool(os.environ.get("INCLUDE_DOWNLOAD", "false"))

    logging.info("FFmpeg GPU capabilities:")
    subprocess.run('ffmpeg -hide_banner -hwaccels', shell=True, check=True)

    process_video(s3_bucket, input_path, video_table, uhd_enabled, include_download)