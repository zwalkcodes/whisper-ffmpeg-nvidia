import os
import subprocess
import boto3
import json
from datetime import datetime
import whisper
import logging
import requests

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
    visibility_timeout = 1800  # 30min - should be longer than max processing time

    for attempt in range(max_empty_attempts):
        logging.info(f"Polling attempt {attempt + 1}/{max_empty_attempts}")
        
        # Try to get a message with visibility timeout
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=wait_time,
            VisibilityTimeout=visibility_timeout,
            AttributeNames=['All']
        )

        if 'Messages' in response:
            message = response['Messages'][0]
            receipt_handle = message['ReceiptHandle']
            try:
                logging.info(f"Processing message: {message['MessageId']}")
                set_termination_protection(True)
                
                # Process the message
                process_message(message['Body'])

                # Delete the message only after successful processing
                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=receipt_handle
                )
                logging.info(f"Successfully processed and deleted message: {message['MessageId']}")
                set_termination_protection(False)
                
                # Continue polling for more messages
                continue
                
            except Exception as e:
                logging.error(f"Error processing message: {e}")
                set_termination_protection(False)
                
                # Delete the message even if processing failed
                # This prevents infinite retry loops
                try:
                    sqs.delete_message(
                        QueueUrl=QUEUE_URL,
                        ReceiptHandle=receipt_handle
                    )
                    logging.info(f"Deleted failed message: {message['MessageId']}")
                except Exception as delete_error:
                    logging.error(f"Failed to delete message: {delete_error}")
                
                raise
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
                uhd_enabled  # Pass UHD_ENABLED to process_video
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
    """Use Whisper to transcribe the audio file."""
    try:
        logging.info(f"Loading Whisper model and starting transcription for {local_file_path}")
        model = whisper.load_model("large").cuda()
        result = model.transcribe(local_file_path, word_timestamps=True)

        # Save the transcription as a JSON file
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

def process_video(s3_bucket, input_path, video_table, uhd_enabled):
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
        
        # Define progress steps
        progress_steps = {
            'DOWNLOAD_COMPLETED': 10,
            'AUDIO_PROCESSING_COMPLETED': 20,
            'VIDEO_PROCESSING_COMPLETED': 70,
            'PLAYLIST_CREATION_COMPLETED': 90,
            'UPLOAD_COMPLETED': 100
        }

        local_input = download_from_s3(input_path)
        update_progress(input_key, progress_steps['DOWNLOAD_COMPLETED'], video_table)

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
            raise ValueError(f"Input video is not in 16:9 aspect ratio. Got {actual_ratio:.3f}, expected {target_ratio:.3f}")
        
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
                subprocess.run(audio_cmd, shell=True, check=True)
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
         
            update_progress(input_key, progress_steps['AUDIO_PROCESSING_COMPLETED'], video_table)
            
            # Define video variants with only H.264
            variants = [
                {
                    "name": "UHD-H264",
                    "size": "3840x2160",
                    "video_codec": "h264_nvenc",
                    "video_opts": "-preset slow -rc vbr_hq -qmin 0 -qmax 28 -b:v 12M -profile:v main",
                    "bitrate": "12000k",
                    'codec': 'avc1.640028',
                    "audio_opts": "-c:a aac -b:a 192k"
                },
                {
                    "name": "1080P-H264",
                    "size": "1920x1080",
                    "video_codec": "h264_nvenc",
                    "video_opts": "-preset slow -rc vbr_hq -qmin 0 -qmax 28 -b:v 6M -profile:v main",
                    "bitrate": "6000k",
                    'codec': 'avc1.640028',
                    "audio_opts": "-c:a aac -b:a 192k"
                },
                {
                    "name": "720P-H264",
                    "size": "1280x720",
                    "video_codec": "h264_nvenc",
                    "video_opts": "-preset slow -rc vbr_hq -qmin 0 -qmax 28 -b:v 4M -profile:v main",
                    "bitrate": "4000k",
                    'codec': 'avc1.64001f',
                    "audio_opts": "-c:a aac -b:a 192k"
                },
                {
                    "name": "540P-H264",
                    "size": "960x540",
                    "video_codec": "h264_nvenc",
                    "video_opts": "-preset slow -rc vbr_hq -qmin 0 -qmax 28 -b:v 2M -profile:v main",
                    "bitrate": "2500k",
                    'codec': 'avc1.64001f',
                    "audio_opts": "-c:a aac -b:a 192k"
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
                variant_progress = progress_steps['AUDIO_PROCESSING_COMPLETED'] + \
                                 ((progress_steps['PLAYLIST_CREATION_COMPLETED'] - progress_steps['AUDIO_PROCESSING_COMPLETED']) * 
                                  (idx + 1) / total_variants)
                
                logging.info(f"Processing variant {idx + 1}/{total_variants}: {variant['name']}")
                cmd = (
                    f'ffmpeg -y -i {local_input} '
                    f'-c:v {variant["video_codec"]} {variant["video_opts"]} '
                    f'-maxrate {variant["bitrate"]} -bufsize {int(1.5 * int(variant["bitrate"].replace("k", "000")))} '
                    f'-vf scale={variant["size"].replace("x", ":")} '
                    f'{variant["audio_opts"]} '
                    f'-g {keyframe_interval} -keyint_min {keyframe_interval} '
                    f'-sc_threshold 0 '
                    f'-f hls '
                    f'-hls_time 6 '
                    f'-hls_playlist_type vod '
                    f'-hls_segment_filename "{work_dir}/{base_name}-{variant["name"]}-%03d.ts" '
                    f'{variant_playlist}'
                )
                subprocess.run(cmd, shell=True, check=True)
                
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
            update_progress(input_key, progress_steps['PLAYLIST_CREATION_COMPLETED'], video_table)
       
            # Delete the input video file
            os.remove(local_input)

            # Upload all segments and manifests to S3
            for root, _, files in os.walk(work_dir):
                for file in files:
                    local_file = os.path.join(root, file)
                    s3_key = f"{transcoding_path}{file}"
                    upload_to_s3(local_file, s3_key)

            # Send status event: Upload completed
            update_progress(input_key, progress_steps['UPLOAD_COMPLETED'], video_table)

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
    Update video processing progress in DynamoDB, including error messages.
    If the item does not exist, it will be created.
    """
    try:
        # Define the item to be put or updated
        item = {
            'FileName': {'S': video_id},
            'ProcessingStatus': {'S': str(percent_complete)},
            'updatedAt': {'S': datetime.utcnow().isoformat()},
            'CreatedDate': {'S': datetime.utcnow().isoformat()}
        }

        if error_message:
            item['ErrorMessage'] = {'S': error_message}

        # Use PutItem to create or replace the item
        dynamodb.put_item(
            TableName=table_name,
            Item=item
        )
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
            numeric_bitrate = variant["bitrate"].replace("k", "000")
            average_bandwidth = int(numeric_bitrate) // 2
            combined_codecs = f'{variant["codec"]},mp4a.40.2'

            f.write(f'#EXT-X-STREAM-INF:BANDWIDTH={numeric_bitrate},AVERAGE-BANDWIDTH={average_bandwidth},RESOLUTION={variant["size"]},CODECS="{combined_codecs}",FRAME-RATE={frame_rate},CLOSED-CAPTIONS=NONE\n')
            f.write(f'{variant_playlist_m3u8}\n')

if __name__ == "__main__":
    poll_queue()