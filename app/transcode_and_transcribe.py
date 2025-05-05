import os
import subprocess
import boto3
import json
from datetime import datetime
import whisper
import logging
import torch
import mimetypes
import pathlib

region = os.getenv('AWS_REGION', 'us-west-2')  # Default to 'us-west-1' if not set

# Initialize AWS clients
s3_client = boto3.client('s3', region_name=region)
ec2 = boto3.client('ec2', region_name=region)
dynamodb = boto3.client('dynamodb', region_name='us-west-1')

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/app/whisper_transcription.log'),
        logging.StreamHandler()  # This sends logs to console
    ]
)

def get_mpeg_ts_offset(input_file: str) -> int:
    """
    Return start-PTS in 90 kHz ticks (MPEG-TS clock).
    If it cannot be detected, fall back to 0.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time",
        "-read_intervals", "%+#1",          # first packet only
        "-of", "json", input_file,
    ]
    try:
        out      = subprocess.check_output(cmd, text=True)
        packets  = json.loads(out).get("packets", [])
        pts_time = float(packets[0]["pts_time"]) if packets else 0.0
    except Exception as exc:
        logging.warning("Unable to get PTS for %s (%s). Using 0.", input_file, exc)
        pts_time = 0.0

    return int(round(pts_time * 90_000))

def optimize_gpu():
    if torch.cuda.is_available():
        # Enable TF32 for faster processing on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
        torch.cuda.empty_cache()

def download_from_s3(s3_path):
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])
    local_path = f"/tmp/{os.path.basename(key)}"
    
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, local_path)
    return local_path

_CONTENT_TYPES = {
     ".m3u8": "application/vnd.apple.mpegurl",
     ".vtt":  "text/vtt",
    ".ts":   "video/mp2t",                # MPEG-TS segments (H.264)
    ".m4s":  "video/iso.segment",         # CMAF/fMP4 segments (H.265)
    ".mp4":  "video/mp4"
 }

def upload_to_s3(local_path: str, s3_url: str, public: bool = True):
    bucket, *key_parts = s3_url.replace("s3://", "").split("/")
    key = "/".join(key_parts)

    # Pick MIME type: explicit map first, then fallback to Python's guess‑table
    ext = pathlib.Path(local_path).suffix.lower()
    content_type = _CONTENT_TYPES.get(ext) or mimetypes.guess_type(local_path)[0] \
                   or "binary/octet-stream"

    extra = {"ContentType": content_type}

    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key, ExtraArgs=extra)

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

def transcribe_audio(local_file_path: str,
                     output_file: str,
                     offset_ticks: int = 0):
    """Transcribe with Whisper and embed `mpegts_offset`."""
    try:
        logging.info(f"Loading Whisper model and starting transcription for {local_file_path}")
        
        # Log GPU usage before loading the model
        log_gpu_usage()
        
        # Load the model
        model = whisper.load_model("medium").cuda()
        
        # Log GPU usage after loading the model
        log_gpu_usage()

        result  = model.transcribe(local_file_path, word_timestamps=True)
        result["mpegts_offset"] = offset_ticks

        log_gpu_usage()

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

def get_audio_bitrate(input_file):
    # Use ffprobe to extract the audio bitrate
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=bit_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get audio bitrate: {result.stderr}")
    return int(result.stdout.strip())

def process_video(s3_bucket, input_path, video_table, uhd_enabled, include_download=False):
    """
    Process the video using the provided parameters.
    """
    input_key = os.path.basename(input_path)
    update_progress(input_key, 0, video_table)  # Starting progress

    try:
        # Log GPU usage before starting video processing
        log_gpu_usage()

        # Use the parameters as needed in your process_video logic
        logging.info(f"Starting video processing for bucket: {s3_bucket}, key: {input_key}")

        # Define paths within the bucket
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        transcoding_path = f"s3://{s3_bucket}/transcoding_samples/"
        transcription_path = f"s3://{s3_bucket}/transcriptions/{base_name}.json"
        download_path   = f"s3://{s3_bucket}/downloads/{os.path.basename(input_path)}"
        originals_path  = f"s3://{s3_bucket}/originals/{os.path.basename(input_path)}"

        # ---------------- obtain source ----------------
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

        # Check if resolution is too small
        min_width, min_height = 1920, 1080  # Minimum resolution for 1080p
        if width < min_width or height < min_height:
            error_message = "LOW RESOLUTION! Resolution is less than 1080p."
            logging.error(error_message)
            update_progress(input_key, 10, video_table, error_message=error_message)

        # Create temporary working directory
        work_dir = f"/tmp"
        os.makedirs(work_dir, exist_ok=True)
        
        try:
            # Check audio fidelity
            original_bitrate = get_audio_bitrate(local_input)
            logging.info(f"Original audio bitrate: {original_bitrate} bits per second")
            audio_bitrate = min(original_bitrate, 256000)  # Use the lower of the original or 256K

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
                    f'-c:a aac -b:a {audio_bitrate} '
                    f'{audio_output}'
                )
                try:
                    subprocess.run(audio_cmd, shell=True, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    logging.error(f"FFmpeg command failed:\nOutput: {e.output}\nError: {e.stderr}")
                    raise
                logging.info("Audio extraction completed")

            # Log GPU usage after audio extraction
            log_gpu_usage()
            
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
                },
                {
                    "name": "360P-H264",
                    "size": "640x360",
                    "video_codec": "h264_nvenc",
                    "video_opts": "-preset slow -rc vbr_hq -b:v 1M -cq:v 25 -qmin 22 -qmax 34 -maxrate 1.5M -bufsize 3M -profile:v main",
                    "bitrate": "1M",  # Estimated bandwidth
                    "segment_duration": 4,
                    "codec": "avc1.64001e",
                    "audio_opts": "-c:a aac -b:a 96k"
                },

                # ---------- NEW HEVC / H265 fMP4 VARIANTS ----------
                {
                    "name": "UHD-H265",
                    "size": "3840x2160",
                    "video_codec": "hevc_nvenc",
                    "video_opts": "-preset slow -rc vbr_hq -b:v 8M  -cq:v 23 -maxrate 10M -bufsize 30M -profile:v main10",
                    "bitrate": "8M",
                    "segment_duration": 10,
                    "codec": "hvc1.2.4.L153.B0",
                    "audio_opts": "-c:a aac -b:a 256k",
                    "fmp4": True
                },
                {
                    "name": "1080P-H265",
                    "size": "1920x1080",
                    "video_codec": "hevc_nvenc",
                    "video_opts": "-preset slow -rc vbr_hq -b:v 5M  -cq:v 25 -maxrate 6M -bufsize 20M -profile:v main10",
                    "bitrate": "5M",
                    "segment_duration": 8,
                    "codec": "hvc1.2.4.L123.B0",
                    "audio_opts": "-c:a aac -b:a 192k",
                    "fmp4": True
                },
                {
                    "name": "720P-H265",
                    "size": "1280x720",
                    "video_codec": "hevc_nvenc",
                    "video_opts": "-preset slow -rc vbr_hq -b:v 3M  -cq:v 26 -maxrate 4M -bufsize 12M -profile:v main10",
                    "bitrate": "3M",
                    "segment_duration": 6,
                    "codec": "hvc1.2.4.L120.B0",
                    "audio_opts": "-c:a aac -b:a 128k",
                    "fmp4": True
                }
            ]

            # Filter variants based on input resolution and H.265 support
            filtered_variants = [
                v for v in variants
                if int(v['size'].split('x')[0]) <= width and int(v['size'].split('x')[1]) <= height
            ]

            # If no variants are available, log an error and update the database
            if not filtered_variants:
                error_message = "No suitable video variants for processing due to low resolution."
                logging.error(error_message)
                update_progress(input_key, 0, video_table, error_message=error_message)
                raise ValueError(error_message)

            # If UHD is not enabled, remove UHD variants
            if not uhd_enabled:
                filtered_variants = [v for v in filtered_variants if "UHD" not in v['name']]

            # Get the frame rate of the input video
            frame_rate = get_frame_rate(local_input)
            keyframe_interval = int(frame_rate * 2)  # 2-second interval

            # Create HLS segments and playlists for each variant
            total_variants = len(filtered_variants)
            for idx, variant in enumerate(filtered_variants):
                variant_playlist = f"{work_dir}/{base_name}-{variant['name']}.m3u8"
       
                # Calculate progress for this variant
                variant_progress = 20 + \
                                 ((90 - 20) * 
                                  (idx + 1) / total_variants)
                
                logging.info(f"Processing variant {idx + 1}/{total_variants}: {variant['name']}")
                # --- H.264 uses TS, H.265 uses CMAF/fMP4 ---
                seg_ext  = "m4s" if variant.get("fmp4") else "ts"
                seg_flag = "-hls_segment_type fmp4 " \
                           f'-init_seg_name "{base_name}-{variant["name"]}-init.mp4" ' \
                           if variant.get("fmp4") else ""

                cmd = (
                    f'ffmpeg -y -i {local_input} '
                    f'-c:v {variant["video_codec"]} {variant["video_opts"]} '
                    f'-vf scale={variant["size"].replace("x", ":")} '
                    f'{variant["audio_opts"]} '
                    f'-g {keyframe_interval} -keyint_min {keyframe_interval} '
                    f'-sc_threshold 40 '
                    f'-f hls '
                    f'-hls_time {variant["segment_duration"]} '
                    f'-hls_playlist_type vod ' \
                    f'{seg_flag}' \
                    f'-hls_segment_filename "{work_dir}/{base_name}-{variant["name"]}-%03d.{seg_ext}" '
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

            # Log GPU usage after video processing
            log_gpu_usage()

            update_progress(input_key, 70, video_table)

            probe_seg    = f"{work_dir}/{base_name}-360P-H264-000.ts"
            offset_ticks = get_mpeg_ts_offset(probe_seg)

             # Transcribe the audio file
            transcription_output = f"{work_dir}/{base_name}.json"
            if os.path.exists(transcription_output) and os.path.getsize(transcription_output) > 0:
                logging.info(f"Skipping transcription, file already exists: {transcription_output}")
            else:
                logging.info("Starting transcription")
                transcribe_audio(audio_output, transcription_output, offset_ticks)

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
         
            update_progress(input_key, 80, video_table)

            # Create the master playlist
            master_playlist_file = f"{work_dir}/{base_name}.m3u8"
            non_uhd_variants = [v for v in filtered_variants if "UHD" not in v['name']]
            create_master_playlist(master_playlist_file, non_uhd_variants, frame_rate, base_name)

            # Create an additional UHD playlist if UHD_ENABLED is true
            if uhd_enabled:
                uhd_variants = [v for v in filtered_variants if "UHD" in v['name']]
                uhd_playlist_file = f"{work_dir}/{base_name}-UHD.m3u8"
                create_master_playlist(uhd_playlist_file, uhd_variants + non_uhd_variants, frame_rate, base_name)

            # Send status event: Playlist creation completed
            update_progress(input_key, 90, video_table)
       
            # --- Preserve original uploaded file ---
            logging.info(f"Uploading original to originals folder: {originals_path}")
            try:
                upload_to_s3(local_input, originals_path)
            except Exception as e:
                logging.warning(f"Failed to copy original to originals/: {e}")

            # Optional download-friendly 720p MP4 rendition
            if include_download:
                download_file = f"{work_dir}/{base_name}.mp4"

                if not os.path.exists(download_file):
                    logging.info("Creating 720P MP4 for download: %s", download_file)
                    dl_cmd = (
                        f'ffmpeg -y -i {local_input} '
                        f'-c:v h264_nvenc -preset slow -rc vbr_hq -b:v 4M -vf scale=1280:720 '
                        f'-c:a aac -b:a 128k {download_file}'
                    )
                    try:
                        subprocess.run(dl_cmd, shell=True, check=True, capture_output=True, text=True)
                    except subprocess.CalledProcessError as e:
                        logging.error("FFmpeg (download) failed:\nOutput: %s\nError: %s", e.output, e.stderr)
                        raise

                logging.info("Uploading 720P MP4 to downloads folder: %s", download_path)
                upload_to_s3(download_file, download_path)

            # Always delete the local input file after processing
            logging.info(f"Deleting local input file: {local_input}")
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
                existing_error = response['Item'].get('ErrorMessage', {}).get('S', '')
                low_res_warning = "LOW RESOLUTION! Resolution is less than 1080p."
                new_error_message = existing_error

                # Preserve the low-resolution warning
                if low_res_warning in existing_error:
                    new_error_message = existing_error
                elif error_message:
                    new_error_message = error_message

                # Determine the processing status
                processing_status = "ERROR" if error_message and error_message != low_res_warning else str(percent_complete)

                update_expression = "SET ProcessingStatus = :status, UpdatedAt = :time"
                expression_values = {
                    ':status': {'S': processing_status},
                    ':time': {'S': current_time}
                }
                
                if new_error_message:
                    update_expression += ", ErrorMessage = :error"
                    expression_values[':error'] = {'S': new_error_message}

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
                    'ProcessingStatus': {'S': "ERROR" if error_message and error_message != low_res_warning else str(percent_complete)},
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

def create_master_playlist(file_path, variants, frame_rate, base_name):
    """
    Writes:
      • master playlist  (file_path)
      • {base_name}_{lang}.m3u8 subtitle playlists (same dir as master)

    Assumes the WebVTT files live at ../subtitles/{base_name}_{lang}.vtt
    relative to the master playlist.
    """
    master_dir = os.path.dirname(file_path) or "."

    # If any variant uses fMP4 (HEVC) bump playlist version to 7 per HLS spec.
    has_fmp4 = any(v.get("fmp4") for v in variants)

    master_lines = [
        "#EXTM3U",
        f"#EXT-X-VERSION:{7 if has_fmp4 else 6}",
        "#EXT-X-INDEPENDENT-SEGMENTS",
        ""
    ]

    subtitles = [("English", "en", True), ("Español", "es", False)]
    target_duration = 3600  # long enough for whole video

    for name, lang, is_default in subtitles:
        default_flag = "YES" if is_default else "NO"
        vtt_path     = f"../subtitles/{base_name}_{lang}.vtt"
        sub_m3u8     = f"{base_name}_{lang}.m3u8"
        sub_m3u8_path = os.path.join(master_dir, sub_m3u8)

        # ---- write subtitle playlist (.m3u8) ----
        with open(sub_m3u8_path, "w") as sf:
            sf.write("#EXTM3U\n")
            sf.write("#EXT-X-VERSION:6\n")
            sf.write(f"#EXT-X-TARGETDURATION:{target_duration}\n")
            sf.write(f'#EXT-X-MAP:URI="{vtt_path}"\n')
            sf.write(f"#EXTINF:{target_duration:.3f},\n")
            sf.write(f"{vtt_path}\n")
            sf.write("#EXT-X-ENDLIST\n")

        # ---- add track reference to master ----
        master_lines.append(
            f'#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="subs",'
            f'NAME="{name}",LANGUAGE="{lang}",DEFAULT={default_flag},'
            f'AUTOSELECT=YES,FORCED=NO,URI="{sub_m3u8}"'
        )

    # ---------- add video variants ----------
    for variant in variants:
        # Derive the per-variant media playlist name from the variant name
        variant_playlist = f"{base_name}-{variant['name']}.m3u8"
        bps = int(variant["bitrate"].rstrip("M")) * 1_000_000
        avg = int(bps * 0.8)
        codecs = f'{variant["codec"]},mp4a.40.2'

        master_lines.append(
            f'#EXT-X-STREAM-INF:BANDWIDTH={bps},AVERAGE-BANDWIDTH={avg},'
            f'RESOLUTION={variant["size"]},CODECS="{codecs}",'
            f'FRAME-RATE={frame_rate:.3f},CLOSED-CAPTIONS=NONE,'
            'SUBTITLES="subs"'
        )
        master_lines.append(variant_playlist)

    # ---------- write master playlist ----------
    with open(file_path, "w") as f:
        f.write("\n".join(master_lines))
        f.write("\n")

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