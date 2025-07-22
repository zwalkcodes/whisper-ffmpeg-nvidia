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
                     offset_ticks: int = 0,
                     language: str = "en",
                     initial_prompt: str = None,
                     clip_duration: int = None):
    """
    Transcribe with Whisper and embed `mpegts_offset`.
    
    Args:
        local_file_path: Path to the audio file for transcription
        output_file: Output JSON file path
        offset_ticks: MPEG-TS clock ticks offset
        language: Explicitly set language code (e.g., "en")
        initial_prompt: Optional prompt to guide the transcription
        clip_duration: If set, only transcribe first N seconds (for testing)
    """
    try:
        logging.info(f"Loading Whisper model and starting transcription for {local_file_path}")
        
        # For testing, if clip_duration is set, create a smaller clip
        input_file = local_file_path
        if clip_duration:
            test_clip = f"{os.path.dirname(local_file_path)}/test_clip_{clip_duration}s.wav"
            clip_cmd = (
                f'ffmpeg -y -i {local_file_path} '
                f'-t {clip_duration} '
                f'-c:a copy {test_clip}'
            )
            subprocess.run(clip_cmd, shell=True, check=True, capture_output=True, text=True)
            input_file = test_clip
            logging.info(f"Created {clip_duration}-second test clip for transcription testing")
        
        # Log GPU usage before loading the model
        log_gpu_usage()
        
        # Load the model
        model = whisper.load_model("medium").cuda()
        
        # Log GPU usage after loading the model
        log_gpu_usage()

        # Set up transcription options
        options = {
            "word_timestamps": True,
            "language": language,  # Explicitly set the language
        }
        
        if initial_prompt:
            options["initial_prompt"] = initial_prompt
            
        result = model.transcribe(input_file, **options)
        result["mpegts_offset"] = offset_ticks

        log_gpu_usage()

        # Save the transcription
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        logging.info(f"Saved transcription to {output_file}")

        # Clean up test clip if created
        if clip_duration and os.path.exists(test_clip):
            os.remove(test_clip)

    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise

def check_if_vfr(input_file):
    """Check if the input file has variable frame rate"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "json", input_file,
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
        data = json.loads(out)
        
        if 'streams' not in data or not data['streams']:
            return False
        
        stream = data['streams'][0]
        avg_frame_rate = stream.get('avg_frame_rate', '0/0')
        r_frame_rate = stream.get('r_frame_rate', '0/0')
        
        # Convert frame rates to float for comparison
        def rate_to_float(rate_str):
            if '/' in rate_str:
                num, denom = map(int, rate_str.split('/'))
                return num / denom if denom != 0 else 0
            return float(rate_str) if rate_str.strip() != '0/0' else 0
        
        avg_rate = rate_to_float(avg_frame_rate)
        r_rate = rate_to_float(r_frame_rate)
        
        # If rates differ significantly, it's likely VFR
        return abs(avg_rate - r_rate) > 0.01
    except Exception as e:
        logging.warning(f"VFR check failed: {e}. Assuming CFR.")
        return False

def remux_to_cfr(input_file, output_file):
    cmd = (
        f'ffmpeg -y -i {input_file} '
        f'-vf fps=fps=30 -c:v h264_nvenc -preset fast '
        f'-vsync cfr -c:a copy {output_file}'
    )
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logging.info(f"Successfully remuxed to CFR: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to remux to CFR: {e.stderr}")
        # Fall back to using the original file
        return False

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
            warning_message = f"WARNING: Input video is not in 16:9 aspect ratio ({width}x{height} = {actual_ratio:.3f}). Will add pillar boxing/letter boxing as needed."
            logging.warning(warning_message)
            print(warning_message)
            # Save warning to database but continue processing
            update_progress(input_key, 10, video_table, error_message=warning_message)
            # Continue processing instead of raising an error
        else:
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
            # Check if source has Variable Frame Rate (VFR)
            is_vfr = check_if_vfr(local_input)
            if is_vfr:
                logging.info("Detected Variable Frame Rate (VFR) source. Will re-mux to constant frame rate.")
                fixed_input = f"{work_dir}/{base_name}-cfr.mp4"
                remux_to_cfr(local_input, fixed_input)
                # Use the fixed input for subsequent processing
                source_file = fixed_input
            else:
                source_file = local_input

            # Extract audio in optimal format for transcription (PCM 16-bit, 16kHz, mono)
            audio_output_wav = f"{work_dir}/{base_name}-transcription.wav"
            
            # Check if audio file exists and has content
            if os.path.exists(audio_output_wav) and os.path.getsize(audio_output_wav) > 0:
                logging.info(f"Skipping audio extraction, file already exists: {audio_output_wav}")
            else:
                logging.info(f"Starting audio extraction to: {audio_output_wav}")
                audio_cmd = (
                    f'ffmpeg -y -i {source_file} '
                    f'-vn -ac 1 -ar 16000 -sample_fmt s16 '  # Mono, 16kHz, 16-bit PCM
                    f'-acodec pcm_s16le {audio_output_wav}'
                )
                try:
                    subprocess.run(audio_cmd, shell=True, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    logging.error(f"FFmpeg command failed:\nOutput: {e.output}\nError: {e.stderr}")
                    raise
                logging.info("Audio extraction for transcription completed")

            # Also extract high-quality audio for streaming (if not already done)
            audio_output_aac = f"{work_dir}/{base_name}-AAC.mp4"
            # Check audio fidelity
            original_bitrate = get_audio_bitrate(source_file)
            logging.info(f"Original audio bitrate: {original_bitrate} bits per second")
            audio_bitrate = min(original_bitrate, 256000)  # Use the lower of the original or 256K
            
            if os.path.exists(audio_output_aac) and os.path.getsize(audio_output_aac) > 0:
                logging.info(f"Skipping AAC audio extraction, file already exists: {audio_output_aac}")
            else:
                logging.info(f"Starting AAC audio extraction to: {audio_output_aac}")
                aac_cmd = (
                    f'ffmpeg -y -hwaccel cuda -c:v h264_cuvid '
                    f'-i {source_file} '
                    f'-vn '
                    f'-c:a aac -b:a {audio_bitrate} '
                    f'{audio_output_aac}'
                )
                try:
                    subprocess.run(aac_cmd, shell=True, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    logging.error(f"FFmpeg command failed:\nOutput: {e.output}\nError: {e.stderr}")
                    raise
                logging.info("AAC audio extraction completed")

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
            frame_rate = get_frame_rate(source_file)
            # keyframe_interval = int(frame_rate * 2)  # 2-second interval

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

                seg_len = variant['segment_duration']          # 4, 6, 8 or 10
                keyint  = int(frame_rate * seg_len)            # 30 fps → 120/180/240/300

                hls_flags = 'independent_segments'
                if variant.get("fmp4"):
                    hls_flags += '+split_by_time'              # needed for fMP4/CMAF
                    hls_flags += '+program_date_time'

                cmd = (
                    f'ffmpeg -y -avoid_negative_ts make_zero -start_at_zero -i {source_file} '
                    f'-c:v {variant["video_codec"]} {variant["video_opts"]} '
                    f'-vf scale={variant["size"].replace("x", ":")} '
                    f'{variant["audio_opts"]} '
                    f'-g {keyint} -keyint_min {keyint} '
                    f'-sc_threshold 0 '                            # disable scene-cut I-frames
                    f'-force_key_frames "expr:gte(t,n_forced*{seg_len})" '  # key-frame EXACTLY every seg_len
                    f'-hls_flags {hls_flags} '
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
                # Use the WAV file optimized for transcription
                transcribe_audio(audio_output_wav, transcription_output, offset_ticks)

            try:
                 # Upload transcription to S3
                upload_to_s3(transcription_output, transcription_path)
                # Delete the audio output files after successful upload
                if os.path.exists(audio_output_wav):
                    os.remove(audio_output_wav)
                if os.path.exists(audio_output_aac):
                    os.remove(audio_output_aac)
                if os.path.exists(transcription_output):
                    os.remove(transcription_output)
                logging.info("Deleted audio output files")
            except Exception as e:
                logging.error("S3 upload failed: %s", e)
                raise e
         
            update_progress(input_key, 80, video_table)

            # Separate H.264 and H.265 variants for playlist creation
            h264_variants = [v for v in filtered_variants if "H264" in v['name']]
            h265_variants = [v for v in filtered_variants if "H265" in v['name']]
            
            # BACKWARD COMPATIBILITY: Main playlist remains H.264 only (same as before)
            master_playlist_file = f"{work_dir}/{base_name}.m3u8"
            non_uhd_h264_variants = [v for v in h264_variants if "UHD" not in v['name']]
            create_master_playlist(master_playlist_file, non_uhd_h264_variants, frame_rate, base_name, hls_version=6)
            
            # Create H.265 playlist for newer devices
            if h265_variants:
                h265_playlist_file = f"{work_dir}/{base_name}-hevc.m3u8"
                non_uhd_h265_variants = [v for v in h265_variants if "UHD" not in v['name']]
                create_master_playlist(h265_playlist_file, non_uhd_h265_variants, frame_rate, base_name, hls_version=7)
            
            # Create combined modern playlist with both codecs for auto-switching
            modern_playlist_file = f"{work_dir}/{base_name}-auto.m3u8"
            combined_non_uhd = [v for v in filtered_variants if "UHD" not in v['name']]
            create_master_playlist(modern_playlist_file, combined_non_uhd, frame_rate, base_name)
            
            # Create UHD playlists if enabled
            if uhd_enabled:
                # H.264 UHD playlist
                uhd_h264_playlist_file = f"{work_dir}/{base_name}-UHD.m3u8"
                uhd_h264_variants = [v for v in h264_variants if "UHD" in v['name']] + non_uhd_h264_variants
                create_master_playlist(uhd_h264_playlist_file, uhd_h264_variants, frame_rate, base_name, hls_version=6)
                
                # H.265 UHD playlist (if variants exist)
                if h265_variants:
                    uhd_h265_playlist_file = f"{work_dir}/{base_name}-UHD-hevc.m3u8"
                    uhd_h265_variants = [v for v in h265_variants if "UHD" in v['name']] + non_uhd_h265_variants
                    create_master_playlist(uhd_h265_playlist_file, uhd_h265_variants, frame_rate, base_name, hls_version=7)

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
                # Encode to a temp file (avoid in-place overwrite) but keep final S3 name
                download_file = f"{work_dir}/{base_name}-720p.mp4"
                logging.info("Creating 720P MP4 for download: %s", download_file)
                dl_cmd = (
                    f'ffmpeg -y -i {source_file} '
                    f'-c:v h264_nvenc -preset slow -rc vbr_hq -b:v 4M -vf scale=1280:720 '
                    f'-c:a aac -b:a 128k {download_file}'
                )
                try:
                    subprocess.run(dl_cmd, shell=True, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    logging.error("FFmpeg (download) failed:\nOutput: %s\nError: %s", e.output, e.stderr)
                    raise

                logging.info("Uploading 720P MP4 to downloads folder (will overwrite if exists): %s", download_path)
                upload_to_s3(download_file, download_path)

            # Always delete the local input file after processing
            logging.info(f"Deleting local input file: {local_input}")
            os.remove(local_input)
            
            # Remove the fixed CFR file if it was created
            if is_vfr and os.path.exists(fixed_input):
                logging.info(f"Deleting fixed CFR file: {fixed_input}")
                os.remove(fixed_input)

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

def create_master_playlist(file_path, variants, frame_rate, base_name, hls_version=None):
    """
    Writes:
      • master playlist  (file_path)
      • {base_name}_{lang}.m3u8 subtitle playlists (same dir as master)

    Args:
        file_path: Path to write the master playlist
        variants: List of variant dictionaries
        frame_rate: Frame rate of the video
        base_name: Base name for file paths
        hls_version: Explicit HLS version to use (if None, will be determined automatically)
    
    Assumes the WebVTT files live at ../subtitles/{base_name}_{lang}.vtt
    relative to the master playlist.
    """
    master_dir = os.path.dirname(file_path) or "."

    # If any variant uses fMP4 (HEVC) bump playlist version to 7 per HLS spec.
    has_fmp4 = any(v.get("fmp4") for v in variants)
    
    # Allow overriding HLS version explicitly
    version = hls_version if hls_version is not None else (7 if has_fmp4 else 6)

    master_lines = [
        "#EXTM3U",
        f"#EXT-X-VERSION:{version}",
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