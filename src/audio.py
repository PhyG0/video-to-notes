
import os
import subprocess
import ffmpeg

def extract_audio(video_path: str, output_path: str = "temp_audio.wav") -> str:
    """
    Extracts audio from a video file using ffmpeg-python.
    Converts to 16kHz mono WAV which is optimal for Whisper.
    
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the extracted audio.
        
    Returns:
        str: Path to the extracted audio file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_path, ac=1, ar='16000') 
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
        raise

def cleanup_audio(audio_path: str):
    """Removes the temporary audio file."""
    if os.path.exists(audio_path):
        os.remove(audio_path)
