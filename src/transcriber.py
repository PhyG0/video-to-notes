
import os
from faster_whisper import WhisperModel
from datetime import timedelta

def format_timestamp(seconds: float) -> str:
    """Formats seconds into MM:SS string."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    minutes, remainder = divmod(total_seconds, 60)
    return f"{minutes:02d}:{remainder:02d}"

def transcribe_audio(audio_path: str, model_size: str = "medium", device: str = "cuda", compute_type: str = "float16"):
    """
    Transcribes audio file using faster-whisper.
    
    Args:
        audio_path (str): Path to the audio file.
        model_size (str): Whisper model size (tiny, base, small, medium, large-v2).
        device (str): Device to run the model on ('cuda' or 'cpu').
        compute_type (str): Quantization type ('float16', 'int8_float16', 'int8').
        
    Yields:
        dict: Segment data containing start_time, end_time, and text.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Loading Whisper model '{model_size}' on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print("Transcribing audio...")
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

    for segment in segments:
        yield {
            "start": format_timestamp(segment.start),
            "end": format_timestamp(segment.end),
            "text": segment.text.strip()
        }
