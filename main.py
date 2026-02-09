
import argparse
import os
import sys
from src.audio import extract_audio, cleanup_audio
from src.transcriber import transcribe_audio
from src.formatter import save_transcript

def main():
    parser = argparse.ArgumentParser(description="Extract audio from video and transcribe it using Whisper.")
    parser.add_argument("input_video", help="Path to the input video file (MP4).")
    parser.add_argument("--output", "-o", help="Path to save the output transcript (default: <video_name>.md).")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device to use for transcription (default: cuda).")
    parser.add_argument("--model", "-m", default="medium", help="Whisper model size (default: medium).")
    parser.add_argument("--format", "-f", choices=["markdown", "json"], default="markdown", help="Output format (default: markdown).")
    parser.add_argument("--keep-audio", action="store_true", help="Keep the extracted audio file after transcription.")
    
    args = parser.parse_args()
    
    video_path = args.input_video
    if not os.path.isfile(video_path):
        print(f"Error:Input file '{video_path}' does not exist.")
        sys.exit(1)
        
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = args.output if args.output else f"{base_name}.md"
    temp_audio_path = f"{base_name}_temp.wav"
    
    print(f"Processing '{video_path}'...")
    
    try:
        # Step 1: Extract Audio
        print("Extracting audio...")
        audio_file = extract_audio(video_path, temp_audio_path)
        print(f"Audio extracted to '{audio_file}'")
        
        # Step 2: Transcribe
        # Using float16 compute type for CUDA devices
        compute_type = "float16" if args.device == "cuda" else "int8"
        
        print("Starting transcription...")
        segments = transcribe_audio(
            audio_file, 
            model_size=args.model, 
            device=args.device,
            compute_type=compute_type
        )
        
        # Step 3: Save Output
        print(f"Saving transcript to '{output_path}'...")
        save_transcript(segments, output_path, format=args.format)
        
        print(f"Done! Transcript saved to '{output_path}'.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if not args.keep_audio and os.path.exists(temp_audio_path):
            print("Cleaning up temporary audio file...")
            cleanup_audio(temp_audio_path)

if __name__ == "__main__":
    # Add local 'bin' folder to PATH for portability
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bin_dir = os.path.join(base_dir, "bin")
    if os.path.exists(bin_dir):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]

    # Disable symlinks for HuggingFace Hub to avoid WinError 1314
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        
    main()
