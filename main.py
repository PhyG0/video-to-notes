
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
    parser.add_argument("--ai", action="store_true", help="Generate detailed AI notes using Ollama.")
    parser.add_argument("--ai-model", default="llama3", help="Ollama model to use (default: llama3).")
    
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
        # Collect segments for AI processing
        all_segments = list(segments)
        save_transcript(all_segments, output_path, format=args.format)
        
        print(f"Done! Transcript saved to '{output_path}'.")

        # Step 4: AI Note Generation (Optional)
        if args.ai:
            from src.ai import generate_tutorial_notes, check_ollama_server
            
            print("Checking Ollama availability...")
            if not check_ollama_server():
                print("Error: Ollama server is not reachable. Is 'ollama serve' running?")
                print("Skipping AI note generation.")
            else:
                full_text = "\n".join([f"[{s['start']}-{s['end']}] {s['text']}" for s in all_segments])
                
                ai_output_path = args.output.replace(".md", "_notes.md") if args.output else f"{base_name}_notes.md"
                print(f"Generating AI notes using model '{args.ai_model}'...")
                print("This may take a while depending on the video length...")
                
                notes = generate_tutorial_notes(full_text, model=args.ai_model)
                
                with open(ai_output_path, "w", encoding="utf-8") as f:
                    f.write(notes)
                
                print(f"AI Notes saved to '{ai_output_path}'")

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
