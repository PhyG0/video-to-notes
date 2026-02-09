
import gradio as gr
import os
import tempfile

# Add local 'bin' folder to PATH for portability
base_dir = os.path.dirname(os.path.abspath(__file__))
bin_dir = os.path.join(base_dir, "bin")
if os.path.exists(bin_dir):
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]

# Disable symlinks for HuggingFace Hub to avoid WinError 1314
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from src.audio import extract_audio, cleanup_audio
from src.transcriber import transcribe_audio
from src.formatter import save_transcript

# Global state to store transcription results for AI processing
transcription_state = {
    "segments": None,
    "transcript_path": None
}

def transcribe_video(video_file, whisper_model, device, progress=gr.Progress()):
    """Transcription only - no AI processing."""
    global transcription_state
    
    if video_file is None:
        return "Please upload a video file.", "", None
    
    video_path = video_file.name if hasattr(video_file, 'name') else video_file
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    output_dir = tempfile.mkdtemp()
    transcript_path = os.path.join(output_dir, f"{base_name}.md")
    temp_audio_path = os.path.join(output_dir, f"{base_name}_temp.wav")
    
    logs = []
    transcript_text = ""
    
    try:
        progress(0.1, desc="Extracting audio...")
        logs.append("Extracting audio from video...")
        audio_file = extract_audio(video_path, temp_audio_path)
        logs.append("Audio extracted successfully.")
        
        progress(0.3, desc="Transcribing audio...")
        logs.append(f"Loading Whisper model '{whisper_model}' on {device}...")
        
        compute_type = "float16" if device == "cuda" else "int8"
        segments = transcribe_audio(
            audio_file, 
            model_size=whisper_model, 
            device=device,
            compute_type=compute_type
        )
        
        all_segments = list(segments)
        logs.append(f"Transcription complete. {len(all_segments)} segments found.")
        
        progress(0.8, desc="Saving transcript...")
        save_transcript(all_segments, transcript_path, format="markdown")
        
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
        
        # Store for AI processing
        transcription_state["segments"] = all_segments
        transcription_state["transcript_path"] = transcript_path
        
        logs.append("Transcript saved successfully!")
        progress(1.0, desc="Done!")
        
    except Exception as e:
        logs.append(f"Error: {str(e)}")
        import traceback
        logs.append(traceback.format_exc())
    finally:
        if os.path.exists(temp_audio_path):
            cleanup_audio(temp_audio_path)
    
    log_output = "\n".join(logs)
    transcript_download = transcript_path if os.path.exists(transcript_path) else None
    
    return log_output, transcript_text, transcript_download


def generate_ai_notes(ai_model, progress=gr.Progress()):
    """Generate AI notes from existing transcription."""
    global transcription_state
    
    if transcription_state["segments"] is None:
        return "Please transcribe a video first.", "", None
    
    all_segments = transcription_state["segments"]
    transcript_path = transcription_state["transcript_path"]
    
    output_dir = os.path.dirname(transcript_path)
    base_name = os.path.splitext(os.path.basename(transcript_path))[0]
    notes_path = os.path.join(output_dir, f"{base_name}_notes.md")
    
    logs = []
    notes_text = ""
    
    try:
        progress(0.1, desc="Checking Ollama...")
        logs.append(f"Generating AI notes using model '{ai_model}'...")
        
        from src.ai import generate_tutorial_notes, check_ollama_server
        
        if not check_ollama_server():
            logs.append("Error: Ollama server is not reachable.")
            logs.append("Make sure you have Ollama installed and run 'ollama serve'.")
            return "\n".join(logs), "Error: Ollama server not available.", None
        
        progress(0.3, desc="Generating notes...")
        full_text = "\n".join([f"[{s['start']}-{s['end']}] {s['text']}" for s in all_segments])
        notes_text = generate_tutorial_notes(full_text, model=ai_model)
        
        progress(0.9, desc="Saving notes...")
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(notes_text)
        
        logs.append("AI notes generated successfully!")
        progress(1.0, desc="Done!")
        
    except Exception as e:
        logs.append(f"Error: {str(e)}")
        import traceback
        logs.append(traceback.format_exc())
    
    log_output = "\n".join(logs)
    notes_download = notes_path if os.path.exists(notes_path) else None
    
    return log_output, notes_text, notes_download


# Build UI
with gr.Blocks(title="Video to Notes", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎥 Video to Notes
    Upload a video to extract a timestamped transcript, then optionally generate AI notes.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload & Settings")
            video_input = gr.File(label="Upload Video", file_types=[".mp4", ".mkv", ".avi", ".mov"])
            
            whisper_model = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large-v2"],
                value="small",
                label="Whisper Model"
            )
            device = gr.Radio(
                choices=["cuda", "cpu"],
                value="cpu",
                label="Device"
            )
            
            transcribe_btn = gr.Button("1️⃣ Transcribe Video", variant="primary")
            
            gr.Markdown("---")
            gr.Markdown("### AI Options (Requires Ollama)")
            ai_model = gr.Dropdown(
                choices=["llama3", "mistral", "gemma2"],
                value="llama3",
                label="Ollama Model"
            )
            ai_btn = gr.Button("2️⃣ Generate AI Notes", variant="secondary")
        
        with gr.Column(scale=2):
            gr.Markdown("### Output")
            log_output = gr.Textbox(label="Logs", lines=6, interactive=False)
            
            with gr.Tabs():
                with gr.TabItem("Transcript"):
                    transcript_output = gr.Markdown(label="Transcript")
                    transcript_download = gr.File(label="Download Transcript")
                
                with gr.TabItem("AI Notes"):
                    notes_output = gr.Markdown(label="AI Notes")
                    notes_download = gr.File(label="Download Notes")
    
    # Event handlers
    transcribe_btn.click(
        fn=transcribe_video,
        inputs=[video_input, whisper_model, device],
        outputs=[log_output, transcript_output, transcript_download]
    )
    
    ai_btn.click(
        fn=generate_ai_notes,
        inputs=[ai_model],
        outputs=[log_output, notes_output, notes_download]
    )


if __name__ == "__main__":
    demo.queue()  # Enable queue to prevent timeouts
    demo.launch(inbrowser=True)
