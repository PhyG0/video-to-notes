
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

# Global state
transcription_state = {
    "segments": None,
    "transcript_path": None
}


def get_ollama_models():
    """Fetch available models from Ollama."""
    try:
        import ollama
        response = ollama.list()
        # Handle different response formats
        if hasattr(response, 'models'):
            models = response.models
        elif isinstance(response, dict) and 'models' in response:
            models = response['models']
        else:
            models = []
        
        model_names = []
        for m in models:
            if hasattr(m, 'model'):
                name = m.model.split(':')[0]
            elif isinstance(m, dict) and 'name' in m:
                name = m['name'].split(':')[0]
            elif isinstance(m, dict) and 'model' in m:
                name = m['model'].split(':')[0]
            else:
                continue
            model_names.append(name)
        
        return model_names if model_names else ["No models installed"]
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return ["Ollama not running"]


def refresh_models():
    """Refresh the Ollama model list."""
    models = get_ollama_models()
    return gr.update(choices=models, value=models[0] if models else None)


def transcribe_video(video_file, whisper_model, device, progress=gr.Progress()):
    """Transcription only."""
    global transcription_state
    
    if video_file is None:
        return "❌ Please upload a video file first.", "", None, ""
    
    video_path = video_file.name if hasattr(video_file, 'name') else video_file
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    output_dir = tempfile.mkdtemp()
    transcript_path = os.path.join(output_dir, f"{base_name}.md")
    temp_audio_path = os.path.join(output_dir, f"{base_name}_temp.wav")
    
    logs = []
    transcript_text = ""
    status = ""
    
    try:
        status = f"🔧 Model: {whisper_model} | Device: {device.upper()}"
        
        progress(0.1, desc="Extracting audio...")
        logs.append("📦 Extracting audio from video...")
        audio_file = extract_audio(video_path, temp_audio_path)
        logs.append("✅ Audio extracted.")
        
        progress(0.3, desc=f"Loading {whisper_model} model...")
        logs.append(f"🧠 Loading Whisper '{whisper_model}' on {device}...")
        
        compute_type = "float16" if device == "cuda" else "int8"
        segments = transcribe_audio(
            audio_file, 
            model_size=whisper_model, 
            device=device,
            compute_type=compute_type
        )
        
        all_segments = list(segments)
        logs.append(f"✅ Transcribed {len(all_segments)} segments.")
        
        progress(0.8, desc="Saving...")
        save_transcript(all_segments, transcript_path, format="markdown")
        
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
        
        transcription_state["segments"] = all_segments
        transcription_state["transcript_path"] = transcript_path
        
        logs.append("✅ Done! You can now generate AI notes.")
        progress(1.0, desc="Complete!")
        status = f"✅ Transcribed with {whisper_model} on {device.upper()}"
        
    except Exception as e:
        logs.append(f"❌ Error: {str(e)}")
        status = "❌ Transcription failed"
    finally:
        if os.path.exists(temp_audio_path):
            cleanup_audio(temp_audio_path)
    
    log_output = "\n".join(logs)
    transcript_download = transcript_path if os.path.exists(transcript_path) else None
    
    return log_output, transcript_text, transcript_download, status


def generate_ai_notes(ai_model, progress=gr.Progress()):
    """Generate AI notes from transcription."""
    global transcription_state
    
    if transcription_state["segments"] is None:
        return "❌ No transcript available. Please transcribe a video first.", "", None, ""
    
    all_segments = transcription_state["segments"]
    transcript_path = transcription_state["transcript_path"]
    
    output_dir = os.path.dirname(transcript_path)
    base_name = os.path.splitext(os.path.basename(transcript_path))[0]
    notes_path = os.path.join(output_dir, f"{base_name}_notes.md")
    
    logs = []
    notes_text = ""
    status = f"🤖 Using: {ai_model}"
    
    try:
        progress(0.1, desc="Connecting to Ollama...")
        logs.append(f"🤖 Generating notes with '{ai_model}'...")
        
        from src.ai import generate_tutorial_notes, check_ollama_server
        
        if not check_ollama_server():
            logs.append("❌ Ollama server not reachable.")
            logs.append("💡 Run 'ollama serve' in a terminal.")
            return "\n".join(logs), "", None, "❌ Ollama not running"
        
        progress(0.2, desc=f"Processing with {ai_model}...")
        full_text = "\n".join([f"[{s['start']}-{s['end']}] {s['text']}" for s in all_segments])
        notes_text = generate_tutorial_notes(full_text, model=ai_model)
        
        progress(0.9, desc="Saving...")
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(notes_text)
        
        logs.append("✅ AI notes generated!")
        status = f"✅ Generated with {ai_model}"
        progress(1.0, desc="Complete!")
        
    except Exception as e:
        logs.append(f"❌ Error: {str(e)}")
        status = "❌ AI generation failed"
    
    log_output = "\n".join(logs)
    notes_download = notes_path if os.path.exists(notes_path) else None
    
    return log_output, notes_text, notes_download, status


# Custom CSS for better styling
custom_css = """
.status-box { 
    padding: 10px; 
    border-radius: 8px; 
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    font-weight: bold;
}
.step-header {
    font-size: 1.1em;
    margin-bottom: 10px;
    color: #e94560;
}
"""

# Build UI
with gr.Blocks(title="Video to Notes", theme=gr.themes.Soft(primary_hue="purple"), css=custom_css) as demo:
    gr.Markdown("""
    # 🎥 Video to Notes
    **Step 1:** Upload video → Transcribe → **Step 2:** Generate AI notes (optional)
    """)
    
    with gr.Row():
        # LEFT COLUMN - Controls
        with gr.Column(scale=1):
            # Step 1: Transcription
            gr.Markdown("### 📝 Step 1: Transcribe")
            video_input = gr.File(label="Upload Video", file_types=[".mp4", ".mkv", ".avi", ".mov"])
            
            with gr.Row():
                whisper_model = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large-v2"],
                    value="small",
                    label="Model",
                    scale=2
                )
                device = gr.Radio(
                    choices=["cuda", "cpu"],
                    value="cpu",
                    label="Device",
                    scale=1
                )
            
            transcribe_btn = gr.Button("▶️ Transcribe", variant="primary", size="lg")
            transcribe_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
            
            gr.Markdown("---")
            
            # Step 2: AI Notes
            gr.Markdown("### 🤖 Step 2: AI Notes")
            with gr.Row():
                ai_model = gr.Dropdown(
                    choices=get_ollama_models(),
                    label="Ollama Model",
                    scale=3
                )
                refresh_btn = gr.Button("🔄", scale=1, size="sm")
            
            ai_btn = gr.Button("✨ Generate Notes", variant="secondary", size="lg")
            ai_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
        
        # RIGHT COLUMN - Output
        with gr.Column(scale=2):
            log_output = gr.Textbox(label="📋 Logs", lines=5, interactive=False)
            
            with gr.Tabs():
                with gr.TabItem("📄 Transcript"):
                    transcript_output = gr.Markdown()
                    transcript_download = gr.File(label="Download")
                
                with gr.TabItem("📝 AI Notes"):
                    notes_output = gr.Markdown()
                    notes_download = gr.File(label="Download")
    
    # Event handlers
    refresh_btn.click(fn=refresh_models, outputs=[ai_model])
    
    transcribe_btn.click(
        fn=transcribe_video,
        inputs=[video_input, whisper_model, device],
        outputs=[log_output, transcript_output, transcript_download, transcribe_status]
    )
    
    ai_btn.click(
        fn=generate_ai_notes,
        inputs=[ai_model],
        outputs=[log_output, notes_output, notes_download, ai_status]
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(inbrowser=True)
