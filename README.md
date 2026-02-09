
# Video-to-Notes 🎥📝

A powerful, offline tool that extracts audio from video files, transcribes it using OpenAI's Whisper model, and generates detailed step-by-step tutorial notes using a local LLM (Ollama).

## Features
- **Offline Transcription:** Uses `faster-whisper` for high-speed, local transcription.
- **AI Note Generation:** Integrates with [Ollama](https://ollama.com) to create structured, detailed guides from transcripts.
- **Portable:** Includes a local `bin` folder for FFmpeg, so no system-wide installation is required.
- **Timestamps:** Preserves precise timestamps for every segment.

## Prerequisites
1.  **Python 3.8+** installed on your system.
2.  (Optional but recommended) **NVIDIA GPU** for faster transcription.
3.  **Ollama** installed (only required if using AI note generation features).
    -   Download from [ollama.com](https://ollama.com).
    -   Run `ollama serve` and pull a model: `ollama pull llama3`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PhyG0/video-to-notes.git
    cd video-to-notes
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Basic Transcription (No AI Notes)
Extracts audio and generates a Markdown transcript with timestamps.

```bash
python main.py input_video.mp4 --model medium --device cuda
```

**Options:**
-   `--model`: `tiny`, `base`, `small`, `medium`, `large-v2` (larger = more accurate but slower).
-   `--device`: `cuda` (GPU) or `cpu`.
-   `--format`: `markdown` (default) or `json`.

### Detailed AI Tutorial Generation 🤖
Extracts audio, transcribes it, and then uses a local LLM to write a step-by-step guide.

**Pre-requisite:** Ensure Ollama is running (`ollama serve`).

```bash
python main.py input_video.mp4 --ai --ai-model llama3
```

**AI Options:**
-   `--ai`: Enables AI note generation.
-   `--ai-model`: Specifies the Ollama model to use (default: `llama3`, can use `mistral`, etc.).

## Troubleshooting

-   **"FFmpeg not found":** The tool includes a local `bin` folder. Ensure you are running the script from the root of the project directory so it can find the local binaries.
-   **"Ollama server is not reachable":** Make sure you have installed Ollama and it is running in the background.
-   **Out of Memory (CUDA):** Try using a smaller Whisper model (`--model small` or `--model base`) or switch to `--device cpu`.
