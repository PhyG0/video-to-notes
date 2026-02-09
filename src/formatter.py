
import json

def save_transcript(transcript_generator, output_path: str, format: str = "markdown"):
    """
    Saves the transcript to a file in the specified format.
    
    Args:
        transcript_generator: Generator yielding transcript segments.
        output_path: Path to save the output file.
        format: Output format ('markdown' or 'json').
    """
    if format == "json":
        data = list(transcript_generator)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    else:
        # Default to Markdown
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Video Transcript\n\n")
            for segment in transcript_generator:
                line = f"**{segment['start']} - {segment['end']}**: {segment['text']}\n"
                f.write(line)
                # Determine if we should flush to disk periodically? 
                # For now, writing line by line is safer for long generations if we crash.
                f.flush() 
