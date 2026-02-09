
import ollama
import time

def generate_tutorial_notes(transcript_text, model="llama3"):
    """
    Generates detailed tutorial notes from a transcript using Ollama.
    
    Args:
        transcript_text (str): The full text of the transcript.
        model (str): The Ollama model to use.
        
    Returns:
        str: The generated tutorial notes in Markdown format.
    """
    # Chunking strategy: 
    # For a 4-hour video, the transcript will be huge. 
    # We need to split it based on token count or character count.
    # Llama 3 8B has 8k context. approx 24k characters safely.
    # Let's be conservative with 15k characters per chunk to allow room for the prompt and output.
    
    chunk_size = 15000
    overlap = 500
    
    chunks = []
    start = 0
    while start < len(transcript_text):
        end = start + chunk_size
        chunk = transcript_text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap # Move forward with overlap

    print(f"Split transcript into {len(chunks)} chunks for processing.")
    
    final_notes = "# Detailed Video Tutorial\n\n"
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        
        prompt = f"""
You are a technical documenter. Your task is to convert the following video transcript segment into a detailed Step-by-Step Tutorial Guide.

**Rules:**
1. RETAIN ALL DETAILS: Do not summarize into vague points. Capture every click, command, setting, and code snippet.
2. STRUCTURE: Use clear headings (##), bullet points, and code blocks.
3. CONTEXT: If a step is a continuation from the previous part, continue logically.
4. NO FLUFF: Remove conversational filler (e.g., "Um", "So guys", "Welcome back"). Keep it strictly instructional.

**Transcript Segment:**
{chunk}

**Detailed Tutorial:**
"""
        try:
            response = ollama.chat(model=model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            
            content = response['message']['content']
            final_notes += f"\n\n## Part {i+1}\n\n{content}"
            
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            final_notes += f"\n\n## Part {i+1} (Error)\n\n[Failed to generate notes for this section: {e}]\n"

    return final_notes

def check_ollama_server():
    """Checks if Ollama server is running."""
    try:
        # Simple check - list models to see if we can connect
        ollama.list()
        return True
    except Exception:
        return False
