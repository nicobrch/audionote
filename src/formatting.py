from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import streamlit as st
import gc
import torch
from typing import Dict, List


def unload_ollama_model():
    """
    Unload Ollama model from memory and clear GPU cache.
    """
    try:
        # Clear GPU cache if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        st.info("🗑️ Ollama model unloaded from memory")
    except Exception as e:
        st.warning(f"Warning during Ollama model cleanup: {str(e)}")


def create_formatted_transcription(transcription_text: str, segments: List[Dict], model_name: str = "gpt-oss:20b") -> str:
    """
    Process transcription using LangChain Ollama to create a well-formatted result.

    Args:
        transcription_text: Raw transcription text from Whisper
        segments: List of timestamped segments from Whisper
        model_name: Ollama model to use for processing

    Returns:
        str: Formatted and corrected transcription
    """
    try:
        # Initialize Ollama LLM with specified model
        llm = OllamaLLM(model=model_name)

        try:
            # Create formatted segments text for context
            formatted_segments = []
            for segment in segments:
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()

                # Format timestamp as MM:SS
                start_min, start_sec = divmod(int(start_time), 60)
                end_min, end_sec = divmod(int(end_time), 60)

                timestamp = f"[{start_min:02d}:{start_sec:02d}-{end_min:02d}:{end_sec:02d}]"
                formatted_segments.append(f"{timestamp} {text}")

            segments_text = "\n".join(formatted_segments)

            # Create the prompt template
            prompt_template = PromptTemplate(
                input_variables=["raw_transcription", "timestamped_segments"],
                template="""You are an expert transcription editor. Your task is to process the following audio transcription and create a well-formatted, coherent result.

                Please:
                1. Fix any transcription errors, typos, or incoherences
                2. Improve grammar and sentence structure while maintaining the original meaning
                3. Add proper punctuation and formatting
                4. Format the text using markdown (use **bold** for emphasis, proper headings, bullet points if needed)
                5. Organize content into logical paragraphs and sections
                6. Ensure the text flows naturally and is easy to read
                7. Preserve all important information from the original transcription
                8. DO NOT include any timestamps in the final output
                9. Maintain the original language the transcription was in

                Raw Transcription:
                {raw_transcription}

                Timestamped Segments (for reference only, do not include timestamps in output):
                {timestamped_segments}

                Please provide a clean, well-formatted markdown transcription:"""
            )

            # Format the prompt
            formatted_prompt = prompt_template.format(
                raw_transcription=transcription_text,
                timestamped_segments=segments_text
            )

            # Process with Ollama
            with st.spinner(f"🤖 Processing transcription with AI ({model_name})..."):
                formatted_result = llm.invoke(formatted_prompt)

            return formatted_result

        finally:
            # Always unload Ollama model from memory
            if 'llm' in locals():
                del llm
            unload_ollama_model()

    except Exception as e:
        st.error(f"Error processing transcription with AI: {str(e)}")
        # Fallback to basic formatting if AI processing fails
        return create_fallback_formatting(transcription_text, segments)


def create_fallback_formatting(transcription_text: str, segments: List[Dict]) -> str:
    """
    Fallback formatting method if AI processing fails.

    Args:
        transcription_text: Raw transcription text
        segments: List of timestamped segments

    Returns:
        str: Basic markdown formatted transcription
    """
    if not segments:
        return f"# Transcription\n\n{transcription_text}"

    formatted_lines = []
    formatted_lines.append("# Transcription\n")

    current_paragraph = []
    last_end_time = 0

    for i, segment in enumerate(segments):
        start_time = segment.get('start', 0)
        text = segment.get('text', '').strip()

        # Check if we should start a new paragraph (gap > 2 seconds or every 5 segments)
        if (start_time - last_end_time > 2.0) or (i > 0 and i % 5 == 0):
            if current_paragraph:
                formatted_lines.append(" ".join(current_paragraph) + "\n")
                current_paragraph = []

        current_paragraph.append(text)
        last_end_time = segment.get('end', 0)

    # Add the last paragraph
    if current_paragraph:
        formatted_lines.append(" ".join(current_paragraph))

    return "\n".join(formatted_lines)


def check_ollama_availability() -> tuple[bool, str]:
    """
    Check if Ollama service is available without loading models.

    Returns:
        tuple: (is_available, status_message)
    """
    try:
        import requests
        # Check if Ollama service is running by hitting the API
        response = requests.get(
            "http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            return True, "Ollama service available"
        else:
            return False, "Ollama service not responding"
    except Exception as e:
        return False, f"Ollama service unavailable: {str(e)}"
