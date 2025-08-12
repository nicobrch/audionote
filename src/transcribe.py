import whisper
import tempfile
import os
import torch
import gc
import streamlit as st


def check_cuda_availability() -> tuple[bool, str]:
    """
    Check if CUDA is available for GPU acceleration.

    Returns:
        tuple: (is_available, device_info)
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return True, f"CUDA GPU: {device_name}"
    else:
        return False, "CPU"


def unload_whisper_model():
    """
    Unload Whisper model from memory and clear GPU cache.
    """
    try:
        # Clear GPU cache if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        st.info("🗑️ Whisper model unloaded from memory")
    except Exception as e:
        st.warning(f"Warning during model cleanup: {str(e)}")


def transcribe_audio(
    audio_file,
    model_size: str = "base",
    language: str = "auto"
) -> dict:
    """
    Transcribe an audio file using OpenAI Whisper.

    Args:
        audio_file: Streamlit uploaded file object or file path
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large', 'turbo')
        language: Language code ('es', 'en', 'auto' for automatic detection)

    Returns:
        dict: Transcription result containing text and metadata
    """
    # Validate model size
    valid_models = ["tiny", "base", "small", "medium", "large", "turbo"]
    if model_size not in valid_models:
        raise ValueError(f"Model size must be one of: {valid_models}")

    # Validate language
    valid_languages = ["auto", "es", "en", "fr",
                       "de", "it", "pt", "ru", "ja", "ko", "zh"]
    if language not in valid_languages:
        raise ValueError(f"Language must be one of: {valid_languages}")

    try:
        # Check CUDA availability
        cuda_available, device_info = check_cuda_availability()
        device = "cuda" if cuda_available else "cpu"

        # Load the Whisper model
        with st.spinner(f"Loading Whisper model ({model_size}) on {device_info}..."):
            model = whisper.load_model(model_size, device=device)

        try:
            # Handle Streamlit uploaded file
            if hasattr(audio_file, 'read'):
                # Create a temporary file to save the uploaded audio
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    temp_file.write(audio_file.read())
                    temp_file_path = temp_file.name

                try:
                    # Transcribe the audio
                    with st.spinner("Transcribing audio..."):
                        if language == "auto":
                            result = model.transcribe(temp_file_path)
                        else:
                            result = model.transcribe(
                                temp_file_path, language=language)

                    return {
                        "text": result["text"],
                        "language": result.get("language", language),
                        "segments": result.get("segments", []),
                        "model_used": model_size,
                        "device_used": device_info,
                        "success": True
                    }

                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

            else:
                # Handle file path
                with st.spinner("Transcribing audio..."):
                    if language == "auto":
                        result = model.transcribe(audio_file)
                    else:
                        result = model.transcribe(
                            audio_file, language=language)

                return {
                    "text": result["text"],
                    "language": result.get("language", language),
                    "segments": result.get("segments", []),
                    "model_used": model_size,
                    "device_used": device_info,
                    "success": True
                }

        finally:
            # Always unload Whisper model from memory
            if 'model' in locals():
                del model
            unload_whisper_model()

    except Exception as e:
        return {
            "text": "",
            "language": language,
            "segments": [],
            "model_used": model_size,
            "device_used": "N/A",
            "success": False,
            "error": str(e)
        }


def format_transcription_segments(segments: list) -> str:
    """
    Format transcription segments with timestamps.

    Args:
        segments: List of segment dictionaries from Whisper

    Returns:
        str: Formatted transcription with timestamps
    """
    if not segments:
        return ""

    formatted_text = []
    for segment in segments:
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        text = segment.get('text', '').strip()

        # Format timestamp as MM:SS
        start_min, start_sec = divmod(int(start_time), 60)
        end_min, end_sec = divmod(int(end_time), 60)

        timestamp = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
        formatted_text.append(f"{timestamp} {text}")

    return "\n".join(formatted_text)
