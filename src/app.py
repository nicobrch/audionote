import streamlit as st
from transcribe import transcribe_audio, check_cuda_availability
from formatting import create_formatted_transcription, check_ollama_availability


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="AudioNote",
        page_icon="🎵",
        layout="wide"
    )

    # Sidebar for audio upload
    with st.sidebar:
        st.header("Audio Upload")

        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['mp3', 'wav', 'ogg', 'm4a', 'flac'],
            help="Upload an audio file to get started"
        )

        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            st.write(f"File size: {uploaded_file.size / (1024*1024):.2f} MB")

            # Transcription settings
            st.header("Transcription Settings")

            # Display CUDA availability
            cuda_available, device_info = check_cuda_availability()
            if cuda_available:
                st.success(f"🚀 GPU Available: {device_info}")
            else:
                st.info(f"💻 Using CPU: {device_info}")

            # Display Ollama availability
            ollama_available, ollama_status = check_ollama_availability()
            if ollama_available:
                st.success(f"🤖 AI Formatting: Available")
            else:
                st.warning(f"🤖 AI Formatting: Unavailable")
                st.caption(ollama_status)

            model_size = st.selectbox(
                "Whisper Model Size",
                options=["tiny", "base", "small", "medium", "large", "turbo"],
                index=1,  # Default to "base"
                help="Larger models are more accurate but slower"
            )

            language = st.selectbox(
                "Language",
                options=["auto", "en", "es", "fr", "de",
                         "it", "pt", "ru", "ja", "ko", "zh"],
                index=0,  # Default to "auto"
                help="Select language or use auto-detection"
            )

            # Transcribe button
            if st.button("🎯 Transcribe Audio", type="primary"):
                st.session_state.start_transcription = True
                st.session_state.model_size = model_size
                st.session_state.language = language

    # Main content area
    st.title("🎵 AudioNote")
    st.write(
        "Welcome to AudioNote! Upload an audio file using the sidebar to get started.")

    # Main content based on upload status
    if uploaded_file is not None:
        # Initialize session state for transcription
        if 'transcription_result' not in st.session_state:
            st.session_state.transcription_result = None
        if 'start_transcription' not in st.session_state:
            st.session_state.start_transcription = False

        # Handle transcription
        if st.session_state.get('start_transcription', False):
            with st.container():
                # Perform transcription
                result = transcribe_audio(
                    uploaded_file,
                    model_size=st.session_state.get('model_size', 'base'),
                    language=st.session_state.get('language', 'auto')
                )

                st.session_state.transcription_result = result
                st.session_state.start_transcription = False

        # Display transcription results
        if st.session_state.transcription_result:
            result = st.session_state.transcription_result

            if result['success']:
                st.success("✅ Transcription completed successfully!")

                # Display metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Model Used", result['model_used'])
                with col2:
                    st.metric("Detected Language", result['language'].upper())
                with col3:
                    st.metric("Device", result.get('device_used', 'N/A'))
                with col4:
                    st.metric("Segments", len(result.get('segments', [])))

                # Process transcription with AI formatting
                if 'formatted_transcription' not in st.session_state or st.session_state.get('current_result_id') != id(result):
                    # Check Ollama availability
                    ollama_available, ollama_status = check_ollama_availability()

                    if ollama_available:
                        formatted_text = create_formatted_transcription(
                            result['text'],
                            result.get('segments', [])
                        )
                        st.session_state.formatted_transcription = formatted_text
                        st.session_state.current_result_id = id(result)
                    else:
                        st.warning(
                            f"⚠️ AI formatting unavailable: {ollama_status}")
                        st.session_state.formatted_transcription = result['text']
                        st.session_state.current_result_id = id(result)

                # Display formatted transcription
                st.subheader("📝 AI-Processed Transcription")
                formatted_text = st.session_state.get(
                    'formatted_transcription', result['text'])

                st.text_area(
                    "Formatted Transcription",
                    value=formatted_text,
                    height=400,
                    help="AI-processed transcription with corrections, formatting, and strategic timestamps"
                )

                # Download button for formatted transcription
                st.download_button(
                    label="💾 Download Formatted Transcription",
                    data=formatted_text,
                    file_name=f"{uploaded_file.name}_formatted_transcription.txt",
                    mime="text/plain"
                )

            else:
                st.error(
                    f"❌ Transcription failed: {result.get('error', 'Unknown error')}")

        else:
            st.info(
                "Click the 'Transcribe Audio' button in the sidebar to start transcription.")

    else:
        st.info("Please upload an audio file from the sidebar to begin.")


if __name__ == "__main__":
    main()
