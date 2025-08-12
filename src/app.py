import streamlit as st
from transcribe import transcribe_audio, format_transcription_segments, check_cuda_availability


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

                # Display transcription text
                st.subheader("📝 Transcription")
                st.text_area(
                    "Full Text",
                    value=result['text'],
                    height=200,
                    help="Complete transcription of the audio file"
                )

                # Display timestamped segments
                if result.get('segments'):
                    st.subheader("⏱️ Timestamped Segments")
                    formatted_segments = format_transcription_segments(
                        result['segments'])
                    st.text_area(
                        "Segments with Timestamps",
                        value=formatted_segments,
                        height=300,
                        help="Transcription broken down by time segments"
                    )

                # Download button for transcription
                st.download_button(
                    label="💾 Download Transcription",
                    data=result['text'],
                    file_name=f"{uploaded_file.name}_transcription.txt",
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
