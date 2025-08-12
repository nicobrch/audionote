import streamlit as st


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

    # Main content area
    st.title("🎵 AudioNote")
    st.write(
        "Welcome to AudioNote! Upload an audio file using the sidebar to get started.")

    # Placeholder for main content
    if uploaded_file is not None:
        st.info(
            "Audio file uploaded successfully. Main processing features will be implemented here.")
    else:
        st.info("Please upload an audio file from the sidebar to begin.")


if __name__ == "__main__":
    main()
