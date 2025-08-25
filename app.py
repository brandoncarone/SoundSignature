import streamlit as st
from openai import OpenAI
import os
import dotenv
from PIL import Image
import base64
from io import BytesIO
from typing import Optional, IO, Dict, Tuple
from audio_recorder_streamlit import audio_recorder
import select
import sys
import librosa
import librosa.display
import essentia
import essentia.standard as es
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sp
import io
from pathlib import Path
import tempfile
import madmom
from madmom.features.tempo import TempoEstimationProcessor
from madmom.features.beats import RNNBeatProcessor
import shutil
import time
import pandas as pd
from datetime import datetime
#import crema
#import basic_pitch

dotenv.load_dotenv()

# Define constants
demucs_model = "htdemucs_6s"
two_stems = None   # only separate one stems from the rest, e.g., "vocals"
mp3 = True

mp3_rate = 320
float32 = False
int24 = False
upload_dir = "uploaded_files"
output_dir = "separated_files"
extensions = ['mp3', 'wav', 'flac', 'ogg']


# Function to create a thread and run the assistant with streaming response
def run_assistant(client, conversation):
    # 1) Create (or reuse) a thread with your chat history
    thread = client.beta.threads.create(messages=conversation)

    # 2) Stream the run and render deltas as they arrive
    placeholder = st.empty()
    streamed_text = []

    with st.spinner("Comparing songs..."):
        # The stream context yields structured events
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=st.secrets["OPENAI_ASSISTANT_ID"],
        ) as stream:
            for event in stream:
                # Newer SDKs use either .type or .event; handle both safely
                etype = getattr(event, "type", None) or getattr(event, "event", None)

                # Text delta events arrive token-by-token / chunk-by-chunk
                if etype == "response.output_text.delta":
                    # event.delta holds the text fragment in newer SDKs
                    delta = getattr(event, "delta", "") or getattr(getattr(event, "data", None), "delta", "")
                    if delta:
                        streamed_text.append(delta)
                        placeholder.markdown("".join(streamed_text))
                # Some SDK builds also emit per‑message deltas
                elif etype == "thread.message.delta":
                    parts = getattr(event, "data", None)
                    if parts and hasattr(parts, "delta") and hasattr(parts.delta, "content"):
                        for c in parts.delta.content:
                            t = getattr(c, "text", None)
                            if t and hasattr(t, "value"):
                                streamed_text.append(t.value)
                                placeholder.markdown("".join(streamed_text))
                # When the response is complete
                elif etype in ("response.completed", "thread.run.completed"):
                    # We’ll break; final text will be fetched below for robustness
                    break

            # Block until the run is fully done (handles tool calls etc.)
            stream.until_done()

    # 3) Retrieve the final assistant message text (authoritative)
    msgs = client.beta.threads.messages.list(thread_id=thread.id)
    # messages are usually newest-first; grab the first assistant message
    latest_assistant_text = ""
    for m in msgs.data:
        if m.role == "assistant" and m.content:
            # concatenate any text parts (there can be multiple)
            chunks = []
            for c in m.content:
                if hasattr(c, "text") and hasattr(c.text, "value"):
                    chunks.append(c.text.value)
            latest_assistant_text = "".join(chunks).strip()
            break

    # Make sure the placeholder shows the final text (in case last chunks arrived post-loop)
    if latest_assistant_text:
        placeholder.markdown(latest_assistant_text)
    else:
        latest_assistant_text = "".join(streamed_text).strip()
        if latest_assistant_text:
            placeholder.markdown(latest_assistant_text)

    return latest_assistant_text


# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()

    return base64.b64encode(img_byte).decode('utf-8')

def save_uploaded_file(dir, uploaded_file):
    # Create a directory to save uploaded files if it doesn't exist
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Get the file details
    file_name = uploaded_file.name

    # Save the uploaded file to the specified directory
    file_path = os.path.join(dir, file_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

#    st.success(f"File saved to {file_path}")
    return file_path

def list_audio_files(upload_dir):
    return [str(file) for file in Path(upload_dir).glob("*.mp3")]

#Function to perform basic audio analysis including waveform and spectrogram
def analyze_audio(file):
    # Ensure the file exists
    if not os.path.exists(file):
        st.error(f"File not found: {file}")
        return None, None, None, None

    # Load the audio file using librosa
    y, sr = librosa.load(file)
    duration = len(y) / sr
    max_amplitude = np.max(y)
    min_amplitude = np.min(y)
    mean_amplitude = np.mean(y)

    stft = librosa.stft(y)
    magnitude_spectrogram = np.abs(stft)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pulse_clarity = np.std(onset_env)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_flux = np.sqrt(np.sum(np.diff(magnitude_spectrogram, axis=1)**2, axis=0))
    rms_energy = librosa.feature.rms(y=y)

    # Load audio using Essentia's MonoLoader
    loader = es.MonoLoader(filename=file)
    audio = loader()

    loudness_extractor = es.Loudness()
    loudness = loudness_extractor(audio)

    key_extractor = es.KeyExtractor()
    key, scale, strength = key_extractor(audio)

    return sr, duration, magnitude_spectrogram, max_amplitude, min_amplitude, mean_amplitude, tempo, beats, pulse_clarity, spectral_centroids, spectral_bandwidth, spectral_flux, rms_energy, loudness, key, scale, strength

# def analyze_audio(file):
#     # Ensure the file exists
#     if not os.path.exists(file):
#         st.error(f"File not found: {file}")
#         return None, None, None, None
#
#     # Load the audio file using librosa
#     y, sr = librosa.load(file)
#     duration = len(y) / sr
#     stft = librosa.stft(y)
#     magnitude_spectrogram = np.abs(stft)
#
#     # Replace librosa's tempo extraction with madmom
#     # Use madmom to estimate tempo
#     beat_processor = RNNBeatProcessor()(file)
#     tempo_processor = TempoEstimationProcessor(fps=100, min_bpm=65, max_bpm=200)  # Adjust fps if needed
#     madmom_tempo = tempo_processor(beat_processor)
#
#     # Extract the estimated BPM from madmom output
#     if len(madmom_tempo) > 0:
#         madmom_bpm = madmom_tempo[0][0]  # First element is the tempo in BPM
#     else:
#         madmom_bpm = None
#
#     onset_env = librosa.onset.onset_strength(y=y, sr=sr)
#     pulse_clarity = np.std(onset_env)
#     spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
#     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#     spectral_flux = np.sqrt(np.sum(np.diff(magnitude_spectrogram, axis=1) ** 2, axis=0))
#     rms_energy = librosa.feature.rms(y=y)
#
#     # Load audio using Essentia's MonoLoader
#     loader = es.MonoLoader(filename=file)
#     audio = loader()
#
#     loudness_extractor = es.Loudness()
#     loudness = loudness_extractor(audio)
#
#     key_extractor = es.KeyExtractor()
#     key, scale, strength = key_extractor(audio)
#
#     return sr, duration, magnitude_spectrogram, madmom_bpm, pulse_clarity, spectral_centroids, spectral_bandwidth, spectral_flux, rms_energy, loudness, key, scale, strength
def format_analysis_results_for_prompt(results):
    formatted_results = []
    for result in results:
        formatted_result = f"""
        Song: {result['file_name']}
        Duration: {result['duration']:.2f} seconds
        Tempo: {result['tempo']} BPM
        Pulse Clarity (std of onset strength): {result['pulse_clarity']}
        Key: {result['key']}
        Key Strength: {result['key_strength']}
        Average Spectral Centroid: {result['spectral_centroids']}
        Average Spectral Bandwidth: {result['spectral_bandwidth']}
        Spectral Flux: {result['spectral_flux']}
        Average RMS Energy: {result['rms_energy']}
        Loudness: {result['loudness']}
        """
        formatted_results.append(formatted_result)
    return "\n".join(formatted_results)

def append_message(role, content):
    st.session_state.messages.append({
        "role": role,
        "content": content
    })

def chroma(audio_path):
    y, sr = librosa.load(audio_path)
    y_441 = librosa.resample(y, orig_sr=sr, target_sr=44100)
    dcp = madmom.audio.chroma.DeepChromaProcessor()
    ml_chroma = dcp(y_441)
    fig, ax = plt.subplots(nrows=1, figsize=(18, 5))
    fig.suptitle(f"Song: {audio_path}")
    img = librosa.display.specshow(ml_chroma.T, hop_length=2048, y_axis='chroma', x_axis='time', ax=ax)
    fig.colorbar(img, ax=[ax])
    st.pyplot(fig)

def find_files(input_dir):
    out = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower().lstrip(".") in extensions:
                out.append(os.path.join(root, file))
    return out

def copy_process_streams(process: sp.Popen):
    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: Dict[int, Tuple[IO[bytes], IO[str]]] = {
        p_stdout.fileno(): (p_stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())

    while fds:
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            p_stream, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2 ** 16)
            if not raw_buf:
                fds.remove(fd)
                continue
            buf = raw_buf.decode()
            std.write(buf)
            std.flush()

def separate_single(input_file, outp):
    cmd = [sys.executable, "-m", "demucs.separate", "-o", outp, "-n", demucs_model]
    if mp3:
        cmd += ["--mp3", f"--mp3-bitrate={mp3_rate}"]
    if float32:
        cmd += ["--float32"]
    if int24:
        cmd += ["--int24"]
    if two_stems is not None:
        cmd += [f"--two-stems={two_stems}"]

    cmd.append(str(input_file))
    print("Going to separate the file:")
    print("With command: ", " ".join(cmd))
    p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    copy_process_streams(p)
    p.wait()
    if p.returncode != 0:
        print("Command failed, something went wrong.")


def display_audio_files(files):
    for file in files:
        file_label = os.path.basename(file)
        st.write(f"**{file_label}**")
        audio_bytes = open(file, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")
        with open(file, "rb") as f:
            st.download_button(
                label=f"Download {file_label}",
                data=f,
                file_name=file_label,
                mime="audio/mp3"
            )

        y, sr = librosa.load(file)
        stft = librosa.stft(y)
        mag_spectrogram = np.abs(stft)
        fig, ax = plt.subplots(figsize=(12, 6))
        spectrogram = librosa.amplitude_to_db(mag_spectrogram)
        img = librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log', ax=ax)
        ax.set_title('Spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        st.pyplot(fig)

def clear_upload_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def log_conversation(conversation):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs = []
    for message in conversation:
        logs.append([timestamp, message["role"], message["content"]])
    df = pd.DataFrame(logs, columns=["Timestamp", "Role", "Content"])

    if not os.path.isfile("conversation_log.csv"):
        df.to_csv("conversation_log.csv", mode='a', header=True, index=False)
    else:
        df.to_csv("conversation_log.csv", mode='a', header=False, index=False)

def reset_conversation():
    st.session_state.clear()
    st.session_state.conversation_reset = True
    clear_upload_dir(upload_dir)
    clear_upload_dir(output_dir)

def main():

    # --- Page Config ---
    st.set_page_config(
        page_title="🎵 What Type of Music Do You Like? 🎵",
        page_icon="🤖🎵",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Session State ---
    # Use session state to store global variables
    if 'conversation_reset' not in st.session_state:
        st.session_state.conversation_reset = False

    if 'audio_analysis_done' not in st.session_state:
        st.session_state.audio_analysis_done = False

    if 'audio_analysis_results' not in st.session_state:
        st.session_state.audio_analysis_results = []

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'processing_stems' not in st.session_state:
        st.session_state.processing_stems = False

    if 'chosen_file' not in st.session_state:
        st.session_state.chosen_file = None

    if 'chosen_file_path' not in st.session_state:
        st.session_state.chosen_file_path = None

    if 'questions_asked' not in st.session_state:
        st.session_state['questions_asked'] = {}

    # Initialize special_action
    special_action = False

    # --- Header ---
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: #6ca395; font-size: 60px;">🤖🎵 SoundSignature 💬</h1>
            <h2 style="color: #6ca395; font-size: 40px;"> <i>What Type of Music Do You Like?</i> </h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    # # --- Header ---
    # st.html("""<h1 style="text-align: center; color: #6ca395;">🤖🎵 <i>What Type of Music Do You Like?</i> 💬</h1>""")

    # # --- Side Bar ---
    # with st.sidebar:
    #     default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv(
    #         "OPENAI_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
    #     with st.popover("🔐 OpenAI API Key"):
    #         openai_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)",
    #                                        value=default_openai_api_key, type="password")
    #
    # # --- Main Content ---
    # # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
    # if openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key:
    #     st.write("#")
    #     st.warning("⬅️ Please introduce your OpenAI API Key (make sure to have funds) to continue...")
    #
    #     with st.sidebar:
    #         st.write("#")
    #         st.write("#")
    #
    # else:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_api_key)


    # --- Display previous analysis results ---
    if 'audio_analysis_results' in st.session_state:
        #st.write("## Initial Audio Analyses")
        for result in st.session_state.audio_analysis_results:
            st.write(f"**Song:** {result['file_name']}")
            st.write(f"**Duration:** {result['duration']:.2f} seconds")
            st.write(f"**Tempo:** {result['tempo']} BPM")
            st.write(f"**Pulse Clarity (std of onset strength):** {result['pulse_clarity']}")
            st.write(f"**Key:** {result['key']}")
            st.write(f"**Key Strength:** {result['key_strength']}")
            st.write(f"**Average Spectral Centroid:** {result['spectral_centroids']}")
            st.write(f"**Average Spectral Bandwidth:** {result['spectral_bandwidth']}")
            st.write(f"**Spectral Flux:** {result['spectral_flux']}")
            st.write(f"**Average RMS Energy:** {result['rms_energy']}")
            st.write(f"**Loudness:** {result['loudness']}")

            # Recreate audio widget from path (no big bytes in session)
            with open(result['path'], "rb") as fh:
                st.audio(BytesIO(fh.read()), format="audio/mp3")

            # Optionally replot spectrogram quickly (can also wrap in an expander)
            y, sr = librosa.load(result['path'])
            S = np.abs(librosa.stft(y))
            fig, ax = plt.subplots(figsize=(12, 6))
            img = librosa.display.specshow(librosa.amplitude_to_db(S), sr=sr, x_axis='time', y_axis='log', ax=ax)
            ax.set_title('Spectrogram')
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.write("---")  # Separator for each file analysis

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar model options and inputs
    with st.sidebar:

        st.divider()

        model = st.selectbox("Select a model:", [
            "gpt-4o",
            "gpt-4.1",
            "o3-mini",
            "gpt-4.5-preview",
            "o1"
        ], index=1)

        with st.popover("⚙️ Model parameters"):
            model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

        audio_response = st.toggle("Audio response", value=False)
        if audio_response:
            cols = st.columns(2)
            with cols[0]:
                tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
            with cols[1]:
                tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=1)

        model_params = {
            "model": model,
            "temperature": model_temp,
        }

        st.button(
            "🗑️ Reset conversation",
            on_click=reset_conversation,
        )

        st.divider()

        st.write("### **🖼️ Add an image:**")

        def add_image_to_messages():
            if st.session_state.uploaded_img or (
                    "camera_img" in st.session_state and st.session_state.camera_img):
                img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                img = get_image_base64(raw_img)
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{img}"}
                        }]
                    }
                )

        cols_img = st.columns(2)

        with cols_img[0]:
            with st.popover("📁 Upload"):
                st.file_uploader(
                    "Upload an image",
                    type=["png", "jpg", "jpeg"],
                    accept_multiple_files=False,
                    key="uploaded_img",
                    on_change=add_image_to_messages,
                )

        with cols_img[1]:
            with st.popover("📸 Camera"):
                activate_camera = st.checkbox("Activate camera")
                if activate_camera:
                    st.camera_input(
                        "Take a picture",
                        key="camera_img",
                        on_change=add_image_to_messages,
                    )

        # Audio Upload
        st.write("#")
        st.write("### **🎤 Speak to Me:**")

        audio_prompt = None
        if "prev_speech_hash" not in st.session_state:
            st.session_state.prev_speech_hash = None

        speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395", )
        if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
            st.session_state.prev_speech_hash = hash(speech_input)
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", speech_input),
            )

            audio_prompt = transcript.text

        st.divider()

        with st.expander("How do we analyze audio?", expanded=False):
            st.markdown("""
            ### Understanding Audio Analysis: Short-Time Fourier Transform (STFT)

            When we analyze music, we need to break down the sound into its basic components: time and frequency. One of the fundamental tools we use for this is called the **Short-Time Fourier Transform (STFT)**.

            In simple terms, the STFT is a mathematical method that allows us to "look" at the frequency content of a sound over short periods of time. Rather than treating an entire song as one block of sound, the STFT breaks the audio into tiny segments, assesses which frequencies are present and how loud they are, and then puts them together to form a **spectrogram**.

            #### What does this do for us?

            The spectrogram gives us a **visual representation** of sound, showing how the frequencies (pitch) in the music change over time. Here's an example using a track from Frank Ocean's *Nights*:

            """)

            st.image("nights.png", caption="Spectrogram of 'Nights' by Frank Ocean", use_container_width=True)

            st.markdown("""
            - On the vertical axis, you see **frequency** (Hz), which corresponds to the pitch or tone of the sound. Higher up on the graph are higher-pitched sounds.
            - On the horizontal axis, you see **time**, which represents the progression of the song.
            - The **color** represents the **intensity** or loudness of the sound at each frequency and time. Bright spots (in red) are louder, while darker spots (in blue) are quieter.

            Using this method, we can isolate specific elements of the music, like beats, melodies, or harmonic textures. For example, we can look at which parts of the song have more high-frequency content (like cymbals or high-pitched vocals) or lower-frequency elements (like bass or kick drums).

            The STFT is just one of the many powerful tools we use to transform raw audio into useful data, which can then be further analyzed to understand the music's structure, emotion, and even genre!
            """)

        with st.expander("Feature Analysis Overview", expanded=False):
            st.markdown("""
            In our analysis of music through audio features, the features are grouped into four distinct categories—Tempo and Rhythmicity, Harmony and Melody, Timbre and Texture, and Energy and Dynamics—each representing fundamental aspects of musical expression and perception.

            - **Tempo and Rhythmicity** address the temporal aspects of music, focusing on how the speed and rhythm contribute to the structure and feel of a piece.
            - **Harmony and Melody** encompass the tonal and harmonic elements, providing insights into the musical scales, keys, and their stability.
            - **Timbre and Texture** relate to the quality and color of the sound.
            - **Energy and Dynamics** involve the power and intensity of the music.

            By breaking down audio features into these categories, we can more effectively analyze and understand the complex interplay of elements that make music a rich and emotive experience.

            """)

        with st.expander("Tempo and Rhythmicity", expanded=False):
            st.markdown("""
            **Tempo**  

            - *Definition*: The speed or pace of the music, measured in beats per minute (BPM).  

            - *Importance*: Essential for determining the energy level and mood conveyed by a piece of music.

            **Pulse Clarity**  

            - *Definition*: The clarity of the rhythmic pulse or beat, indicating how distinctly the beat is perceived.  

            - *Importance*: Important for understanding the rhythm's role in the musical experience and its impact on listener engagement.
            """)

        with st.expander("Harmony and Melody", expanded=False):
            st.markdown("""
            **Key**  

            - *Definition*: The tonal center or home base of a piece of music, identified by key and mode (e.g., C major or A minor).  

            - *Importance*: Offers insights into the harmonic framework of the music, influencing the emotional and psychological response of the listener.

            **Key Strength**  

            - *Definition*: The clarity and stability of the detected musical key throughout the piece.  

            - *Importance*: Indicates the strength of the harmonic structure, which can enhance the emotional impact of the music.
            """)

        with st.expander("Timbre and Texture", expanded=False):
            st.markdown("""
            **Spectral Centroid**  

            - *Definition*: Represents the 'center of mass' of the sound spectrum, indicating the sound's perceived brightness.  

            - *Importance*: Helps in characterizing the tonal quality or color of music which affects the texture perceived by listeners.

            **Spectral Bandwidth**  

            - *Definition*: Measures the width of significant frequency bands within the sound spectrum, reflecting the sound's fullness.  

            - *Importance*: Used to assess the richness and texture of the sound, influencing how layered and lush the music feels.

            **Spectral Flux**  

            - *Definition*: The rate of change in the power spectrum, indicative of musical onsets or texture changes.  

            - *Importance*: Provides insights into the dynamic changes within the music, highlighting transitions and variations in musical phrases.
            """)

        with st.expander("Energy and Dynamics", expanded=False):
            st.markdown("""
            **RMS Energy**  

            - *Definition*: The average power of the audio signal, calculated as the root mean square of the amplitude, independent of human perception.  

            - *Importance*: Reflects the overall energy of the sound, correlating with how dynamic and powerful the music is experienced.

            **Loudness**  

            - *Definition*: A psychoacoustic measure that incorporates frequency weighting to mimic the human ear's sensitivity to different frequencies, reflecting the perceived intensity.  

            - *Importance*: Influences the listener's perception of the music's impact and energy, affecting the emotional response to the music.
            """)

    # Chat input
    user_text = st.chat_input(
        "Please start by uploading your files in the following format: 'SongTitle_ArtistName.MP3'...")
    prompt = user_text or audio_prompt
    if prompt:
        # Flag to check if a special action is requested
        special_action = False

        # Check if the user asked for stem separation
        if "stems" in prompt.lower() or st.session_state.processing_stems:
            special_action = True
            st.session_state.processing_stems = True

        # Check if the user asked for audio analysis
        if "chroma" in prompt.lower():
            special_action = True
            audio_files = list_audio_files(upload_dir)
            with st.spinner("Creating chroma plots... This may take a moment..."):
                for audio_file in audio_files:
                    chroma(audio_file)

        # Bypass the chatbot's response if a special action was requested
        if not special_action:

            # Displaying the new messages
            with st.chat_message("user"):
                st.markdown(prompt)
                append_message("user", prompt or audio_prompt)

            # Send the analysis results to the assistant
            with st.chat_message("assistant"):
                assistant_response = run_assistant(client, st.session_state.messages)
                st.write(assistant_response)
                append_message("assistant", assistant_response)
                log_conversation(st.session_state.messages)

    if st.session_state.processing_stems:
        audio_files = list_audio_files(upload_dir)

        if not audio_files:
            st.write("No audio files available for processing.")
        else:
            # Display available songs for stem separation
            file_options = {os.path.basename(file): file for file in audio_files}
            st.session_state.chosen_file = st.selectbox("Select a song to separate stems:",
                                                        options=list(file_options.keys()), key="file_selector")

            if st.button("Separate Stems"):
                st.session_state.chosen_file_path = file_options[st.session_state.chosen_file]

            if st.session_state.chosen_file_path:

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                with st.spinner(f"Separating stems for {os.path.basename(st.session_state.chosen_file_path)}... This may take a few minutes..."):
                    separate_single(st.session_state.chosen_file_path, output_dir)
                    stem_files = find_files(
                        os.path.join(output_dir, "htdemucs_6s", os.path.splitext(st.session_state.chosen_file)[0]))
                    display_audio_files(stem_files)
                st.session_state.processing_stems = False
                st.session_state.chosen_file_path = None
                append_message("assistant", "Here are the separated stems for the selected song.")



            # --- Added Audio Response ---
            if audio_response:
                response = client.audio.speech.create(
                    model=tts_model,
                    voice=tts_voice,
                    input=st.session_state.messages[-1]["content"],
                )
                audio_base64 = base64.b64encode(response.content).decode('utf-8')
                audio_html = f"""
                <audio controls autoplay>
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/mp3">
                </audio>
                """
                st.html(audio_html)

    # --- Audio Analysis Section ---
    if not special_action and not st.session_state.audio_analysis_done:
        st.write("#")
        st.write("### **🎵 Upload 5-10 of your favorite songs to view your report!**")

        with st.form("upload_form", clear_on_submit=False):
            uploads = st.file_uploader(
                "Upload MP3s", type=["mp3"], accept_multiple_files=True, key="uploader"
            )
            submitted = st.form_submit_button("Analyze")

        # Sleek note under the uploader
        st.markdown("""
        <div style="text-align: center; color: #B0B0B0; font-size: small;">
            Please refer to the expandable descriptors in the sidebar for more information on how we carry out the analyses!
        </div>
        """, unsafe_allow_html=True)

        if submitted and uploads:
            # Track analyzed filenames to avoid repetition on reruns
            if "analyzed_files" not in st.session_state:
                st.session_state.analyzed_files = set()

            for up in uploads:
                if up.name in st.session_state.analyzed_files:
                    continue  # skip duplicates across reruns

                file_path = save_uploaded_file(upload_dir, up)

                with st.spinner(f"Analyzing {up.name}..."):
                    # If you ever want to do partial analysis, pass duration=90.0 to librosa.load
                    # Right now we analyze the FULL song:
                    (sr, duration, magnitude_spectrogram, tempo, pulse_clarity,
                     spectral_centroids, spectral_bandwidth, spectral_flux,
                     rms_energy, loudness, key, scale, strength) = analyze_audio(file_path)

                    st.success(f"Analysis for {up.name} complete!")

                    fig, ax = plt.subplots(figsize=(12, 6))
                    spectrogram = librosa.amplitude_to_db(magnitude_spectrogram)
                    img = librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log', ax=ax)
                    ax.set_title('Spectrogram')
                    fig.colorbar(img, ax=ax, format="%+2.0f dB")
                    st.pyplot(fig, use_container_width=True)  # <- fixes deprecation note

                    with open(file_path, "rb") as fh:
                        audio_bytes = fh.read()
                    st.audio(BytesIO(audio_bytes), format="audio/mp3")

                    st.session_state.audio_analysis_results.append({
                        'file_name': up.name,
                        'path': file_path,
                        'duration': duration,
                        'tempo': int(tempo),
                        'pulse_clarity': pulse_clarity,
                        'key': key + ' ' + scale,
                        'key_strength': strength,
                        'spectral_centroids': float(np.mean(spectral_centroids)),
                        'spectral_bandwidth': float(np.mean(spectral_bandwidth)),
                        'spectral_flux': float(np.mean(spectral_flux)),
                        'rms_energy': float(np.mean(rms_energy)),
                        'loudness': loudness,
                    })
                    st.session_state.analyzed_files.add(up.name)
                    plt.close(fig)  # free memory

                    st.write(f"**Song:** {up.name}")
                    st.write(f"**Duration:** {duration:.2f} seconds")
                    st.write(f"**Tempo:** {tempo} BPM")
                    st.write(f"**Pulse Clarity (std of onset strength):** {pulse_clarity}")
                    st.write(f"**Key:** {key} {scale}")
                    st.write(f"**Key Strength:** {strength}")
                    st.write(f"**Average Spectral Centroid:** {spectral_centroids}")
                    st.write(f"**Average Spectral Bandwidth:** {spectral_bandwidth}")
                    st.write(f"**Spectral Flux:** {spectral_flux}")
                    st.write(f"**Average RMS Energy:** {rms_energy}")
                    st.write(f"**Loudness:** {loudness}")
                    st.write("---")

            # Send the analysis results to the assistant once per submit
            with st.chat_message("assistant"):
                formatted_results = format_analysis_results_for_prompt(st.session_state.audio_analysis_results)
                prompt = f"Here are the audio analysis results:\n{formatted_results}"
                append_message("user", prompt)
                assistant_response = run_assistant(client, st.session_state.messages)
                st.write(assistant_response)
                append_message("assistant", assistant_response)
                log_conversation(st.session_state.messages)

            st.session_state.audio_analysis_done = True
            st.stop()  # prevent any further rerun work this cycle

            #st.write("Looking for inspiration?")


            # # Add buttons for further queries and handle their responses
            # query_buttons = {
            #     "What do you think these song choices say about me as a person?": None,
            #     "Based on these songs, what are some other songs you would recommend I listen to?": None,
            #     "Can you give me the stems (separate MP3s for vocals, bass, drums, and other) of one of the songs?": None,
            #     "Which movies do you think I might like based on my music choices?": None
            # }
            #
            # for question, response in query_buttons.items():
            #     if st.button(question):
            #         # Displaying the new messages
            #         st.session_state.questions = True
            #         with st.chat_message("user"):
            #             st.markdown(question)
            #             append_message("user", question)
            #
            #         # Send the analysis results to the assistant
            #         with st.chat_message("assistant"):
            #             assistant_response = run_assistant(client, st.session_state.messages)
            #             st.write(assistant_response)
            #             append_message("assistant", assistant_response)
            #             log_conversation(st.session_state.messages)

            # # Place to add interaction buttons and handle their responses
            # questions = [
            #     "What do you think these song choices say about me as a person?",
            #     "Based on these songs, what are some other songs you would recommend I listen to?",
            #     "Can you give me the stems (separate MP3s for vocals, bass, drums, and other) of one of the songs?",
            #     "Which movies do you think I might like based on my music choices?"
            # ]
            #
            # for question in questions:
            #     if st.button(question):
            #         if question not in st.session_state.questions_asked:
            #             st.session_state.questions_asked[question] = True
            #             append_message('user', question)
            #             response = run_assistant(client, st.session_state.messages)
            #             append_message('assistant', response)
            #             st.write(response)



if __name__ == "__main__":
    main()