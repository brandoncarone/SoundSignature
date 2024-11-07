# SoundSignature: A Personalized Music Analysis Web Application

Welcome to the **SoundSignature** repository! This application combines state-of-the-art Music Information Retrieval (MIR), Natural Language Processing (NLP), and the OpenAI GPT-4o model to analyze users’ favorite songs. Through this analysis, SoundSignature provides personalized insights into the user’s musical preferences, acoustic characteristics of chosen songs, and more, all accessible through an interactive chatbot interface.

You can try SoundSignature online here: [SoundSignature on Streamlit](https://soundsignature.streamlit.app).

## Overview

**SoundSignature** is a cloud-based web application that performs feature extraction on user-uploaded songs, interprets these features in accessible ways, and integrates a custom OpenAI Assistant to deliver personalized insights. This project bridges the gap between complex music processing and user-friendly interaction, catering to both casual listeners and musically knowledgeable users. The application leverages advanced MIR and DSP tools and packages such as `librosa`, `essentia`, `madmom`, and others to extract meaningful musical features and provide an in-depth “music profile” for each user.

## Key Features

- **Acoustic Feature Extraction**:
  - Extracts musical and acoustic features like BPM, key and mode, spectral centroid, and loudness.
  - Uses `librosa` and `essentia` for accurate audio processing and analysis.
  
- **Interactive AI Assistant**:
  - A custom OpenAI GPT-4 assistant interprets extracted features and contextualizes them.
  - Allows users to ask specific questions about their music preferences, song characteristics, and provides personalized insights.

- **Tools for Musicians**:
  - Integrates advanced tools for source separation (`DEMUCS`), chord recognition (`CREMA`), and audio-to-MIDI conversion (`basic-pitch`).
  - Users can request stem separation, chord recognition, and MIDI extraction directly via the chatbot interface.

- **Educational and Analytical Platform**:
  - Simplifies complex music terminology, allowing users to learn about musical concepts such as spectral flux, rhythm clarity, and harmonic stability.
  - Offers additional insights on cultural and historical contexts of songs, enhancing music appreciation.

- **Cloud-Based Processing**:
  - Built with `Streamlit` for ease of deployment and interaction.
  - Scalable and accessible from any device with internet access.

## System Architecture

### Feature Extraction Pipeline
SoundSignature uses several Python packages to extract core musical and acoustic features clustered across:

- **Tempo and Rhythmicity** (e.g., BPM, pulse clarity)
- **Harmony and Melody** (e.g., key, mode, and key strength)
- **Timbre and Texture** (e.g., spectral centroid, bandwidth, and flux)
- **Energy and Dynamics** (e.g., RMS energy, loudness)

### AI Assistant Integration
- The assistant is powered by the OpenAI GPT-4o API, with a custom prompt that enables it to interpret musical data and generate meaningful responses. If you'd like to use this yourself, you must create an OpenAI Assistant to handle the API calls.
- Users can ask follow-up questions, receive recommendations, and interact to learn more about their music preferences.
- The assistant also draws from it's knowledge to include information regarding the cultural and historical backgrounds of the artists, as well as the lyrical content of the songs.

### Music Processing Tools
- **DEMUCS**: For music source separation, allowing users to isolate instruments or vocals from uploaded tracks.
- I'm currently working on adding the `basic-pitch` audio-to-midi converter and the `CREMA` chord recognition model.

## Installation

To get started with SoundSignature:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/brandoncarone/SoundSignature.git
   cd SoundSignature
   ```
2. **Install Dependencies**:
   Make sure to install the required Python packages. We recommend setting up a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up OpenAI API Key**:
   - Obtain an OpenAI API key from the [OpenAI Platform](https://platform.openai.com/).
   - Create a `.env` file in the root directory and add your API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_ASSISTANT_ID=your_openai_assistant_ID_here
   ```
4. **Run the Application**:
   ```bash
   streamlit run appp.py
   ```
    (Use appp with 3 p's to run locally, as app.py is the script to run this app on the streamlit cloud)

## Usage

- **Upload Songs**: Upload your favorite songs in `.mp3` format, ensuring the file name format is `SongName_ArtistName.mp3`.
- **Feature Extraction**: SoundSignature will extract musical features and display them on the app interface.
- **Ask Questions**: Engage with the chatbot by asking questions about the extracted features, asking for recommendations, or exploring the cultural/historical background of the songs.
- **Advanced Tools**: Request additional analyses, such as stem extraction, through specific keywords in your chatbot queries (e.g., "Give me the stems").

## Citation

If you use **SoundSignature** in your research, please cite the following paper:

> B. J. Carone and P. Ripollés, "SoundSignature: What Type of Music do you Like?," *2024 IEEE 5th International Symposium on the Internet of Sounds (IS2)*, Erlangen, Germany, 2024, pp. 1-10, doi: [10.1109/IS262782.2024.10704174](https://doi.org/10.1109/IS262782.2024.10704174).

You can access the full paper on IEEE Xplore: [SoundSignature: What Type of Music Do You Like?](https://ieeexplore.ieee.org/document/10704174?figureId=fig1#fig1).

## User Study and Results

SoundSignature has been validated through a pilot user study, indicating high user satisfaction with the insights provided. Participants valued the detailed breakdown of their musical tastes and the ability to interact with a chatbot that could understand their music preferences. For more on this study, please refer to the research paper linked above.

## Future Directions

Future developments for SoundSignature include:

- Enhanced feature extraction pipeline with additional audio features.
- Integration of cognitive metrics like musical surprisal.
- Expanding interactivity by allowing users to build personalized music profiles and connect with others.

## Contributing

We welcome contributions to SoundSignature! Please feel free to open issues or submit pull requests to improve the project.

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

## Contact

For questions or feedback, please contact Brandon James Carone at `bcarone@nyu.edu`.
