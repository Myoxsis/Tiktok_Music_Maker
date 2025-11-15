# TikTok Music Maker

## Overview
TikTok Music Maker is a Streamlit application that turns any uploaded audio file into a beat-synced, TikTok-ready vertical MP4 visualizer. The app analyzes the track with `librosa`, renders circular or bar-style spectrum animations with matplotlib, and muxes the original audio back into the exported video so it is ready to share immediately.

## Features
- 🎧 **Beat-aware visuals** – Detects tempo and beat positions to drive spectrum pulses and beat “reverb” effects.
- 🌀 **Template options** – Switch between circular and bar spectrum templates and tune frame rate, beat glow, and playback speed.
- ✂️ **Flexible trimming** – Slider-based controls let you render only a specific section of the song and optionally limit the maximum duration.
- 🔊 **Preview before export** – Listen to time-stretched audio previews, inspect the waveform, and confirm settings before rendering.
- 📹 **MP4 export with audio** – Uses `matplotlib.animation.FFMpegWriter` plus `ffmpeg` to deliver a ready-to-upload MP4 that includes the source audio.

## Prerequisites
- **Python**: 3.9+ is recommended for compatibility with Streamlit, librosa, and the numerical stack.
- **ffmpeg**: Required to mux the processed visuals with the original audio. Ensure `ffmpeg` is on your `PATH` (`ffmpeg -version`).
- **System libraries**: The usual scientific Python dependencies (NumPy, SciPy, soundfile, etc.) are pulled in via `requirements.txt`. On Linux you may need `libsndfile` and build tools (`build-essential`, `libffi-dev`).

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the App
```bash
streamlit run app.py
```
Streamlit prints a local URL (and optionally a remote/public URL if you enable it). Open the link in your browser to interact with the UI.

## Usage Tips
1. **Upload audio** – MP3, WAV, FLAC, and OGG files are accepted. After upload the waveform preview plus playback-speed adjusted audio preview help confirm the source.
2. **Template & sliders**:
   - *Template radio*: Choose **Circular spectrum** for a radial ring or **Bar spectrum** for stacked columns.
   - *FPS slider*: Higher FPS (e.g., 45–60) produces smoother animations but increases render time.
   - *Playback speed slider*: Adjust 0.5×–1.5× to time-stretch the audio/visuals. The waveform plot updates to match the new duration.
   - *Start/End slider*: Drag handles to pick a subsection of the track; combine with **Max duration** to keep outputs short for TikTok.
   - *Beat reverb checkbox*: Adds glow/echo effects during strong beats.
3. **Render MP4** – Click **Render MP4** once satisfied with settings. Rendering time scales with FPS, selection length, and template complexity.
4. **Preview & download** – When rendering finishes, the app embeds the newly created MP4 in the page so you can inspect it before downloading.

## Troubleshooting
- **`ffmpeg` not found**: The render step shells out to `ffmpeg` to mux visuals with the original audio. Install it from [ffmpeg.org](https://ffmpeg.org/download.html) or your package manager (`brew install ffmpeg`, `choco install ffmpeg`, `apt install ffmpeg`) and ensure the binary is in your `PATH`.
- **Missing system deps**: If `pip install -r requirements.txt` fails with errors about `libsndfile` or compilation, install the corresponding OS packages first (`sudo apt install libsndfile1` on Debian/Ubuntu).
- **Long render times**: Lower the FPS slider or shorten the selected time range. Rendering is CPU-bound because every frame is drawn via matplotlib.

## Rendered Video Storage & Downloading
Rendered videos are written to a temporary directory (e.g., `/tmp/tmpabcd/visualizer.mp4`) during each session. After a successful render the app:
1. Displays the MP4 inline via `st.video` so you can replay it without leaving the page.
2. Offers a **Download MP4** button that streams the freshly created `visualizer.mp4` to your browser.
Because the file lives in a temp directory, download it immediately if you want to keep it—the temp folder is cleaned up when the Streamlit session restarts.

## Additional Notes
- Keep the Streamlit process running while rendering; closing the browser or stopping the server interrupts the job.
- Higher-resolution renders (1080×1920) are baked in for TikTok. Resize parameters in `app.py` (`W`, `H`) if you need different aspect ratios.

Enjoy crafting mesmerizing, beat-synced visuals for your TikTok tracks! 🎶
