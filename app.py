import io
import os
import shutil
import tempfile
import math
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import soundfile as sf

import streamlit as st
import librosa
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.signal import butter, filtfilt

# Use a non-interactive backend (important when running under Streamlit)
matplotlib.use("Agg")

# ---------- CONFIG ----------
W, H = 1080, 1920   # TikTok-style vertical resolution
FPS_DEFAULT = 30


# ---------- AUDIO ANALYSIS ----------

def analyze_audio(audio_path):
    """
    Load audio and compute beat times.
    Returns: y, sr, beat_times (np.array of seconds)
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return y, sr, beat_times


def apply_speed_change(y, rate):
    """Return audio samples stretched/compressed by `rate`."""
    if rate == 1.0:
        return y
    if rate <= 0:
        raise ValueError("Playback speed must be greater than 0.")
    return librosa.effects.time_stretch(y, rate=rate)


def lowpass_filter(y, sr, cutoff=150.0, numtaps=101):
    """Simple windowed-sinc low-pass filter used for bass isolation."""
    if cutoff <= 0 or sr <= 0 or cutoff >= sr / 2:
        return y

    taps = max(3, int(numtaps))
    if taps % 2 == 0:
        taps += 1  # ensure odd number of taps for symmetry

    n = np.arange(taps) - (taps - 1) / 2
    normalized_cutoff = cutoff / sr
    h = 2 * normalized_cutoff * np.sinc(2 * normalized_cutoff * n)
    window = np.hamming(taps)
    h *= window
    h /= np.sum(h)

    filtered = np.convolve(y, h, mode="same")
    return filtered.astype(y.dtype, copy=False)


def apply_bass_boost(y, sr, boost_db, cutoff=150.0):
    """Boost low-frequency content by `boost_db` decibels."""
    if boost_db <= 0:
        return y

    low_freq = lowpass_filter(y, sr, cutoff=cutoff)
    gain = 10 ** (boost_db / 20.0)
    boosted = y + (gain - 1.0) * low_freq
    boosted = np.clip(boosted, -1.0, 1.0)
    return boosted.astype(y.dtype, copy=False)


def apply_level2a_bass_enhancement(y, sr, amount=0.0):
    """
    Apply a multi-stage "Level-2A" bass enhancer.

    The effect mimics a two-amp analog chain:
    1. Isolate sub (<=90 Hz) and punch (90-220 Hz) bands.
    2. Drive each band with soft saturation to thicken harmonics.
    3. Recombine with the dry signal plus a hint of high-frequency air.

    Parameters
    ----------
    y : np.ndarray
        Mono audio samples (-1..1).
    sr : int
        Sample rate of ``y``.
    amount : float
        0.0-1.0 blend amount. Higher values push the saturation harder.
    """

    if amount <= 0.0 or sr <= 0 or y.size == 0:
        return y

    # Work in float32 to keep numerical noise low and saturation predictable.
    y_work = y.astype(np.float32, copy=True)

    # Stage 1: isolate sub frequencies.
    sub = lowpass_filter(y_work, sr, cutoff=90.0, numtaps=401)

    # Stage 2: capture 90-220 Hz punch by subtracting the subs from a broader lowpass.
    low_ext = lowpass_filter(y_work, sr, cutoff=220.0, numtaps=401)
    punch = low_ext - sub

    # Stage 3: preserve a bit of high-end definition so the mix does not get muddy.
    high_shelf = y_work - lowpass_filter(y_work, sr, cutoff=1200.0, numtaps=401)

    # Soft saturation for musical harmonics.
    sub_drive = 1.0 + 4.0 * amount
    punch_drive = 1.0 + 2.5 * amount

    sub_processed = np.tanh(sub * sub_drive)
    punch_processed = np.tanh(punch * punch_drive)

    # Recombine with a balance tailored for "Level-2A" punchiness.
    dry_gain = 1.0 - 0.35 * amount
    sub_gain = 0.55 + 0.35 * amount
    punch_gain = 0.35 + 0.25 * amount
    air_gain = 0.08 * amount

    enhanced = (
        dry_gain * y_work
        + sub_gain * sub_processed
        + punch_gain * punch_processed
        + air_gain * high_shelf
    )

    # Additional harmonic sparkle derived from the two driven bands.
    harmonic = np.tanh((sub_processed + punch_processed) * (1.5 + amount))
    enhanced += harmonic * (0.12 * amount)

    peak = np.max(np.abs(enhanced))
    if peak > 1.0:
        enhanced /= peak

    return enhanced.astype(y.dtype, copy=False)


def apply_reverb(y, sr, mix, delay_ms=80.0, decay=0.45, echoes=5):
    """Simple Schroeder-style reverb made from a stack of echoes."""
    if mix <= 0.0:
        return y

    delay = max(1, int(sr * (delay_ms / 1000.0)))
    impulse_length = delay * echoes + 1
    impulse = np.zeros(impulse_length, dtype=np.float32)
    for i in range(echoes):
        impulse[i * delay] = decay ** i

    wet = np.convolve(y, impulse, mode="full")[: len(y)]
    wet /= max(1e-6, np.max(np.abs(wet)))

    mixed = (1.0 - mix) * y + mix * wet
    mixed = np.clip(mixed, -1.0, 1.0)
    return mixed.astype(y.dtype, copy=False)


def detect_bpm(y, sr):
    """Estimate BPM for a mono signal."""
    if y.size == 0:
        return 0.0
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)


def time_stretch_to_bpm(y, sr, target_bpm):
    """Stretch the input so its tempo approaches ``target_bpm``."""
    original_bpm = detect_bpm(y, sr)
    if original_bpm <= 0:
        return y
    rate = target_bpm / original_bpm
    if rate <= 0:
        return y
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y, sr, semitones):
    """Pitch-shift the audio by the given number of semitones."""
    if abs(semitones) < 1e-6:
        return y
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)


def synth_cowbell(sr, duration=0.15):
    """Generate a short metallic cowbell-like hit."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = [800.0, 1200.0, 1600.0]
    signal = np.zeros_like(t)
    for f in freqs:
        signal += np.sin(2 * math.pi * f * t)
    env = np.exp(-t * 25.0)
    signal *= env
    signal = np.tanh(2.0 * signal)
    peak = np.max(np.abs(signal)) + 1e-9
    return (signal / peak).astype(np.float32)


def make_cowbell_track(length_samples, sr, bpm, gain):
    """Create a repeating cowbell pattern aligned to the detected BPM."""
    if length_samples == 0:
        return np.zeros(0, dtype=np.float32)

    cb = synth_cowbell(sr)
    cb_len = len(cb)
    output = np.zeros(length_samples, dtype=np.float32)

    beat_dur = 60.0 / max(1e-6, bpm)
    pattern_beats = [0.0, 1.5, 2.0, 3.5]
    bar_duration = beat_dur * 4
    total_duration = length_samples / sr

    t = 0.0
    while t < total_duration:
        for pb in pattern_beats:
            hit_time = t + pb * beat_dur
            if hit_time >= total_duration:
                break
            idx = int(hit_time * sr)
            end = min(idx + cb_len, length_samples)
            if idx < 0 or idx >= length_samples:
                continue
            length_here = end - idx
            output[idx:end] += gain * cb[:length_here]
        t += bar_duration

    peak = np.max(np.abs(output)) + 1e-9
    if peak > 1.0:
        output /= peak
    return output


def soft_distortion(x, drive_db):
    """Apply tanh-based soft clipping."""
    if abs(drive_db) < 1e-6:
        return x
    drive = 10 ** (drive_db / 20.0)
    y = np.tanh(x * drive)
    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak).astype(np.float32)


def low_shelf_eq(y, sr, cutoff_hz=200.0, gain_db=3.0, order=4):
    """Very simple low-shelf style boost via low-pass blending."""
    if abs(gain_db) < 1e-6:
        return y

    nyq = 0.5 * sr
    if nyq <= 0:
        return y
    norm_cutoff = min(max(cutoff_hz / nyq, 0.0), 0.999)
    b, a = butter(order, norm_cutoff, btype="low")
    y_lp = filtfilt(b, a, y).astype(np.float32)
    g = 10 ** (gain_db / 20.0)
    y_out = y + (g - 1.0) * y_lp
    peak = np.max(np.abs(y_out)) + 1e-9
    if peak > 1.0:
        y_out /= peak
    return y_out.astype(np.float32)


def make_brazilian_phonk_audio(
    y,
    sr,
    target_bpm=170.0,
    pitch_down_semitones=-4.0,
    cowbell_gain=0.7,
    distortion_drive_db=6.0,
    bass_boost_db=3.0,
):
    """Remix the audio using a simplified Brazilian phonk recipe."""
    if y.size == 0:
        return y

    y_work = y.astype(np.float32, copy=True)
    y_work = time_stretch_to_bpm(y_work, sr, target_bpm)
    y_work = pitch_shift(y_work, sr, pitch_down_semitones)

    cow = make_cowbell_track(len(y_work), sr, bpm=target_bpm, gain=cowbell_gain)
    mix = y_work + cow
    peak = np.max(np.abs(mix)) + 1e-9
    if peak > 1.0:
        mix /= peak

    if abs(bass_boost_db) > 1e-6:
        mix = low_shelf_eq(mix, sr, cutoff_hz=200.0, gain_db=bass_boost_db)
    if abs(distortion_drive_db) > 1e-6:
        mix = soft_distortion(mix, distortion_drive_db)

    peak = np.max(np.abs(mix)) + 1e-9
    if peak > 0.99:
        mix = mix / peak * 0.99
    return mix.astype(np.float32, copy=False)


def audio_array_to_wav_bytes(y, sr):
    """Serialize a mono waveform to WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV")
    buffer.seek(0)
    return buffer.read()


def beat_intensity(t, beat_times, window=0.06):
    """
    Return a value 0..1 indicating how close time t is to a beat.
    window = +/- window seconds around beat.
    """
    if beat_times.size == 0:
        return 0.0
    idx = np.argmin(np.abs(beat_times - t))
    dist = abs(beat_times[idx] - t)
    if dist < window:
        return float(1.0 - dist / window)
    return 0.0


def get_spectrum_at_time(y, sr, t, n_fft=2048, n_bands=32):
    """
    Compute a simple magnitude spectrum at time t, grouped into n_bands.
    Returns normalized band magnitudes in [0,1].
    """
    center = int(t * sr)
    half = n_fft // 2
    start = max(center - half, 0)
    end = min(center + half, len(y))

    window = np.zeros(n_fft, dtype=np.float32)
    audio_slice = y[start:end]
    window[:len(audio_slice)] = audio_slice

    # Hann window for smoother spectrum
    windowed = window * np.hanning(n_fft)
    spectrum = np.fft.rfft(windowed)
    magnitudes = np.abs(spectrum)

    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    max_freq = sr / 2
    band_edges = np.linspace(0, max_freq, n_bands + 1)

    bands = []
    for i in range(n_bands):
        f_min, f_max = band_edges[i], band_edges[i + 1]
        mask = (freqs >= f_min) & (freqs < f_max)
        if np.any(mask):
            bands.append(magnitudes[mask].mean())
        else:
            bands.append(0.0)

    bands = np.array(bands, dtype=np.float32)
    max_val = bands.max()
    if max_val > 0:
        bands /= max_val
    return bands


# ---------- VISUAL TEMPLATES ----------

CENTER = (W // 2, H // 2)
BAND_COLORMAP = cm.get_cmap("plasma")


GRADIENT_PRESETS = {
    "Neon purple / blue": ((40, 5, 85), (25, 180, 255)),
    "Electric pink / teal": ((255, 20, 147), (0, 220, 200)),
    "Midnight indigo": ((12, 6, 35), (10, 48, 85)),
}
DEFAULT_GRADIENT_PRESET = "Neon purple / blue"

_BACKGROUND_CACHE = {}


def get_gradient_background(
    size,
    top_color=(8, 3, 30),
    bottom_color=(5, 45, 80),
):
    """Return (and cache) a subtle vertical gradient background."""

    key = (size, top_color, bottom_color)
    cached = _BACKGROUND_CACHE.get(key)
    if cached is not None:
        return cached

    width, height = size
    gradient = Image.new("RGB", size)
    grad_draw = ImageDraw.Draw(gradient)

    if height <= 1:
        grad_draw.rectangle([0, 0, width, height], fill=top_color)
    else:
        for y in range(height):
            ratio = y / (height - 1)
            color = tuple(
                int(top * (1 - ratio) + bottom * ratio)
                for top, bottom in zip(top_color, bottom_color)
            )
            grad_draw.line((0, y, width, y), fill=color)

    _BACKGROUND_CACHE[key] = gradient
    return gradient


def resolve_gradient_colors(preset_name):
    return GRADIENT_PRESETS.get(
        preset_name, GRADIENT_PRESETS[DEFAULT_GRADIENT_PRESET]
    )


def get_band_style(index, value, total_bands, min_width=2, max_width=10):
    """Return (color, width, glow_alpha) for a spectral band."""
    if total_bands <= 1:
        normalized_index = 0.0
    else:
        normalized_index = index / (total_bands - 1)

    base_color = BAND_COLORMAP(normalized_index)  # RGBA 0..1 values
    brightness = 0.4 + 0.6 * float(value)
    r, g, b = [
        int(255 * min(1.0, channel * brightness))
        for channel in base_color[:3]
    ]

    width = min_width + (max_width - min_width) * float(value)
    glow_alpha = int(60 + 150 * float(value))
    return (r, g, b), max(1, int(width)), max(10, min(255, glow_alpha))


def draw_circular_spectrum_frame(
    t,
    y,
    sr,
    beat_times,
    reverb_amount=0.0,
    n_bands=64,
    gradient_colors=None,
):
    """Circular spectrum with mirrored bands, gradient background, and glow."""

    bands = get_spectrum_at_time(y, sr, t, n_bands=n_bands)
    beat = beat_intensity(t, beat_times)

    if gradient_colors is None:
        gradient_colors = GRADIENT_PRESETS[DEFAULT_GRADIENT_PRESET]
    top_color, bottom_color = gradient_colors

    background = get_gradient_background((W, H), top_color, bottom_color)
    img = background.copy()
    draw = ImageDraw.Draw(img)
    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    pulse_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pulse_draw = ImageDraw.Draw(pulse_layer)

    base_radius = 250
    max_extra = 240 * (1 + 0.5 * beat)

    # Pre-draw the minimal circle so it is always visible behind the spectrum bars.
    circle_color = (30, 30, 50)
    circle_thickness = 4
    draw.ellipse(
        (
            CENTER[0] - base_radius,
            CENTER[1] - base_radius,
            CENTER[0] + base_radius,
            CENTER[1] + base_radius,
        ),
        outline=circle_color,
        width=circle_thickness,
    )

    n = len(bands)
    wave_points = []
    for i, v in enumerate(bands):
        extra = float(v) * max_extra
        r1 = base_radius
        r2 = base_radius + extra

        color, width, glow_alpha = get_band_style(i, v, n)

        for offset in (0.0, math.pi):
            angle = (2 * math.pi * i / n) + offset
            x1 = CENTER[0] + r1 * math.cos(angle)
            y1 = CENTER[1] + r1 * math.sin(angle)
            x2 = CENTER[0] + r2 * math.cos(angle)
            y2 = CENTER[1] + r2 * math.sin(angle)

            wave_points.append((angle % (2 * math.pi), (x2, y2)))

            draw.line((x1, y1, x2, y2), width=width, fill=color)

            glow_width = max(width + 2, int(width * 1.8))
            glow_color = (*color, glow_alpha)
            glow_draw.line((x1, y1, x2, y2), width=glow_width, fill=glow_color)

    # Draw a wavy outline following the tips of the bars to emphasize the
    # circular spectrum "wave".
    if len(wave_points) > 2:
        wave_points.sort(key=lambda item: item[0])
        ordered_points = [pt for _, pt in wave_points]
        wave_outline = ordered_points + [ordered_points[0]]
        wave_color = (180, 200, 255, 90)
        glow_draw.line(wave_outline, width=6, fill=wave_color)

    # "Reverb" ring pulse on strong beats
    if reverb_amount > 0.0 and beat > 0.0:
        ring_radius = base_radius + max_extra * 1.1
        thickness = max(2, int(2 + (20 * beat * reverb_amount)))
        ring_color = (220, 220, 255, int(80 + 120 * beat * reverb_amount))
        pulse_draw.ellipse(
            (
                CENTER[0] - ring_radius,
                CENTER[1] - ring_radius,
                CENTER[0] + ring_radius,
                CENTER[1] + ring_radius,
            ),
            outline=ring_color,
            width=thickness,
        )

    composed = Image.alpha_composite(img.convert("RGBA"), glow_layer)
    composed = Image.alpha_composite(composed, pulse_layer).convert("RGB")
    return np.array(composed)


def draw_bar_spectrum_frame(t, y, sr, beat_times, reverb_amount=0.0, n_bands=32):
    """
    Bar spectrum visual. If reverb=True, adds a faint echo above the bars on beats.
    """
    bands = get_spectrum_at_time(y, sr, t, n_bands=n_bands)
    beat = beat_intensity(t, beat_times)

    img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin_bottom = 120
    margin_side = 80
    bar_width = (W - 2 * margin_side) / n_bands
    max_bar_height = H * 0.6

    for i, v in enumerate(bands):
        height = float(v) * max_bar_height * (1 + 0.4 * beat)
        x1 = int(margin_side + i * bar_width)
        x2 = int(margin_side + (i + 1) * bar_width * 0.85)
        y2 = H - margin_bottom
        y1 = int(y2 - height)

        # main bar
        draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))

        # "Reverb" echo bar slightly above the main one on strong beats
        if reverb_amount > 0.0 and beat > 0.0:
            echo_height = int(height * (0.1 + 0.6 * beat * reverb_amount))
            echo_y2 = y1 - 10
            echo_y1 = max(echo_y2 - echo_height, 0)
            draw.rectangle([x1, echo_y1, x2, echo_y2], fill=(180, 180, 255))

    return np.array(img)


# ---------- RENDERING WITH MATPLOTLIB + FFMPEG ----------

def render_mp4(
    audio_path,
    output_path,
    template="circular",
    fps=30,
    max_duration=None,
    reverb_amount=0.0,
    start_time=0.0,
    end_time=None,
    gradient_preset=None,
):
    """
    Render an MP4 video from the given audio file and template,
    using matplotlib.animation.FFMpegWriter, and then mux the original audio
    into the exported video so it is ready to post.
    """
    # Analyze audio
    y, sr, beat_times = analyze_audio(audio_path)
    total_duration = len(y) / sr

    segment_start = max(0.0, min(start_time, total_duration))

    if end_time is None:
        segment_end = total_duration
    else:
        segment_end = max(segment_start, min(end_time, total_duration))

    if max_duration is not None:
        segment_end = min(segment_start + max_duration, segment_end)

    start_idx = int(segment_start * sr)
    end_idx = int(segment_end * sr)

    if end_idx <= start_idx:
        raise ValueError("Selected time range is too short to render video.")

    y = y[start_idx:end_idx]
    beat_mask = (beat_times >= segment_start) & (beat_times <= segment_end)
    beat_times = beat_times[beat_mask] - segment_start

    duration = len(y) / sr

    # Time steps for frames
    n_frames = int(duration * fps)
    if n_frames < 1:
        n_frames = 1
    times = np.linspace(0, duration, n_frames, endpoint=False)

    # Setup matplotlib figure
    dpi = 100
    fig_w = W / dpi
    fig_h = H / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])  # full-figure axes
    ax.axis("off")

    # Initial frame
    t0 = times[0]
    gradient_colors = resolve_gradient_colors(gradient_preset)
    if template == "circular":
        frame0 = draw_circular_spectrum_frame(
            t0,
            y,
            sr,
            beat_times,
            reverb_amount=reverb_amount,
            gradient_colors=gradient_colors,
        )
    else:
        frame0 = draw_bar_spectrum_frame(
            t0, y, sr, beat_times, reverb_amount=reverb_amount
        )

    im = ax.imshow(frame0, animated=True)

    def update(i):
        t = times[i]
        if template == "circular":
            frame = draw_circular_spectrum_frame(
                t,
                y,
                sr,
                beat_times,
                reverb_amount=reverb_amount,
                gradient_colors=gradient_colors,
            )
        else:
            frame = draw_bar_spectrum_frame(
                t, y, sr, beat_times, reverb_amount=reverb_amount
            )
        im.set_array(frame)
        return [im]

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=True,
    )

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video_path = temp_video.name
    temp_video.close()

    writer = FFMpegWriter(fps=fps, bitrate=5000)
    ani.save(temp_video_path, writer=writer)

    plt.close(fig)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        temp_video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]

    try:
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to mux audio into the MP4") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed while adding audio: {stderr}") from exc
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


# ---------- STREAMLIT UI ----------

st.set_page_config(page_title="Beat Visualizer", layout="centered")

st.title("🎵 Beat-Synced Visualizer (MP4 via FFMpegWriter)")

st.write(
    "Upload a track and generate a vertical MP4 video with either a circular "
    "spectrum or a bar spectrum visual, synced to the beat.\n\n"
    "The exported MP4 now includes the original audio automatically so it is ready to upload."
)

uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "flac", "ogg"])

preview_y = None
preview_sr = None
detected_phonk_bpm = None

if uploaded_file is not None:
    uploaded_file.seek(0)
    try:
        preview_y, preview_sr = librosa.load(uploaded_file, sr=None, mono=True)
        if preview_y.size > 0:
            detected_phonk_bpm = detect_bpm(preview_y, preview_sr)
    except Exception as exc:
        st.error(f"Unable to analyze uploaded audio: {exc}")
        preview_y, preview_sr = None, None
        detected_phonk_bpm = None
    finally:
        uploaded_file.seek(0)

gradient_options = list(GRADIENT_PRESETS.keys())
default_gradient_index = gradient_options.index(DEFAULT_GRADIENT_PRESET)

col1, col2, col3 = st.columns(3)
with col1:
    template_choice = st.radio(
        "Template",
        ["Circular spectrum", "Bar spectrum"],
    )
with col2:
    fps = st.slider("FPS", 10, 60, FPS_DEFAULT)
with col3:
    gradient_choice = st.selectbox(
        "Background gradient",
        gradient_options,
        index=default_gradient_index,
        help="Applied to the circular spectrum to mimic TikTok's neon gradients.",
    )

reverb_amount = st.slider(
    "Reverb amount",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Blend in a synthetic echo. Move the slider to instantly update the preview audio and visuals.",
)

playback_speed = st.slider(
    "Playback speed",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.05,
    help="Adjust how fast the audio should play in the preview and rendered video.",
)

bass_boost_db = st.slider(
    "Bass boost (dB)",
    min_value=0.0,
    max_value=12.0,
    value=0.0,
    step=0.5,
    help="Increase low-frequency energy before previewing or rendering.",
)

level2a_amount = st.slider(
    "Level-2A bass enhancement",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    help="Analog-style, multi-stage bass enhancer. Blend amounts above 0.4 deliver the signature punch.",
)

phonk_enabled = st.checkbox(
    "Enable Brazilian phonk remix",
    value=False,
    help="Applies BPM-matched time-stretching, -4 semitone pitch drop, cowbell hits, bass boost, and light distortion.",
)

phonk_target_bpm_default = 170.0
if detected_phonk_bpm and detected_phonk_bpm > 0:
    phonk_target_bpm_default = float(
        np.clip(detected_phonk_bpm, 120.0, 220.0)
    )

phonk_target_bpm = phonk_target_bpm_default
phonk_pitch_down = -4.0
phonk_cowbell_gain = 0.7
phonk_distortion_db = 6.0
phonk_bass_boost_db = 3.0

if phonk_enabled:
    phonk_target_bpm = st.slider(
        "Phonk target BPM",
        min_value=120.0,
        max_value=220.0,
        value=float(phonk_target_bpm_default),
        step=1.0,
        help="Detected tempo is stretched to this BPM before remixing.",
    )
    if detected_phonk_bpm and detected_phonk_bpm > 0:
        st.caption(
            f"Detected {detected_phonk_bpm:.1f} BPM from the upload and used it as the default phonk target."
        )
    phonk_pitch_down = st.slider(
        "Pitch shift (semitones)",
        min_value=-12.0,
        max_value=0.0,
        value=-4.0,
        step=0.5,
        help="Negative values drop the track for that darker phonk vibe.",
    )
    phonk_cowbell_gain = st.slider(
        "Cowbell gain",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Controls the level of the synthetic cowbell loop.",
    )
    phonk_distortion_db = st.slider(
        "Distortion drive (dB)",
        min_value=0.0,
        max_value=18.0,
        value=6.0,
        step=0.5,
        help="Adds tanh-based saturation after the remixing stage.",
    )
    phonk_bass_boost_db = st.slider(
        "Phonk bass boost (dB)",
        min_value=0.0,
        max_value=12.0,
        value=3.0,
        step=0.5,
        help="Low-shelf emphasis applied inside the phonk chain.",
    )

max_duration = st.number_input(
    "Max duration (seconds, 0 = full track)",
    min_value=0.0,
    value=10.0,
    step=1.0,
)

start_time = 0.0
end_time = None

if uploaded_file is not None:
    if preview_y is None or preview_sr is None:
        st.error("Uploaded audio could not be decoded for preview.")
    else:
        total_preview_samples = len(preview_y)
        if total_preview_samples == 0:
            st.warning("Uploaded audio appears to be empty.")
        else:
            adjusted_y = apply_speed_change(preview_y, playback_speed)
            if bass_boost_db > 0:
                adjusted_y = apply_bass_boost(adjusted_y, preview_sr, bass_boost_db)
            if level2a_amount > 0:
                adjusted_y = apply_level2a_bass_enhancement(
                    adjusted_y, preview_sr, level2a_amount
                )
            if reverb_amount > 0:
                adjusted_y = apply_reverb(adjusted_y, preview_sr, reverb_amount)
            if phonk_enabled:
                adjusted_y = make_brazilian_phonk_audio(
                    adjusted_y,
                    preview_sr,
                    target_bpm=phonk_target_bpm,
                    pitch_down_semitones=phonk_pitch_down,
                    cowbell_gain=phonk_cowbell_gain,
                    distortion_drive_db=phonk_distortion_db,
                    bass_boost_db=phonk_bass_boost_db,
                )

            adjusted_duration = len(adjusted_y) / preview_sr
            st.write(
                f"Audio duration with {playback_speed:.2f}x speed: {adjusted_duration:.2f} seconds"
            )

            preview_audio_bytes = audio_array_to_wav_bytes(adjusted_y, preview_sr)
            st.audio(preview_audio_bytes, format="audio/wav")
            st.caption("Listen to the adjusted speed before rendering your video.")

            time_axis = np.linspace(0, adjusted_duration, len(adjusted_y))
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(time_axis, adjusted_y, color="#4f8bf9", linewidth=0.8)
            ax.set_title("Waveform preview")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_xlim(0, adjusted_duration)
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            st.caption(
                "Use the Matplotlib zoom and pan controls (upper-right toolbar) "
                "to inspect the waveform before choosing your final start and end times."
            )

            default_end = float(adjusted_duration)
            start_time, end_time = st.slider(
                "Select start and end time (seconds)",
                min_value=0.0,
                max_value=float(adjusted_duration),
                value=(0.0, default_end),
                step=0.1,
            )
            st.caption(
                "The visualizer will render only the selected section."
                " Max duration above still applies if it is greater than 0."
            )

            if start_time == end_time:
                st.warning("Choose a range longer than 0 seconds for rendering.")

render_button = st.button("Render MP4")

if render_button:
    if uploaded_file is None:
        st.error("Please upload an audio file first.")
    else:
        audio_path = None
        output_dir = None
        output_path = None

        # Save uploaded audio to a temp file
        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_audio:
            tmp_audio.write(uploaded_file.read())
            audio_path = tmp_audio.name

        # Prepare output path
        output_dir = tempfile.mkdtemp()
        output_path = os.path.join(output_dir, "visualizer.mp4")

        st.write("Rendering... longer duration and higher FPS will take more time.")
        progress = st.progress(0)

        processed_audio_path = None

        try:
            progress.progress(10)
            template_key = "circular" if "Circular" in template_choice else "bars"
            max_d = max_duration if max_duration > 0 else None

            needs_processing = (
                playback_speed != 1.0
                or bass_boost_db > 0
                or level2a_amount > 0
                or reverb_amount > 0
                or phonk_enabled
            )

            if needs_processing:
                y_full, sr_full = librosa.load(audio_path, sr=None, mono=True)
                if playback_speed != 1.0:
                    y_full = apply_speed_change(y_full, playback_speed)
                if bass_boost_db > 0:
                    y_full = apply_bass_boost(y_full, sr_full, bass_boost_db)
                if level2a_amount > 0:
                    y_full = apply_level2a_bass_enhancement(
                        y_full, sr_full, level2a_amount
                    )
                if reverb_amount > 0:
                    y_full = apply_reverb(y_full, sr_full, reverb_amount)
                if phonk_enabled:
                    y_full = make_brazilian_phonk_audio(
                        y_full,
                        sr_full,
                        target_bpm=phonk_target_bpm,
                        pitch_down_semitones=phonk_pitch_down,
                        cowbell_gain=phonk_cowbell_gain,
                        distortion_drive_db=phonk_distortion_db,
                        bass_boost_db=phonk_bass_boost_db,
                    )

                temp_processed = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_processed.close()
                sf.write(temp_processed.name, y_full, sr_full, format="WAV")
                processed_audio_path = temp_processed.name
            else:
                processed_audio_path = audio_path

            render_mp4(
                audio_path=processed_audio_path,
                output_path=output_path,
                template=template_key,
                fps=int(fps),
                max_duration=max_d,
                reverb_amount=reverb_amount,
                start_time=float(start_time),
                end_time=float(end_time),
                gradient_preset=gradient_choice,
            )

            progress.progress(100)

            st.success("Done! Preview or download your MP4 below:")

            # Show the freshly rendered video directly in the app so users can
            # verify the visuals without needing to download first.
            st.video(output_path)

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.download_button(
                label="Download MP4",
                data=video_bytes,
                file_name="visualizer.mp4",
                mime="video/mp4",
            )

        except Exception as e:
            st.error(f"Error during rendering: {e}")
        finally:
            progress.empty()
            if processed_audio_path and processed_audio_path != audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)

            for temp_path in (audio_path, output_path):
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

            if output_dir:
                shutil.rmtree(output_dir, ignore_errors=True)
