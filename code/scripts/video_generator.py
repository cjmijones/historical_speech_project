# video_generator_221_fixed.py
from pathlib import Path
from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, ColorClip, vfx

def _resolve_output_path(output_path: str | Path, audio_path: str | Path) -> Path:
    """If output_path is a directory, create a filename from the audio stem."""
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4":
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / (Path(audio_path).stem + ".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def create_video(
    image_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    fps: int = 24,
    target_size: tuple[int, int] = (1920, 1080),
    pad_color: tuple[int, int, int] = (0, 0, 0),
    zoom_pct: float = 0.04,  # total extra scale (e.g., 0.04 = +4% over clip)
) -> Path:
    """
    MoviePy 2.2.1: make a YouTube-friendly MP4 from a still image + audio.
    - Preserves aspect ratio; pads to target_size.
    - Optional slow Ken Burns zoom using vfx.Resize via with_effects().
    """
    image_path = Path(image_path)
    audio_path = Path(audio_path)
    out_path = _resolve_output_path(output_path, audio_path)

    # Load audio and get duration
    audio_clip = AudioFileClip(str(audio_path))
    duration = float(audio_clip.duration or 0.001)

    # Background canvas (letterbox/pillarbox)
    bg = ColorClip(size=target_size, color=pad_color, duration=duration)

    # Still image (give it a duration at construction time)
    img = ImageClip(str(image_path), duration=duration)

    # Fit the image inside target_size WITHOUT stretching (use .resized in v2)
    iw, ih = img.size
    tw, th = target_size
    scale = min(tw / iw, th / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    img = img.resized(new_size=(new_w, new_h))  # <-- v2.2.1 method

    # Optional slow zoom over the whole clip (vfx.Resize via with_effects)
    if zoom_pct:
        zoom_effect = vfx.Resize(lambda t: 1.0 + zoom_pct * (t / duration))
        img = img.with_effects([zoom_effect])

    # Center the image over the background (v2 immutable style)
    img = img.with_position("center")  # <-- v2 style

    # Composite and attach audio (setting .audio attribute is fine in v2)
    video = CompositeVideoClip([bg, img], size=target_size)
    video.audio = audio_clip

    # Export H.264/AAC with faststart for streaming
    video.write_videofile(
        str(out_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
        ffmpeg_params=["-movflags", "+faststart"],
    )

    # Cleanup
    video.close()
    audio_clip.close()
    return out_path

# Example:
# create_video(
#     "../assets/backgrounds/James_Madison.jpg",
#     "../data/processed_audio/federalist-10_chunks/federalist-10-part-01-tts.mp3",
#     "../data/video_output"
# )
