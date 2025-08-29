from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import math, subprocess, shutil

FPS = 24
TARGET_W, TARGET_H = 1080, 1920
PAD_COLOR = "black"
ZOOM_PCT = 0.04          # total zoom increase over the whole clip
CODEC_NV = "h264_nvenc"  # or "hevc_nvenc"
NVENC_PRESET = "p5"      # p1 fastest .. p7 slowest/best
NVENC_CQ = "19"          # ~18–21 visually transparent
MAX_WORKERS = 2
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

BACKGROUND_IMAGE_PATH = Path("../assets/backgrounds/James_Madison.jpg")
PROCESSED_AUDIO_DIR = Path("../data/processed_audio/federalist-10_chunks")
BASIC_VIDEO_OUTPUT_DIR = Path("../data/video_output/federalist-10/basic")

def _resolve_output_path(output_path: Path, audio_path: Path) -> Path:
    if output_path.suffix.lower() != ".mp4":
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / (audio_path.stem + ".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def _audio_duration_sec(audio_path: Path) -> float:
    # Robust, no extra deps:
    # ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 file
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(audio_path)],
        text=True
    ).strip()
    try:
        return max(0.001, float(out))
    except ValueError:
        return 0.001

def _nvenc_available() -> bool:
    try:
        encs = subprocess.check_output([FFMPEG, "-hide_banner", "-encoders"], text=True, errors="ignore")
        return ("h264_nvenc" in encs) or ("hevc_nvenc" in encs)
    except Exception:
        return False

def make_cmd(image_path: Path, audio_path: Path, out_path: Path) -> list[str]:
    dur = _audio_duration_sec(audio_path)
    frames = max(1, int(round(dur * FPS)))
    # smooth zoom 1.00 -> 1.00+ZOOM_PCT across N frames
    z_expr = f"min(1+{ZOOM_PCT}, 1+{ZOOM_PCT}*n/{max(1, frames-1)})"

    # Build filtergraph:
    # [0:v] scale->pad -> zoompan (centered) -> setsar -> label [v]
    vf = (
        f"[0:v]"
        f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:{PAD_COLOR},"
        f"zoompan=z={z_expr}:x=(iw-iw/zoom)/2:y=(ih-ih/zoom)/2:d=1:fps={FPS},"
        f"setsar=1[v]"
    )

    use_nv = _nvenc_available()
    use_nv = False
    vcodec = CODEC_NV if use_nv else "libx264"
    vparams = (["-rc:v", "vbr_hq", "-cq:v", str(NVENC_CQ), "-b:v", "0"]
               if use_nv else ["-crf", "19", "-preset", "medium"])

    cmd = [
        FFMPEG, "-y",
        # turn the still into a proper frame stream for zoompan
        "-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.6f}", "-i", str(image_path),
        "-i", str(audio_path),
        "-filter_complex", vf,
        "-map", "[v]", "-map", "1:a",
        "-c:v", vcodec,
        "-preset", NVENC_PRESET, *vparams,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        "-movflags", "+faststart",
        "-r", str(FPS),
        str(out_path),
    ]
    return cmd


def create_video_ffmpeg(image_path: Path, audio_path: Path, output_path: Path) -> Path:
    out_path = _resolve_output_path(output_path, audio_path)
    cmd = make_cmd(image_path, audio_path, out_path)
    # Tip while debugging: remove stdout/stderr redirection to see FFmpeg’s log.
    subprocess.run(cmd, check=True)
    return out_path

def _one_job(audio_file: Path) -> Path:
    return create_video_ffmpeg(BACKGROUND_IMAGE_PATH, audio_file, BASIC_VIDEO_OUTPUT_DIR)

if __name__ == "__main__":
    mp3_files = sorted(PROCESSED_AUDIO_DIR.glob("*.mp3"))
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_one_job, f) for f in mp3_files]
        for fut in as_completed(futures):
            try:
                print("Done:", fut.result().name)
            except Exception as e:
                print("Error:", e)
