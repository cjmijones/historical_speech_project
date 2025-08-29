from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from moviepy import VideoFileClip
from faster_whisper import WhisperModel


def extract_audio_wav16k(video_path: str | Path, audio_path: str | Path) -> None:
    """
    Extract 16 kHz mono PCM WAV using MoviePy/FFmpeg.
    """
    video_path = str(video_path)
    audio_path = str(audio_path)
    with VideoFileClip(video_path) as vid:
        if vid.audio is None:
            raise RuntimeError("Input video has no audio track.")
        # 16 kHz mono PCM S16LE for Whisper/faster-whisper
        vid.audio.write_audiofile(
            audio_path,
            fps=16000,
            nbytes=2,
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"],
            logger=None,
        )


def pick_compute_type(device: str) -> str:
    """
    Choose a good default precision for faster-whisper.
    """
    if device == "cuda":
        return "float16"  # fastest & accurate on NVIDIA GPUs
    # For CPU, int8 or int8_float16 are solid tradeoffs.
    return "int8"


def transcribe_to_word_json(
    audio_path: str | Path,
    out_json_path: str | Path,
    model_name: str = "base.en",
    device: str = "auto",
    compute_type: str | None = None,
    beam_size: int = 1,
    vad_filter: bool = True,
) -> None:
    """
    Run faster-whisper and save a flat list of word-timestamp dicts
    compatible with your downstream pipeline.
    """
    audio_path = str(audio_path)
    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        # Try CUDA first; fall back to CPU silently if CUDA isn't available
        try:
            import torch  # optional; just to probe CUDA
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if compute_type is None:
        compute_type = pick_compute_type(device)

    # Load model
    model = WhisperModel(
        model_name,
        device=device,            # "cuda" or "cpu"
        compute_type=compute_type # "float16" (GPU) | "int8"/"int8_float16" (CPU)
    )

    # Transcribe with word-timestamps
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=vad_filter,
        beam_size=beam_size,
        # You can pass language="en" or task="transcribe" if your content is fixed
    )

    word_timestamps = []
    position_index = 0

    for seg in segments:
        # .words is already tokenized with start/end per word
        if not seg.words:
            continue
        for w in seg.words:
            # Normalize casing and copy fields to your schema
            word_timestamps.append({
                "word": (w.word or "").lower(),
                "start": float(w.start) if w.start is not None else None,
                "end": float(w.end) if w.end is not None else None,
                "position": position_index,
                "matched": False
            })
            position_index += 1

    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(word_timestamps, f, indent=2)

    print(f"Saved {len(word_timestamps)} words to {out_json_path.resolve()}")
    if device == "cuda":
        print(f"[faster-whisper] device=cuda, compute_type={compute_type}")
    else:
        print(f"[faster-whisper] device=cpu, compute_type={compute_type}")


def prepare_file_for_adding_captions_n_headings_thru_html(
    input_video_path: str | Path = "input_video.mp4",
    out_json_path: str | Path = "temp/word_timestamps.json",
    model_name: str = "base.en",
    device: str = "auto",
    compute_type: str | None = None,
    beam_size: int = 1,
    vad_filter: bool = True,
    keep_audio: bool = False,
    audio_path: str | Path | None = None,
) -> None:
    """
    High-level wrapper:
      1) Extract 16k mono WAV
      2) Transcribe with faster-whisper
      3) Save word timestamps to user-specified JSON path
    """
    input_video_path = Path(input_video_path)
    out_json_path = Path(out_json_path)

    # Default audio path next to JSON (but user can override)
    if audio_path is None:
        audio_path = out_json_path.with_suffix(".wav")
    audio_path = Path(audio_path)

    try:
        extract_audio_wav16k(input_video_path, audio_path)
        print(f"Audio extracted -> {audio_path.resolve()}")
    except Exception as e:
        raise RuntimeError(f"Error extracting audio: {e}") from e

    transcribe_to_word_json(
        audio_path=audio_path,
        out_json_path=out_json_path,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    if not keep_audio and audio_path.exists():
        try:
            audio_path.unlink()
        except Exception:
            # Non-fatal; you can keep the WAV if deletion fails
            pass


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract word-level timestamps from a video using faster-whisper."
    )
    p.add_argument("--video", "-i", type=str, required=False, default="input_video.mp4",
                   help="Path to input video.")
    p.add_argument("--out", "-o", type=str, required=False, default="temp/word_timestamps.json",
                   help="Path to output JSON (you choose dir/name).")
    p.add_argument("--model", type=str, default="base.en",
                   help="faster-whisper model name (e.g., tiny, base, base.en, small, medium, large-v3).")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device selection. 'auto' picks cuda if available.")
    p.add_argument("--compute_type", type=str, default=None,
                   help="Override precision (e.g., float16, int8, int8_float16). Default chosen by device.")
    p.add_argument("--beam_size", type=int, default=1, help="Beam size for decoding (1 = greedy).")
    p.add_argument("--no_vad", action="store_true", help="Disable VAD filter.")
    p.add_argument("--keep_audio", action="store_true", help="Keep extracted WAV file.")
    p.add_argument("--audio_path", type=str, default=None,
                   help="Optional path to save the extracted WAV (defaults next to --out).")
    return p


def main():
    args = _build_argparser().parse_args()
    prepare_file_for_adding_captions_n_headings_thru_html(
        input_video_path=args.video,
        out_json_path=args.out,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        keep_audio=args.keep_audio,
        audio_path=args.audio_path,
    )


if __name__ == "__main__":
    main()
