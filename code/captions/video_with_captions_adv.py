from moviepy import (
    TextClip, CompositeVideoClip, VideoFileClip, ColorClip
)
import json, os
from pathlib import Path

def _measure_space(font: str, fontsize: int, color: str) -> int:
    """
    Render ONE space once to get its width for this font/size.
    (MoviePy caches by text+font+size, so this is cheap.)
    """
    space_clip = TextClip(font=font, text=" ", font_size=fontsize, color=color)
    w, _ = space_clip.size
    return w

def resolve_font(font_spec: str) -> str:
    """
    Accepts either a family name like 'Arial-Bold' or a full path.
    Returns a path to a .ttf/.otf that Pillow can load.
    Falls back to common safe fonts if not found.
    """
    p = Path(font_spec)
    if p.exists():
        return str(p)

    # Common Windows fonts directory
    win_fonts = Path(r"C:\Windows\Fonts")
    candidates = []

    if win_fonts.exists():
        # try some reasonable matches for 'family[-style]'
        stem = font_spec.replace("-", "").replace(" ", "").lower()
        for ext in (".ttf", ".otf"):
            for fp in win_fonts.glob(f"*{ext}"):
                name = fp.stem.replace("-", "").replace(" ", "").lower()
                if stem in name:
                    candidates.append(fp)
        if candidates:
            return str(candidates[0])

        # common mappings (extend as you like)
        aliases = {
            "Arial": "arial.ttf",
            "Arial-Bold": "arialbd.ttf",
            "Times New Roman": "times.ttf",
            "Times-Bold": "timesbd.ttf",
            "Verdana": "verdana.ttf",
            "Calibri": "calibri.ttf",
        }
        if font_spec in aliases and (win_fonts / aliases[font_spec]).exists():
            return str(win_fonts / aliases[font_spec])

    # Fallbacks
    # Try DejaVu (often installed with matplotlib)
    for dv in (
        Path(os.getenv("WINDIR", r"C:\Windows")) / "Fonts" / "DejaVuSans.ttf",
        Path.home() / ".local" / "share" / "fonts" / "DejaVuSans.ttf",
    ):
        if dv.exists():
            return str(dv)

    # Last resort: Arial
    arial = win_fonts / "arial.ttf"
    if arial.exists():
        return str(arial)

    # Give up: return back the original (will raise a clear error)
    return font_spec

def layout_words(
    textJSON,
    framesize,
    font="Arial",
    color="white",
    highlight_color="yellow",
    stroke_color="black",
    stroke_width=2,
    bottom_margin_ratio=0.08,
    side_margin_ratio=0.10,
    fontsize_ratio=0.055,
    word_gap_px=0,
    bg_opacity=0,
    bg_color=(64, 64, 64),
):
    """
    Returns (clips, maybe_bg_clips) where clips includes:
      - base words (always visible for the line duration)
      - timed highlight words (visible only during each word's window)
    maybe_bg_clips is empty if bg_opacity <= 0
    """
    full_start = float(textJSON["start"])
    full_end = float(textJSON["end"])
    full_duration = full_end - full_start

    frame_w, frame_h = framesize
    fontsize = int(frame_h * fontsize_ratio)
    left_right_pad = int(frame_w * side_margin_ratio)
    max_line_w = frame_w - 2 * left_right_pad

    # vertical anchor near the bottom
    baseline_y = int(frame_h * (1.0 - bottom_margin_ratio))  # bottom edge
    # We will compute per-line y positions upward from baseline_y - total_block_h.

    # Pre-measure a single space and add user gap
    base_space_w = _measure_space(font, fontsize, color)
    space_w = base_space_w + int(word_gap_px)

    # First pass: compute word sizes + layout (no rendering yet)
    words = []
    for wj in textJSON["textcontents"]:
        word = wj["word"]
        start = float(wj["start"])
        end = float(wj["end"])
        dur = end - start
        # Render once to get size; TextClip is cached by text/font/size
        tmp = TextClip(font=font, text=word, font_size=fontsize, color=color,
                       stroke_color=stroke_color, stroke_width=stroke_width)
        w, h = tmp.size
        words.append({
            "word": word,
            "w": w, "h": h,
            "start": start, "end": end, "dur": dur
        })

    # Simple greedy line-wrap using measured widths
    lines = []  # list of lists of word dicts augmented with x offsets
    x = 0
    line = []
    line_h = words[0]["h"] if words else int(fontsize * 1.2)
    for i, wd in enumerate(words):
        need_w = (wd["w"] if not line else space_w + wd["w"])
        if x + need_w <= max_line_w or not line:
            # put on this line
            wd = dict(wd)  # copy so we can augment
            wd["x"] = x if not line else x + space_w
            wd["y"] = 0  # fill later
            x = wd["x"] + wd["w"]
            line_h = max(line_h, wd["h"])
            line.append(wd)
        else:
            # close previous line
            for wd2 in line:
                wd2["line_h"] = line_h
            lines.append(line)
            # start new
            line = []
            x = 0
            line_h = wd["h"]
            wd = dict(wd)
            wd["x"] = 0
            wd["y"] = 0
            line.append(wd)
            x = wd["w"]

    if line:
        for wd2 in line:
            wd2["line_h"] = line_h
        lines.append(line)

    # Vertical positions: stack lines upward from bottom
    # Compute total block height
    vgap = int(fontsize * 0.18)  # small inter-line gap
    total_h = 0
    for li, L in enumerate(lines):
        lh = max(wd["line_h"] for wd in L) if L else 0
        total_h += lh
        if li < len(lines) - 1:
            total_h += vgap

    block_top_y = baseline_y - total_h
    # Assign y per line
    y_cursor = block_top_y
    for L in lines:
        lh = max(wd["line_h"] for wd in L) if L else 0
        for wd in L:
            wd["y"] = y_cursor
        y_cursor += lh + vgap

    # Center the block horizontally
    # Find line widths to center each line
    clips = []
    bg_clips = []
    block_left_x = left_right_pad

    max_right = 0
    max_bottom = 0

    for L in lines:
        if not L:
            continue
        line_left = min(wd["x"] for wd in L)
        line_right = max(wd["x"] + wd["w"] for wd in L)
        line_w = line_right - line_left
        # center this line within [block_left_x, block_left_x + max_line_w]
        line_x_offset = block_left_x + (max_line_w - line_w) // 2

        # Create base (white) words for whole line duration
        for wd in L:
            base = (
                TextClip(
                    text=wd["word"],
                    font=font, font_size=fontsize, color=color,
                    stroke_color=stroke_color, stroke_width=int(round(stroke_width))
                )
                .with_start(full_start)
                .with_duration(full_duration)
                .with_position((line_x_offset + wd["x"], wd["y"]))
            )
            clips.append(base)

            # Highlight layer for the word only in its time window
            hl = (
                TextClip(
                    text=wd["word"],
                    font=font, font_size=fontsize, color=highlight_color,
                    stroke_color=stroke_color, stroke_width=int(round(stroke_width))
                )
                .with_start(wd["start"])
                .with_duration(wd["dur"])
                .with_position((line_x_offset + wd["x"], wd["y"]))
            )
            clips.append(hl)

            max_right = max(max_right, line_x_offset + wd["x"] + wd["w"])
            max_bottom = max(max_bottom, wd["y"] + wd["line_h"])

    # Optional background plate: one per whole block (faster than per-line); gated by opacity
    if bg_opacity > 0 and clips:
        # add a small padding
        pad = int(fontsize * 0.35)
        left = block_left_x
        right = max_right
        top = block_top_y
        height = max_bottom - top
        width = right - left

        plate = (
            ColorClip(size=(max(1, width + 2*pad), max(1, height + 2*pad)), color=bg_color)
            .with_opacity(bg_opacity)
            .with_start(full_start)
            .with_duration(full_duration)
            .with_position((left - pad, top - pad))
        )
        bg_clips.append(plate)

    return clips, bg_clips

def create_video_with_captions_adv(
    mp4_file: str = "./input_video.mp4",
    linelevel_timestamps: str = "./temp/word_line_timestamps.json",
    video_out_path: str = "./output.mp4",
    word_gap_px: int = 0,
    stroke_width: int = 2,
    bottom_margin_ratio: float = 0.08,
    side_margin_ratio: float = 0.10,
    bg_opacity: float = 0,
    bg_color: str = "64,64,64",
    font: str = "Copperplate-Gothic-Bold",
    font_color: str = "white",
    font_high_light_color: str = "Yellow",
    font_stroke_color: str = "Black",
    fontsize_ratio: float = 0.055,
    nvenc: bool = False,
    codec: str = "libx264",
    preset: str = "medium",
    fps: int = 24,
):

    font_path = resolve_font(font_spec=font)

        # Read JSON
    with open(linelevel_timestamps, "r", encoding="utf-8") as f:
        lines = json.load(f)

    input_video = VideoFileClip(mp4_file)
    frame_size = input_video.size

    # Parse bg color
    try:
        bg_tuple = tuple(int(v) for v in bg_color.split(","))
        if len(bg_tuple) != 3:
            raise ValueError
    except Exception:
        bg_tuple = (64, 64, 64)

    all_caption_layers = []

    # Build ALL word and (optional) background clips in a single pass
    for line in lines:
        word_clips, bg_clips = layout_words(
            line,
            framesize=frame_size,
            font=font_path,
            color=font_color,
            highlight_color=font_high_light_color,
            stroke_color=font_stroke_color,
            stroke_width=stroke_width,
            bottom_margin_ratio=bottom_margin_ratio,
            side_margin_ratio=side_margin_ratio,
            word_gap_px=word_gap_px,
            bg_opacity=bg_opacity,
            bg_color=bg_tuple,
            fontsize_ratio=fontsize_ratio
        )
        # Background first (if any), then words/highlights
        all_caption_layers.extend(bg_clips)
        all_caption_layers.extend(word_clips)

    final = CompositeVideoClip([input_video] + all_caption_layers).with_audio(input_video.audio)


    # Output controls
    codec = "h264_nvenc" if nvenc else codec
    ffmpeg_params = ["-preset", preset, "-movflags", "+faststart"]

    final.write_videofile(
        video_out_path,
        fps=fps,
        codec=codec,
        audio_codec="aac",
        ffmpeg_params=ffmpeg_params,
        threads=0  # let ffmpeg choose; usually fastest
    )

    return


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Faster caption renderer with spacing/background controls.")
    parser.add_argument("--mp4_file", type=str, default="./input_video.mp4")
    parser.add_argument("--linelevel_timestamps", type=str, default="./temp/word_line_timestamps.json")
    parser.add_argument("--video_out_path", type=str, default="./output.mp4")
    # new controls
    parser.add_argument("--word_gap_px", type=int, default=0,
                        help="Extra pixels added between words (on top of the natural space width).")
    parser.add_argument("--bg_opacity", type=float, default=0,
                        help="0 disables background; 0..1 sets plate opacity.")
    parser.add_argument("--bg_color", type=str, default="64,64,64",
                        help="Background RGB as 'R,G,B'")
    parser.add_argument("--font", type=str, default="Copperplate-Gothic-Bold")
    parser.add_argument("--fontsize_ratio", type=float, default=0.055,
                        help="Font size = ratio * video height.")
    parser.add_argument("--nvenc", action="store_true",
                        help="Use h264_nvenc if available (Windows/NVIDIA).")
    parser.add_argument("--codec", type=str, default="libx264",
                        help="Override codec (ignored if --nvenc).")
    parser.add_argument("--preset", type=str, default="medium",
                        help="ffmpeg preset (valid for libx264 or nvenc, e.g., slow, medium, fast, p4, p5).")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    # Read JSON
    with open(args.linelevel_timestamps, "r", encoding="utf-8") as f:
        lines = json.load(f)

    input_video = VideoFileClip(args.mp4_file)
    frame_size = input_video.size

    # Parse bg color
    try:
        bg_tuple = tuple(int(v) for v in args.bg_color.split(","))
        if len(bg_tuple) != 3:
            raise ValueError
    except Exception:
        bg_tuple = (64, 64, 64)

    all_caption_layers = []

    # Build ALL word and (optional) background clips in a single pass
    for line in lines:
        word_clips, bg_clips = layout_words(
            line,
            framesize=frame_size,
            font=args.font,
            color="white",
            highlight_color="yellow",
            stroke_color="black",
            stroke_width=2,
            word_gap_px=args.word_gap_px,
            bg_opacity=args.bg_opacity,
            bg_color=bg_tuple,
            fontsize_ratio=args.fontsize_ratio
        )
        # Background first (if any), then words/highlights
        all_caption_layers.extend(bg_clips)
        all_caption_layers.extend(word_clips)

    final = CompositeVideoClip([input_video] + all_caption_layers).set_audio(input_video.audio)

    # Output controls
    codec = "h264_nvenc" if args.nvenc else args.codec
    ffmpeg_params = ["-preset", args.preset, "-movflags", "+faststart"]

    final.write_videofile(
        args.video_out_path,
        fps=args.fps,
        codec=codec,
        audio_codec="aac",
        ffmpeg_params=ffmpeg_params,
        threads=0  # let ffmpeg choose; usually fastest
    )

if __name__ == "__main__":
    main()
