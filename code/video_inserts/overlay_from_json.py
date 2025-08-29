#!/usr/bin/env python3
"""
Overlay images on a base 1080x1920 video using a JSON "inserts" file.
Compatibility: MoviePy 2.2.1

- Positions: top_left, top_right, bottom_left, bottom_right, center
- layout.scale: fraction of *video width* used for the overlay's width (0–1)
- layout.margin: [x, y] pixel margins from the chosen corner

Usage (CLI):
  python overlay_from_json_v221.py --video_in input.mp4 \
                                   --json_in inserts.json \
                                   --video_out output.mp4

Or import the helpers in a notebook and call directly.

Requires: moviepy>=2.2.1 (pip install moviepy)
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import argparse
import numpy as np

# New v2-style imports
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip

# ------------------------- Data models -------------------------
@dataclass
class Layout:
    pos: str = "top_left"  # top_left|top_right|bottom_left|bottom_right|center
    scale: float = 0.6     # fraction of video width for overlay width
    margin: Tuple[int, int] = (24, 24)

@dataclass
class Insert:
    t_start: float
    t_end: float
    type: str
    asset: Path
    layout: Layout

# ------------------------- Helpers -------------------------

def _parse_inserts(json_path: Path) -> Tuple[int, List[Insert]]:
    obj = json.loads(Path(json_path).read_text(encoding="utf-8"))
    fps = int(obj.get("fps", 24))
    inserts: List[Insert] = []
    for item in obj.get("inserts", []):
        layout = item.get("layout", {})
        ins = Insert(
            t_start=float(item["t_start"]),
            t_end=float(item["t_end"]),
            type=item.get("type", "image_overlay"),
            asset=Path(item["asset"]).expanduser().resolve(),
            layout=Layout(
                pos=layout.get("pos", "top_left"),
                scale=float(layout.get("scale", 0.6)),
                margin=tuple(layout.get("margin", [24, 24]))[:2],
            ),
        )
        inserts.append(ins)
    return fps, inserts


def _pos_for_corner(
    corner: str, margin: Tuple[int, int], video_w: int, video_h: int, 
    overlay_w: int, overlay_h: int,
):
    mx, my = margin
    corner = corner.lower()
    if corner == "top_left":
        return (mx, my)
    if corner == "top_right":
        return (video_w - overlay_w - mx, my)
    if corner == "bottom_left":
        return (mx, video_h - overlay_h - my)
    if corner == "bottom_right":
        return (video_w - overlay_w - mx, video_h - overlay_h - my)
    if corner == "center":
        return ((video_w - overlay_w) // 2, (video_h - overlay_h) // 2)
    return (mx, my)


def _resize_imageclip_to_width(img_clip: ImageClip, target_w: int) -> ImageClip:
    """Resize an ImageClip to target_w (keeping aspect) using a single image_transform."""
    @img_clip.image_transform
    def _do_resize(arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        if w == target_w:
            return arr
        new_h = max(1, int(round(h * (target_w / float(w)))))
        return np.array(Image.fromarray(arr).resize((target_w, new_h), Image.LANCZOS))
    return _do_resize  # returns a new ImageClip

# ------------------------- Main pipeline -------------------------

def build_overlays(
    base_w: int, base_h: int, inserts: List[Insert]
) -> List[ImageClip]:
    overlays: List[ImageClip] = []
    for ins in inserts:
        if ins.type != "image_overlay":
            continue
        if not ins.asset.exists():
            print(f"[WARN] Missing asset: {ins.asset}")
            continue
        # Duration and base sizing
        duration = max(0.0, ins.t_end - ins.t_start)
        target_w = int(max(1, min(base_w, base_w * ins.layout.scale)))

        img = ImageClip(str(ins.asset), duration=duration)  # v2.2.1: duration in constructor
        img = _resize_imageclip_to_width(img, target_w)
        ow, oh = img.size

        # Position
        pos = _pos_for_corner(ins.layout.pos, ins.layout.margin, base_w, base_h, ow, oh)

        # Time placement on the timeline (v2.2.1: use with_* methods)
        img = img.with_start(ins.t_start).with_end(ins.t_end).with_position(pos)

        overlays.append(img)
    return overlays


def main():
    ap = argparse.ArgumentParser(description="Overlay images from JSON onto a video (MoviePy 2.2.1)")
    ap.add_argument("--video_in", required=True, type=Path)
    ap.add_argument("--json_in", required=True, type=Path)
    ap.add_argument("--video_out", required=True, type=Path)
    ap.add_argument("--codec", default="libx264", help="libx264|libx265|h264_nvenc|hevc_nvenc")
    ap.add_argument("--preset", default="medium", help="x264/x265 -preset or NVENC -preset (e.g., p5)")
    ap.add_argument("--crf", type=float, default=18.0, help="CRF/quality (x264/x265) or CQ (NVENC)")
    ap.add_argument("--audio", action="store_true", help="Keep original audio from the base video")
    args = ap.parse_args()

    base = VideoFileClip(str(args.video_in))
    base_w, base_h = base.size

    fps, inserts = _parse_inserts(args.json_in)
    if not inserts:
        raise SystemExit("No inserts found in JSON.")

    overlays = build_overlays(base_w, base_h, inserts)
    comp = CompositeVideoClip([base] + overlays, size=(base_w, base_h))

    out_fps = fps or (base.fps if getattr(base, "fps", None) else 24)

    # Build ffmpeg params compatible with MoviePy 2.2.1
    ffmpeg_params = ["-pix_fmt", "yuv420p"]
    if args.codec in ("libx264", "libx265"):
        ffmpeg_params = ["-preset", args.preset, "-crf", str(args.crf)] + ffmpeg_params
    else:
        # NVENC path or other encoders
        ffmpeg_params = ["-preset", args.preset, "-cq", str(int(args.crf)), "-b:v", "0"] + ffmpeg_params

    comp.write_videofile(
        str(args.video_out),
        codec=args.codec,
        fps=out_fps,
        audio=bool(args.audio),
        ffmpeg_params=ffmpeg_params,
    )

    comp.close(); base.close()


if __name__ == "__main__":
    main()
