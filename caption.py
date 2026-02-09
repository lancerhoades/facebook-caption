import os
import re
import argparse
import subprocess
import string
from pathlib import Path

from pydub import AudioSegment
from openai import OpenAI
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip, vfx

# --- Configuration ---
MAX_CHUNK_SIZE = 24 * 1024 * 1024  # 24 MB
CHUNK_LENGTH_MS = 60 * 1000        # 60 seconds for tighter alignment at chunk edges
MAX_WORDS_PER_SEGMENT = 3          # hard cap of 3 spoken words on screen

# Safe-zone configuration (percentages of width/height)
SAFEZONE_TOP_PCT = float(os.getenv("SAFEZONE_TOP_PCT", "0.14"))
SAFEZONE_BOTTOM_PCT = float(os.getenv("SAFEZONE_BOTTOM_PCT", "0.35"))
SAFEZONE_SIDE_PCT = float(os.getenv("SAFEZONE_SIDE_PCT", "0.06"))
SAFEZONE_PAD_PCT = float(os.getenv("SAFEZONE_PAD_PCT", "0.02"))
SAFEZONE_DEBUG = os.getenv("SAFEZONE_DEBUG", "false").lower() in ("1", "true", "yes", "on")
SAFEZONE_ENFORCE = os.getenv("SAFEZONE_ENFORCE", "true").lower() in ("1", "true", "yes", "on")

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")
client = OpenAI(api_key=api_key)

# ---------- Audio helpers ----------

def extract_audio(video_path: Path, wav_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(wav_path)
    ]
    subprocess.run(cmd, check=True)

def split_audio(wav_path: Path):
    audio = AudioSegment.from_wav(str(wav_path))
    chunks = []
    start_ms = 0
    idx = 0

    while start_ms < len(audio):
        segment = audio[start_ms:start_ms + CHUNK_LENGTH_MS]
        out_path = wav_path.parent / f"{wav_path.stem}_chunk{idx}.wav"
        segment.export(str(out_path), format="wav")

        if out_path.stat().st_size <= MAX_CHUNK_SIZE:
            chunks.append((out_path, start_ms / 1000.0))
        else:
            half = audio[start_ms:start_ms + CHUNK_LENGTH_MS // 2]
            out2 = wav_path.parent / f"{wav_path.stem}_chunk{idx}_smaller.wav"
            half.export(str(out2), format="wav")
            if out2.stat().st_size <= MAX_CHUNK_SIZE:
                chunks.append((out2, start_ms / 1000.0))
            else:
                print(f"[WARNING] Chunk {out2.name} still too large: {out2.stat().st_size}")
        start_ms += CHUNK_LENGTH_MS
        idx += 1

    return chunks

# ---------- Transcription ----------

def _supports_verbose_json(model: str) -> bool:
    # whisper-1 supports verbose_json (words/segments). 4o-transcribe models do not.
    return model == "whisper-1"

def _evenly_time_words(text: str, chunk_seconds: float, offset: float):
    """Approximate per-word timings by distributing evenly across the chunk."""
    # Count only real word tokens (ignore standalone punctuation/hyphens)
    tokens = re.findall(r"[A-Za-z0-9']+", text)
    tokens = [t for t in tokens if t.strip()]
    if not tokens or chunk_seconds <= 0:
        return []
    per_word = chunk_seconds / len(tokens)
    out = []
    for i, tok in enumerate(tokens):
        start = offset + i * per_word
        end = offset + (i + 1) * per_word
        out.append({"start": start, "end": end, "word": tok})
    return out

def transcribe_chunks(chunks, model: str, language: str | None = None):
    words = []
    verbose = _supports_verbose_json(model)
    for path, offset in chunks:
        print(f"[INFO] Transcribing {path.name} (offset {offset:.2f}s) with model={model}")
        with open(path, "rb") as af:
            if verbose:
                resp = client.audio.transcriptions.create(
                    file=af,
                    model=model,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],  # word-level timing for perfect sync
                    **({"language": language} if language else {})
                )
            else:
                resp = client.audio.transcriptions.create(
                    file=af,
                    model=model,
                    response_format="json",
                    **({"language": language} if language else {})
                )

        data = resp if isinstance(resp, dict) else resp.model_dump()
        print("[DEBUG] Raw transcription response keys:", list(data.keys()))

        if verbose:
            wl = data.get("words") or []
            if not wl:
                print("[WARN] verbose_json contained no word timestamps; approximating.")
                chunk_len_s = AudioSegment.from_wav(str(path)).duration_seconds
                wl = _evenly_time_words(data.get("text", ""), chunk_len_s, 0.0)
            for w in wl:
                if {"start", "end", "word"} <= set(w.keys()):
                    words.append({
                        "start": float(w["start"]) + offset,
                        "end": float(w["end"]) + offset,
                        "word": str(w["word"])
                    })
        else:
            text = data.get("text", "") or ""
            chunk_len_s = AudioSegment.from_wav(str(path)).duration_seconds
            approx = _evenly_time_words(text, chunk_len_s, offset)
            if not approx:
                print("[WARN] Empty transcript for chunk; skipping.")
            words.extend(approx)

    print(f"[INFO] Total words collected: {len(words)}")
    return sorted(words, key=lambda x: x["start"])

# ---------- Grouping (strict ≤3 words per tile, punctuation-safe) ----------

_PUNCT = set(string.punctuation)

def _attach_punct(display_words):
    """
    Attach punctuation tokens to the previous word for visual rendering
    (no timing changes).
    """
    out = []
    for w in display_words:
        txt = w["word"]
        if txt in _PUNCT and out:
            out[-1]["word"] = out[-1]["word"] + txt
        else:
            out.append(w)
    return out

def group_into_segments(words, max_words: int = MAX_WORDS_PER_SEGMENT):
    """
    Build caption segments of up to `max_words` real words (ignoring punctuation-only tokens).
    Each segment’s timing is [first_word.start, last_word.end].
    """
    # 1) Filter/normalize
    cleaned = []
    for w in words:
        if not all(k in w for k in ("start", "end", "word")):
            continue
        txt = str(w["word"]).strip()
        if not txt:
            continue
        cleaned.append({"start": float(w["start"]), "end": float(w["end"]), "word": txt})

    # 2) Attach punctuation visually
    vis_words = _attach_punct(cleaned)

    # 3) Build segments
    segments = []
    group = []
    real_count = 0

    def flush():
        nonlocal group, real_count
        if not group:
            return
        start = group[0]["start"]
        end = group[-1]["end"]
        text = " ".join(g["word"] for g in group)
        segments.append((start, end, text))
        group = []
        real_count = 0

    for w in vis_words:
        is_real = not (len(w["word"]) == 1 and w["word"] in _PUNCT)
        if is_real and real_count == max_words:
            flush()
        group.append(w)
        if is_real:
            real_count += 1

    flush()
    print(f"[INFO] Total segments (≤{max_words} words each): {len(segments)}")
    return segments

# ---------- Caption drawing (Pillow) ----------

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """
    Load preferred TTF; fallback to DejaVuSans available in Debian slim images.
    """
    candidates = [
        "/usr/local/share/fonts/MREARLN.TTF",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            key = (p, font_size)
            if key not in _FONT_CACHE:
                _FONT_CACHE[key] = ImageFont.truetype(p, font_size)
            return _FONT_CACHE[key]
    # Last resort: PIL default bitmap font (no size scaling)
    return ImageFont.load_default()

def _safezone_rect(video_w: int, video_h: int):
    left = int(video_w * SAFEZONE_SIDE_PCT)
    right = int(video_w * (1.0 - SAFEZONE_SIDE_PCT))
    top = int(video_h * SAFEZONE_TOP_PCT)
    bottom = int(video_h * (1.0 - SAFEZONE_BOTTOM_PCT))
    return left, top, right, bottom

def _safezone_debug_overlay(video_w: int, video_h: int) -> Image.Image:
    left, top, right, bottom = _safezone_rect(video_w, video_h)
    img = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # No-go regions
    d.rectangle([0, 0, video_w, top], fill=(255, 0, 0, 64))
    d.rectangle([0, bottom, video_w, video_h], fill=(255, 0, 0, 64))
    d.rectangle([0, top, left, bottom], fill=(255, 0, 0, 64))
    d.rectangle([right, top, video_w, bottom], fill=(255, 0, 0, 64))
    # Safe-zone outline
    border = max(2, int(video_h * 0.003))
    d.rectangle([left, top, right, bottom], outline=(0, 255, 0, 180), width=border)
    return img

def _bbox_within_safezone(x: int, y: int, w: int, h: int, safe_rect) -> bool:
    left, top, right, bottom = safe_rect
    return x >= left and y >= top and (x + w) <= right and (y + h) <= bottom

def _render_caption_image_singleline(text: str, safe_width: int, base_fontsize: int, padding_px: int = 12):
    """
    Render ALL-CAPS text on a semi-transparent rounded rectangle background (single line).
    Auto-shrinks font to fit within safe_width.
    """
    text = text.upper().strip()
    fontsize = base_fontsize
    for _ in range(8):
        font = _load_font(fontsize)
        tmp = Image.new("RGBA", (1, 1))
        d = ImageDraw.Draw(tmp)
        bbox = d.textbbox((0, 0), text, font=font, stroke_width=2)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w + 2 * padding_px <= safe_width:
            img = Image.new("RGBA", (w + 2 * padding_px, h + 2 * padding_px), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            # Rounded rectangle background
            bg_radius = max(6, int(h * 0.4))
            d.rounded_rectangle([0, 0, img.width, img.height], radius=bg_radius, fill=(0, 0, 0, 110))
            # Centered text
            d.text((padding_px, padding_px), text, font=font,
                   fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
            return img
        fontsize = max(12, int(fontsize * 0.9))
    # Fallback (if still too wide)
    font = _load_font(fontsize)
    img = Image.new("RGBA", (min(w + 2 * padding_px, safe_width), h + 2 * padding_px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, img.width, img.height], radius=max(6, int(h * 0.4)), fill=(0, 0, 0, 110))
    d.text((padding_px, padding_px), text, font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return img

# ---------- Video compositor ----------

def add_captions(video_path: Path, segments, output_path: Path):
    video = VideoFileClip(str(video_path))
    clips = [video]

    base_fs = max(14, int(video.h / 50))
    padding = base_fs // 2
    safe_left, safe_top, safe_right, safe_bottom = _safezone_rect(video.w, video.h)
    safe_width = safe_right - safe_left
    safe_pad = int(video.h * SAFEZONE_PAD_PCT)

    lead = 0.08  # 80 ms early for perceived sync
    tail = 0.06  # 60 ms linger after last phoneme
    fade_ms = 90
    fade_s = fade_ms / 1000.0

    if SAFEZONE_DEBUG:
        overlay = _safezone_debug_overlay(video.w, video.h)
        clips.append(
            ImageClip(np.array(overlay), transparent=True).set_duration(video.duration)
        )

    for start, end, txt in segments:
        adj_start = max(0, start - lead)
        adj_end = end + tail
        duration = max(0.05, adj_end - adj_start)
        fontsize = int(base_fs * 2.5)  # larger for reels/shorts
        pil_img = _render_caption_image_singleline(txt, safe_width, fontsize)
        np_frame = np.array(pil_img)
        target_bottom = safe_bottom - safe_pad
        pos_y = max(safe_top, min(int(target_bottom - pil_img.height), safe_bottom - pil_img.height))
        pos_x = int((video.w - pil_img.width) / 2)

        if SAFEZONE_ENFORCE:
            if pil_img.height > (safe_bottom - safe_top):
                raise RuntimeError("Caption height exceeds safe-zone height; reduce font size.")
            if not _bbox_within_safezone(
                pos_x, pos_y, pil_img.width, pil_img.height,
                (safe_left, safe_top, safe_right, safe_bottom)
            ):
                raise RuntimeError("Caption bbox intersects safe-zone no-go areas.")

        clip = (
            ImageClip(np_frame, transparent=True)
            .set_start(adj_start)
            .set_duration(duration)
            .set_position((pos_x, pos_y))
            .fx(vfx.fadein, fade_s)
            .fx(vfx.fadeout, fade_s)
        )
        clips.append(clip)

    final = CompositeVideoClip(clips)
    final.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=video.fps,  # pin FPS to avoid micro-drift
        preset="medium",
        ffmpeg_params=["-movflags", "+faststart"]
    )

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Caption a video using OpenAI STT (Whisper/4o-transcribe) and Pillow (no ImageMagick)."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("--output", default="output-captioned.mp4", help="Output mp4 path.")
    parser.add_argument(
        "--model",
        default=os.getenv("TRANSCRIBE_MODEL", "whisper-1"),
        choices=["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
        help="Transcription model. Use whisper-1 for accurate word timestamps."
    )
    parser.add_argument(
        "--language",
        default=None,
        help="ISO-639-1 code for the input language (e.g., 'en'). Improves accuracy/latency."
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=MAX_WORDS_PER_SEGMENT,
        help="Max spoken words per on-screen caption segment."
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    out_video = Path(args.output)
    wav_path = video_path.with_suffix(".wav")
    transcripts_dir = video_path.parent / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    extract_audio(video_path, wav_path)
    chunks = split_audio(wav_path)
    words = transcribe_chunks(chunks, model=args.model, language=args.language)
    cap_segs = group_into_segments(words, max_words=args.max_words)

    txt_file = transcripts_dir / f"{video_path.stem}-captions.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        for s, e, t in cap_segs:
            line = f"{s:.2f} --> {e:.2f}\n{t}\n\n"
            print(f"[DEBUG] Writing caption: {line.strip()}")
            f.write(line)

    print(f"Transcript written to {txt_file.resolve()}")
    add_captions(video_path, cap_segs, out_video)
    print(f"Captioned video saved to {out_video}")

if __name__ == "__main__":
    main()
