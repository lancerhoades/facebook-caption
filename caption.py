import os
import re
import argparse
import subprocess
from pathlib import Path

from pydub import AudioSegment
from openai import OpenAI
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip

# --- Configuration ---
MAX_CHUNK_SIZE = 24 * 1024 * 1024  # 24 MB
CHUNK_LENGTH_MS = 5 * 60 * 1000    # 5 minutes
MAX_CHARS_PER_SEGMENT = 32         # 32–40 works well for mobile

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
    tokens = re.findall(r"[A-Za-z0-9']+|-+", text)
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
                    timestamp_granularities=["segment", "word"],
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
            wl = data.get("words") or [
                w for seg in (data.get("segments") or [])
                for w in (seg.get("words") or [])
            ]
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

# ---------- Grouping ----------

def group_into_segments(words, max_chars=32):
    segments = []
    group = []
    char_count = 0
    for w in words:
        word_text = str(w.get("word", "")).strip()
        if not all(k in w for k in ("start", "end", "word")) or not word_text:
            print(f"[WARNING] Skipping invalid word entry: {w}")
            continue

        added_length = len(word_text) + (1 if group else 0)
        if char_count + added_length > max_chars and group:
            start = group[0]["start"]
            end = group[-1]["end"]
            text = " ".join(gw["word"].strip() for gw in group)
            segments.append((start, end, text))
            group = []
            char_count = 0

        group.append(w)
        char_count += added_length

    if group:
        start = group[0]["start"]
        end = group[-1]["end"]
        text = " ".join(gw["word"].strip() for gw in group)
        segments.append((start, end, text))
    print(f"[INFO] Total segments: {len(segments)}")
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

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int):
    """
    Simple greedy wrap by words to fit max_width. Returns list of lines and max line width.
    """
    words = text.split()
    lines = []
    cur = []
    max_w = 0
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        w_box = draw.textbbox((0, 0), trial, font=font, stroke_width=2)
        w_width = w_box[2] - w_box[0]
        if cur and w_width > max_width:
            line = " ".join(cur)
            lines.append(line)
            max_w = max(max_w, draw.textbbox((0, 0), line, font=font, stroke_width=2)[2])
            cur = [w]
        else:
            cur.append(w)
    if cur:
        line = " ".join(cur)
        lines.append(line)
        max_w = max(max_w, draw.textbbox((0, 0), line, font=font, stroke_width=2)[2])
    return lines, min(max_w, max_width)

def _render_caption_image(text: str, safe_width: int, base_fontsize: int):
    """
    Render uppercase text with white fill and black stroke onto a transparent RGBA image.
    Auto-scales font down if needed to keep within safe_width.
    """
    text = text.upper()
    fontsize = base_fontsize
    for _ in range(6):  # try a few times to fit width
        font = _load_font(fontsize)
        tmp_img = Image.new("RGBA", (safe_width, base_fontsize * 4), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tmp_img)
        lines, max_line_w = _wrap_text(draw, text, font, safe_width)
        line_height = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
        spacing = max(4, int(fontsize * 0.25))
        total_h = len(lines) * line_height + (len(lines) - 1) * spacing
        if max_line_w <= safe_width:
            # render final image
            img = Image.new("RGBA", (safe_width, total_h + spacing * 2), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            y = spacing
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font, stroke_width=2)
                w = bbox[2] - bbox[0]
                x = (safe_width - w) // 2
                draw.text(
                    (x, y), line, font=font,
                    fill=(255, 255, 255, 255),
                    stroke_width=2, stroke_fill=(0, 0, 0, 255)
                )
                y += line_height + spacing
            return img
        fontsize = max(10, int(fontsize * 0.9))  # shrink and try again
    # Fallback: single line without wrap
    font = _load_font(fontsize)
    img = Image.new("RGBA", (safe_width, fontsize * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return img

# ---------- Video compositor ----------

def add_captions(video_path: Path, segments, output_path: Path):
    video = VideoFileClip(str(video_path))
    clips = [video]

    base_fs = max(14, int(video.h / 50))
    padding = base_fs // 2
    safe_width = int(video.w * 0.9)  # 90% of video width
    pos_y = int(video.h * 2 / 3) + padding

    for start, end, txt in segments:
        duration = max(0.05, end - start)
        fontsize = int(base_fs * 2.5)  # larger for reels
        pil_img = _render_caption_image(txt, safe_width, fontsize)
        np_frame = np.array(pil_img)
        clip = ImageClip(np_frame, transparent=True).set_start(start).set_duration(duration)
        clip = clip.set_position(("center", pos_y))
        clips.append(clip)

    final = CompositeVideoClip(clips)
    final.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

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
        "--max-chars",
        type=int,
        default=MAX_CHARS_PER_SEGMENT,
        help="Max characters per on-screen caption segment."
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
    cap_segs = group_into_segments(words, max_chars=args.max_chars)

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
