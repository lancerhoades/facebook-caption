import os
import re
import argparse
import subprocess
from pathlib import Path

from pydub import AudioSegment
from openai import OpenAI
os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# --- Configuration ---
MAX_CHUNK_SIZE = 24 * 1024 * 1024  # 24 MB
CHUNK_LENGTH_MS = 5 * 60 * 1000    # 5 minutes
MAX_CHARS_PER_SEGMENT = 32         # 32–40 works well for mobile

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")
client = OpenAI(api_key=api_key)

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

def _supports_verbose_json(model: str) -> bool:
    # whisper-1 supports verbose_json (words/segments). 4o-transcribe models do not.
    return model == "whisper-1"

def _evenly_time_words(text: str, chunk_seconds: float, offset: float):
    """
    For models that don't return word timestamps, approximate them by
    distributing words evenly across the chunk duration.
    """
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
            # Choose response_format based on model capabilities
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
            # Prefer top-level words; fall back to segments[].words if present
            wl = data.get("words") or [
                w for seg in (data.get("segments") or [])
                for w in (seg.get("words") or [])
            ]
            if not wl:
                print("[WARN] verbose_json contained no word timestamps; falling back to segment spans.")
                # Fallback: approximate across full chunk if only 'text' exists
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
            # 4o-transcribe family: JSON without timestamps. Approximate evenly.
            text = data.get("text", "") or ""
            chunk_len_s = AudioSegment.from_wav(str(path)).duration_seconds
            approx = _evenly_time_words(text, chunk_len_s, offset)
            if not approx:
                print("[WARN] Empty transcript for chunk; skipping.")
            words.extend(approx)

    print(f"[INFO] Total words collected: {len(words)}")
    return sorted(words, key=lambda x: x["start"])

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

def add_captions(video_path: Path, segments, output_path: Path):
    video = VideoFileClip(str(video_path))
    clips = [video]

    # Dynamic sizing for mobile/video aspect ratios
    base_fs = max(14, int(video.h / 50))
    padding = base_fs // 2

    safe_width = int(video.w * 0.9)  # 90% of video width
    font_path = Path(__file__).parent / "fonts" / "MREARLN.TTF"
    font_arg = str(font_path) if font_path.exists() else "Arial"

    for start, end, txt in segments:
        duration = max(0.05, end - start)
        fontsize = int(base_fs * 2.5)  # slightly larger for reels
        pos_y = int(video.h * 2 / 3) + padding

        txt_clip = TextClip(
            txt.upper(),
            fontsize=fontsize,
            color="white",
            font=font_arg,
            size=(safe_width, None),        # wrap text at safe width
            stroke_color="black",
            stroke_width=2,
            method="caption"
        ).set_start(start).set_duration(duration)

        txt_clip = txt_clip.set_position(("center", pos_y))
        clips.append(txt_clip)

    final = CompositeVideoClip(clips)
    final.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

def main():
    parser = argparse.ArgumentParser(
        description="Caption a video using OpenAI STT (Whisper/4o-transcribe) and MoviePy."
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
