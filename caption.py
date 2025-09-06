import os
import argparse
import subprocess
from pathlib import Path

from pydub import AudioSegment
from openai import OpenAI
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.VideoClip import ColorClip

# --- Configuration ---
MAX_CHUNK_SIZE = 24 * 1024 * 1024  # 24 MB
CHUNK_LENGTH_MS = 5 * 60 * 1000     # 5 minutes
GROUP_SIZE = 3                      # words per caption segment
MAX_CHARS_PER_SEGMENT = 20  # Try 32-40 for mobile video


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

def transcribe_chunks(chunks):
    words = []
    for path, offset in chunks:
        print(f"[INFO] Transcribing {path.name} (offset {offset}s)")
        with open(path, "rb") as af:
            resp = client.audio.transcriptions.create(
                file=af,
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=["segment", "word"]
            )
        resp_data = resp.model_dump()
        print("[DEBUG] Raw transcription response:")
        print(resp_data)

        for w in resp_data.get("words", []):  # FIX: top-level words
            words.append({
                "start": w["start"] + offset,
                "end":   w["end"] + offset,
                "word":  w["word"]
            })

    print(f"[INFO] Total words transcribed: {len(words)}")
    return sorted(words, key=lambda x: x["start"])


def group_into_segments(words, max_chars=32):
    segments = []
    group = []
    char_count = 0
    for w in words:
        word_text = w["word"].strip()
        if not all(k in w for k in ("start", "end", "word")) or not word_text:
            print(f"[WARNING] Skipping invalid word entry: {w}")
            continue

        # Account for a space before the word except at the start of the segment
        added_length = len(word_text) + (1 if group else 0)
        if char_count + added_length > max_chars and group:
            # Finish current segment
            start = group[0]["start"]
            end = group[-1]["end"]
            text = " ".join(gw["word"].strip() for gw in group)
            segments.append((start, end, text))
            group = []
            char_count = 0

        group.append(w)
        char_count += added_length

    # Add the last group if any
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
    base_fs = int(video.h / 50)
    padding = base_fs // 2

    safe_width = int(video.w * 0.9)  # 90% of video width to avoid edge cutoff
    fontname = "/usr/local/share/fonts/MREARLN.TTF"

    for start, end, txt in segments:
        duration = end - start
        fontsize = int(base_fs * 2.5)  # Slightly larger for reels

        pos_y = int(video.h * 2 / 3) + padding

        txt_clip = TextClip(
            txt.upper(),
            fontsize=fontsize,
            color="white",
            font=fontname,
            size=(safe_width, None),        # Wraps text at safe width
            stroke_color="black",
            stroke_width=2,                 # Thicker outline
            method="caption"
        ).set_start(start).set_duration(duration)

        txt_clip = txt_clip.set_position(("center", pos_y))
        clips.append(txt_clip)

    final = CompositeVideoClip(clips)
    final.write_videofile(str(output_path), codec="libx264", audio_codec="aac")


def main():
    parser = argparse.ArgumentParser(description="Caption video using OpenAI Whisper API and MoviePy.")
    parser.add_argument("video", help="Path to the video file.")
    parser.add_argument("--output", help="Output mp4 path.", default="output-captioned.mp4")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_video = Path(args.output)
    wav_path = video_path.with_suffix('.wav')
    transcripts_dir = video_path.parent / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)

    extract_audio(video_path, wav_path)
    chunks = split_audio(wav_path)
    words = transcribe_chunks(chunks)
    cap_segs = group_into_segments(words, max_chars=32)

    txt_file = transcripts_dir / f"{video_path.stem}-captions.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        for s, e, t in cap_segs:
            line = f"{s:.2f} --> {e:.2f}\n{t}\n\n"
            print(f"[DEBUG] Writing caption: {line.strip()}")
            f.write(line)
        f.flush()

    print(f"Transcript written to {txt_file.resolve()}")

    add_captions(video_path, cap_segs, out_video)
    print(f"Captioned video saved to {out_video}")

if __name__ == "__main__":
    main()
