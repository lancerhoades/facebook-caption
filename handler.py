import traceback
import os, re, json, tempfile, subprocess, urllib.request, urllib.parse, unicodedata, pathlib
from PIL import Image, ImageDraw, ImageFont
from botocore.client import Config
import boto3, requests
import runpod

print("[BOOT] importing handler.py")

# --------- ENV (minimal & static) ---------
AWS_REGION     = os.getenv("AWS_REGION", "us-east-1")
AWS_S3_BUCKET  = os.getenv("AWS_S3_BUCKET")
S3_PREFIX_BASE = os.getenv("S3_PREFIX_BASE", "jobs")

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
FASTWH_ID      = os.getenv("RUNPOD_FASTWHISPER_ENDPOINT_ID")
FASTWH_VAD     = os.getenv("FASTWH_ENABLE_VAD", "true").lower() in ("1","true","yes","on")
FASTWH_WORDTS  = os.getenv("FASTWH_WORD_TIMESTAMPS", "true").lower() in ("1","true","yes","on")
LANG_HINT      = os.getenv("TRANSCRIBE_LANG") or None  # optional, e.g. "en"

# Choose backend: "fastwhisper" (default) or "openai" (uses caption.py)
CAPTION_BACKEND = os.getenv("CAPTION_BACKEND", "fastwhisper").lower()

# Caption styling (ASS force_style)
FONT_FAMILY      = os.getenv("FONT_FAMILY", "MisterEarl BT")
MAX_WORDS_PER_CU = int(os.getenv("MAX_WORDS_PER_CUE", "0"))     # optional srt reflow (awk) and passed to caption.py when backend=openai
MAX_CUE_DURATION = float(os.getenv("MAX_CUE_DURATION", "0"))    # optional srt reflow (awk)
WORD_GAP_SPLIT_SEC = float(os.getenv("WORD_GAP_SPLIT_SEC", "0.35"))
MIN_WORDS_PER_CUE  = int(os.getenv("MIN_WORDS_PER_CUE", "1"))
SAFEZONE_TOP_PCT = float(os.getenv("SAFEZONE_TOP_PCT", "0.14"))
SAFEZONE_BOTTOM_PCT = float(os.getenv("SAFEZONE_BOTTOM_PCT", "0.35"))
SAFEZONE_SIDE_PCT = float(os.getenv("SAFEZONE_SIDE_PCT", "0.06"))
SAFEZONE_PAD_PCT = float(os.getenv("SAFEZONE_PAD_PCT", "0.02"))
SAFEZONE_DEBUG = os.getenv("SAFEZONE_DEBUG", "false").lower() in ("1","true","yes","on")
SAFEZONE_ENFORCE = os.getenv("SAFEZONE_ENFORCE", "true").lower() in ("1","true","yes","on")
FONT_SIZE_PCT = float(os.getenv("FONT_SIZE_PCT", "0.05"))
CAPTION_FORCE_FASTWH = os.getenv("CAPTION_FORCE_FASTWHISPER", "false").lower() in ("1","true","yes","on")
WORD_HIGHLIGHT = os.getenv("WORD_HIGHLIGHT", "false").lower() in ("1","true","yes","on")
WORD_HIGHLIGHT_COLOR = os.getenv("WORD_HIGHLIGHT_COLOR", "#FFD400")
WORD_INACTIVE_COLOR = os.getenv("WORD_INACTIVE_COLOR", "#FFFFFF")
WORD_HIGHLIGHT_APPROX = os.getenv("WORD_HIGHLIGHT_APPROX", "true").lower() in ("1","true","yes","on")
CAPITALIZE_TERMS = os.getenv(
    "CAPITALIZE_TERMS",
    "God,Jesus,Christ,Lord,Holy Spirit,Holy Ghost,Bible,Scripture,Church,Gospel"
)

print(f"[CFG] S3_BUCKET={AWS_S3_BUCKET!r} region={AWS_REGION!r}")
print(f"[CFG] FASTWH id={FASTWH_ID} vad={FASTWH_VAD} word_ts={FASTWH_WORDTS} lang={LANG_HINT}")
print(f"[CFG] CAPTION_BACKEND={CAPTION_BACKEND}")

if not AWS_S3_BUCKET:
    raise RuntimeError("AWS_S3_BUCKET is required")
if CAPTION_BACKEND == "fastwhisper" and not (RUNPOD_API_KEY and FASTWH_ID):
    raise RuntimeError("RUNPOD_API_KEY and RUNPOD_FASTWHISPER_ENDPOINT_ID are required for fastwhisper backend")

# --------- S3 client ---------
s3 = boto3.client("s3", region_name=AWS_REGION, config=Config(s3={"addressing_style":"virtual"}))

def _key(job_id: str, *parts: str) -> str:
    safe = [p.strip("/").replace("\\","/") for p in parts if p]
    return "/".join([S3_PREFIX_BASE.strip("/"), job_id] + safe)

def _presign(key: str, expires=7*24*3600) -> str:
    return s3.generate_presigned_url("get_object", Params={"Bucket": AWS_S3_BUCKET, "Key": key}, ExpiresIn=expires)

def _upload_tmp_to_s3(path: str, key: str, content_type: str | None = None) -> dict:
    extra = {"ContentType": content_type} if content_type else {}
    s3.upload_file(path, AWS_S3_BUCKET, key, ExtraArgs=extra)
    return {"key": key, "url": _presign(key)}

# --------- utils ---------
def _has_ffmpeg() -> bool:
    from shutil import which
    return which("ffmpeg") is not None and which("ffprobe") is not None

print("[BOOT] ffmpeg present:", _has_ffmpeg())

def _probe_video_size(video_path: str) -> tuple[int, int]:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
            video_path,
        ],
        text=True
    ).strip()
    if "x" not in out:
        raise RuntimeError(f"ffprobe failed to read size: {out}")
    w_s, h_s = out.split("x", 1)
    return int(w_s), int(h_s)

def _safezone_rect(video_w: int, video_h: int):
    left = int(video_w * SAFEZONE_SIDE_PCT)
    right = int(video_w * (1.0 - SAFEZONE_SIDE_PCT))
    top = int(video_h * SAFEZONE_TOP_PCT)
    bottom = int(video_h * (1.0 - SAFEZONE_BOTTOM_PCT))
    return left, top, right, bottom

def _safezone_debug_filters(video_w: int, video_h: int) -> str:
    left, top, right, bottom = _safezone_rect(video_w, video_h)
    top_h = max(0, top)
    bottom_h = max(0, video_h - bottom)
    left_w = max(0, left)
    right_w = max(0, video_w - right)
    border = max(2, int(video_h * 0.003))
    parts = [
        f"drawbox=x=0:y=0:w=iw:h={top_h}:color=red@0.25:t=fill",
        f"drawbox=x=0:y={bottom}:w=iw:h={bottom_h}:color=red@0.25:t=fill",
        f"drawbox=x=0:y={top}:w={left_w}:h={bottom - top}:color=red@0.25:t=fill",
        f"drawbox=x={right}:y={top}:w={right_w}:h={bottom - top}:color=red@0.25:t=fill",
        f"drawbox=x={left}:y={top}:w={right - left}:h={bottom - top}:color=green@0.8:t={border}",
    ]
    return ",".join(parts)

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/local/share/fonts/custom/MREARLN.TTF",
        "/usr/local/share/fonts/MREARLN.TTF",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            key = (p, font_size)
            if key not in _FONT_CACHE:
                _FONT_CACHE[key] = ImageFont.truetype(p, font_size)
            return _FONT_CACHE[key]
    return ImageFont.load_default()

def _check_srt_safezone(srt_text: str, video_w: int, video_h: int, font_size: int):
    safe_left, safe_top, safe_right, safe_bottom = _safezone_rect(video_w, video_h)
    safe_pad = int(video_h * SAFEZONE_PAD_PCT)
    font = _load_font(font_size)
    img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(img)
    for _, _, text in _parse_srt_blocks(srt_text):
        if not text:
            continue
        bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=2, stroke_width=2)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_h > (safe_bottom - safe_top):
            raise RuntimeError("Caption height exceeds safe-zone height; reduce font size.")
        x = int((video_w - text_w) / 2)
        bottom = safe_bottom - safe_pad
        y = int(bottom - text_h)
        if x < safe_left or (x + text_w) > safe_right or y < safe_top or (y + text_h) > safe_bottom:
            raise RuntimeError("Caption bbox intersects safe-zone no-go areas.")

def _hex_to_ass_color(value: str) -> str:
    v = value.strip().lstrip("#")
    if len(v) != 6:
        return "&H00FFFFFF"
    try:
        r = int(v[0:2], 16)
        g = int(v[2:4], 16)
        b = int(v[4:6], 16)
    except ValueError:
        return "&H00FFFFFF"
    return f"&H00{b:02X}{g:02X}{r:02X}"

def _split_token(token: str):
    m = re.match(r"^([^A-Za-z0-9]*)([A-Za-z0-9']+)([^A-Za-z0-9]*)$", token)
    if not m:
        return "", token, ""
    return m.group(1), m.group(2), m.group(3)

def _core_key(token: str) -> str:
    _, core, _ = _split_token(token)
    if core.lower().endswith("'s"):
        core = core[:-2]
    return core.lower()

def _replace_core(token: str, replacement: str) -> str:
    pre, core, post = _split_token(token)
    if core.lower().endswith("'s"):
        return f"{pre}{replacement}'s{post}"
    return f"{pre}{replacement}{post}"

def _parse_cap_terms() -> tuple[dict, list]:
    terms = [t.strip() for t in (CAPITALIZE_TERMS or "").split(",") if t.strip()]
    word_map = {}
    phrases = []
    for term in terms:
        parts = term.split()
        if len(parts) == 1:
            word_map[parts[0].lower()] = parts[0]
        else:
            phrases.append({"keys": [p.lower() for p in parts], "display": parts})
    return word_map, phrases

def _apply_capitalization_to_words(words):
    if not words:
        return
    word_map, phrases = _parse_cap_terms()
    i = 0
    while i < len(words):
        matched = False
        for ph in phrases:
            n = len(ph["keys"])
            if i + n > len(words):
                continue
            ok = True
            for j in range(n):
                if _core_key(words[i + j]["word"]) != ph["keys"][j]:
                    ok = False
                    break
            if ok:
                for j in range(n):
                    words[i + j]["word"] = _replace_core(words[i + j]["word"], ph["display"][j])
                i += n
                matched = True
                break
        if matched:
            continue
        key = _core_key(words[i]["word"])
        if key in word_map:
            words[i]["word"] = _replace_core(words[i]["word"], word_map[key])
        i += 1

def _download_url_to(path: str, url: str):
    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1<<20)
            if not chunk:
                break
            f.write(chunk)

def _escape_for_subtitles(path: str) -> str:
    return path.replace("\\", "\\\\").replace(":", "\\:")

def _slugify(text: str, max_len: int = 64) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    t = t.lower()
    t = re.sub(r"[’'`]", "-", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return (t[:max_len] or "caption")

def _write_single_block_srt(path: str, duration_s: float, text: str):
    ms = int(round(max(0.1, duration_s) * 1000))
    hh, rem = divmod(ms, 3600_000)
    mm, rem = divmod(rem, 60_000)
    ss, mmm = divmod(rem, 1000)
    with open(path, "w", encoding="utf-8") as f:
        f.write("1\n")
        f.write(f"00:00:00,000 --> {hh:02d}:{mm:02d}:{ss:02d},{mmm:03d}\n")
        f.write(text.strip() + "\n\n")

def _burn_captions_ffmpeg(video_path: str, srt_path: str, out_path: str, style: str | None,
                          start_s: float | None = None, end_s: float | None = None,
                          sub_is_ass: bool = False):
    fonts_dir = "/usr/local/share/fonts/custom"
    video_w, video_h = _probe_video_size(video_path)
    font_size = max(18, int(video_h * FONT_SIZE_PCT))
    margin_lr = int(video_w * SAFEZONE_SIDE_PCT)
    margin_v = int(video_h * SAFEZONE_BOTTOM_PCT) + int(video_h * SAFEZONE_PAD_PCT)
    enforced_style = (
        f"Alignment=2,MarginL={margin_lr},MarginR={margin_lr},MarginV={margin_v},Fontsize={font_size}"
    )
    base_style = (
        f"FontName={FONT_FAMILY},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        f"BorderStyle=1,Outline=3,Shadow=0,{enforced_style}"
    )
    eff_style = None
    if not sub_is_ass:
        eff_style = (f"{style.strip()},{enforced_style}" if style else base_style)
    fs_esc = eff_style.replace(",", "\\,").replace(";", "\\;") if eff_style else None
    srt_esc = _escape_for_subtitles(srt_path)
    if fs_esc:
        flt = f"subtitles={srt_esc}:fontsdir={fonts_dir}:force_style={fs_esc}"
    else:
        flt = f"subtitles={srt_esc}:fontsdir={fonts_dir}"
    if SAFEZONE_DEBUG:
        flt = f"{flt},{_safezone_debug_filters(video_w, video_h)}"

    if SAFEZONE_ENFORCE and (not sub_is_ass):
        try:
            with open(srt_path, "r", encoding="utf-8") as f:
                srt_text = f.read()
            _check_srt_safezone(srt_text, video_w, video_h, font_size)
        except Exception as e:
            raise RuntimeError(f"Safe-zone check failed: {e}")

    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-stats","-threads","1","-y"]
    # Trim window (fast seek before input)
    if start_s is not None:
        cmd += ["-ss", f"{float(start_s):.3f}"]
    if end_s is not None and start_s is not None:
        cmd += ["-to", f"{float(end_s):.3f}"]
    cmd += [
        "-i", video_path,
        "-vf", flt, "-c:v","libx264","-preset","veryfast","-crf","22",
        "-c:a","copy","-movflags","+faststart","-shortest",
        out_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", "ignore") if e.stderr else str(e)
        raise RuntimeError(f"ffmpeg failed.\nFilter:\n{flt}\n\nStderr:\n{err}")

# --------- subtitle helpers ---------
def _vtt_to_srt(vtt: str) -> str:
    lines = [ln.rstrip("\n") for ln in vtt.splitlines()]
    out = []; idx = 1; buf = []
    def flush_block():
        nonlocal idx
        if not buf:
            return
        tl = buf[0].replace(".", ",")
        out.append(str(idx)); idx += 1
        out.append(tl)
        for t in buf[1:]:
            out.append(t)
        out.append("")
        buf.clear()
    for ln in lines:
        s = ln.strip()
        if s == "WEBVTT" or s.startswith("NOTE"):
            continue
        if "-->" in s:
            flush_block()
            buf.append(s)
        elif s == "":
            flush_block()
        else:
            buf.append(ln)
    flush_block()
    return "\n".join(out) + "\n"

def _segments_to_srt(segments) -> str:
    def fmt_hms(t):
        ms_total = int(round(float(t) * 1000))
        hh = ms_total // 3_600_000; ms_total %= 3_600_000
        mm = ms_total // 60_000;    ms_total %= 60_000
        ss = ms_total // 1000;      ms = ms_total % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
    out = []; idx = 1
    for s in segments or []:
        start = s.get("start"); end = s.get("end")
        if start is None or end is None:
            ts = s.get("timestamp") or s.get("timestamps")
            if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                start, end = ts[0], ts[1]
        text = (s.get("text") or "").strip()
        if start is None or end is None or not text:
            continue
        out += [str(idx), f"{fmt_hms(start)} --> {fmt_hms(end)}", text, ""]
        idx += 1
    return ("\n".join(out) + "\n") if out else ""

def _normalize_word_spacing(text: str) -> str:
    # Remove spaces before punctuation for cleaner captions.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()

def _words_to_srt(words, max_words: int, max_cue_duration: float, gap_split_sec: float, min_words_per_cue: int) -> str:
    """
    Build SRT from word-level timestamps. Respects max_words and max_cue_duration.
    """
    def fmt_hms(t):
        ms_total = int(round(float(t) * 1000))
        hh = ms_total // 3_600_000; ms_total %= 3_600_000
        mm = ms_total // 60_000;    ms_total %= 60_000
        ss = ms_total // 1000;      ms = ms_total % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    cues = _words_to_cues(words, max_words, max_cue_duration, gap_split_sec, min_words_per_cue)

    out = []
    idx = 1
    for cue in cues:
        start = cue[0]["start"]
        end = cue[-1]["end"]
        if max_cue_duration > 0 and end - start > max_cue_duration:
            end = start + max_cue_duration
        text = _normalize_word_spacing(" ".join(w["word"] for w in cue))
        if not text:
            continue
        out += [str(idx), f"{fmt_hms(start)} --> {fmt_hms(end)}", text, ""]
        idx += 1
    return ("\n".join(out) + "\n") if out else ""

def _fastwh_extract_words(data):
    """
    Extract word-level timestamps from FastWhisper-style responses.
    """
    def pull_words_from_segments(segs):
        words = []
        for s in segs or []:
            for w in s.get("words") or []:
                if {"start", "end", "word"} <= set(w.keys()):
                    words.append({"start": w["start"], "end": w["end"], "word": w["word"]})
        return words

    out = data.get("output")
    if isinstance(out, dict) and isinstance(out.get("segments"), list):
        words = pull_words_from_segments(out["segments"])
        if words:
            return words
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict) and isinstance(first.get("segments"), list):
            words = pull_words_from_segments(first["segments"])
            if words:
                return words

    if isinstance(data.get("segments"), list):
        words = pull_words_from_segments(data["segments"])
        if words:
            return words
    nest = data.get("data")
    if isinstance(nest, dict) and isinstance(nest.get("segments"), list):
        words = pull_words_from_segments(nest["segments"])
        if words:
            return words

    return []

def _parse_srt_blocks(srt_text: str):
    """
    Return list of (start_s, end_s, text). Robust to multi-line text blocks.
    """
    def tosec(hms: str) -> float:
        hh, mm, ssms = hms.split(":")
        ss, ms = ssms.split(",")
        return int(hh)*3600 + int(mm)*60 + int(ss) + int(ms)/1000.0

    blocks = []
    cur = []
    for ln in srt_text.splitlines():
        if ln.strip() == "":
            if cur:
                try:
                    # cur[0] -> index, cur[1] -> "a --> b", rest -> text lines
                    timeline = cur[1].strip()
                    a, b = [x.strip() for x in timeline.split("-->")]
                    start_s = tosec(a); end_s = tosec(b)
                    text = "\n".join(cur[2:]).strip()
                    if text:
                        blocks.append((start_s, end_s, text))
                except Exception:
                    pass
                cur = []
        else:
            cur.append(ln.rstrip("\n"))
    # flush last
    if cur:
        try:
            timeline = cur[1].strip()
            a, b = [x.strip() for x in timeline.split("-->")]
            start_s = tosec(a); end_s = tosec(b)
            text = "\n".join(cur[2:]).strip()
            if text:
                blocks.append((start_s, end_s, text))
        except Exception:
            pass
    return blocks

def _approx_words_from_srt_blocks(srt_text: str):
    words = []
    for start_s, end_s, text in _parse_srt_blocks(srt_text):
        text = (text or "").strip()
        if not text or end_s <= start_s:
            continue
        toks = re.findall(r"[A-Za-z0-9']+", text)
        if not toks:
            continue
        per = (end_s - start_s) / len(toks)
        for i, tok in enumerate(toks):
            w_start = start_s + i * per
            w_end = start_s + (i + 1) * per
            words.append({"start": w_start, "end": w_end, "word": tok})
    return words

def _words_to_cues(words, max_words: int, max_cue_duration: float, gap_split_sec: float, min_words_per_cue: int):
    cues = []
    cur = []
    for w in words:
        txt = str(w.get("word", "")).strip()
        if not txt:
            continue
        start = w.get("start")
        end = w.get("end")
        if start is None or end is None:
            continue

        if cur:
            cur_start = cur[0]["start"]
            next_len = len(cur) + 1
            next_end = float(end)
            next_dur = next_end - cur_start
            gap = float(start) - cur[-1]["end"]
            if (gap_split_sec > 0 and gap > gap_split_sec and len(cur) >= max(1, min_words_per_cue)):
                cues.append(cur)
                cur = []
            elif (max_words > 0 and next_len > max_words) or (max_cue_duration > 0 and next_dur > max_cue_duration):
                cues.append(cur)
                cur = []

        cur.append({"start": float(start), "end": float(end), "word": txt})

    if cur:
        cues.append(cur)
    return cues

def _cues_to_srt(cues) -> str:
    def fmt_hms(t):
        ms_total = int(round(float(t) * 1000))
        hh = ms_total // 3_600_000; ms_total %= 3_600_000
        mm = ms_total // 60_000;    ms_total %= 60_000
        ss = ms_total // 1000;      ms = ms_total % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    out = []
    idx = 1
    for cue in cues:
        start = cue[0]["start"]
        end = cue[-1]["end"]
        text = _normalize_word_spacing(" ".join(w["word"] for w in cue))
        if not text:
            continue
        out += [str(idx), f"{fmt_hms(start)} --> {fmt_hms(end)}", text, ""]
        idx += 1
    return ("\n".join(out) + "\n") if out else ""

def _cues_to_ass_karaoke(cues, video_w: int, video_h: int, font_size: int, margin_lr: int, margin_v: int,
                         active_color: str, inactive_color: str) -> str:
    def fmt_ass_time(t):
        cs = int(round(float(t) * 100))
        h = cs // 360000; cs %= 360000
        m = cs // 6000; cs %= 6000
        s = cs // 100;   cs %= 100
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    pri = _hex_to_ass_color(active_color)
    sec = _hex_to_ass_color(inactive_color)

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {video_w}",
        f"PlayResY: {video_h}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{FONT_FAMILY},{font_size},{pri},{sec},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,2,{margin_lr},{margin_lr},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    lines = []
    for cue in cues:
        start = cue[0]["start"]
        end = cue[-1]["end"]
        parts = []
        for w in cue:
            dur_cs = max(1, int(round((w["end"] - w["start"]) * 100)))
            parts.append(f"{{\\k{dur_cs}}}{w['word']}")
        text = " ".join(parts)
        lines.append(f"Dialogue: 0,{fmt_ass_time(start)},{fmt_ass_time(end)},Default,,0,0,0,,{text}")

    return "\n".join(header + lines) + "\n"

# --------- OpenAI caption.py integration (optional backend) ---------
def _run_caption_py(video_local: str, output_local: str, language: str | None, max_words: int):
    """
    Calls caption.py to generate a captioned MP4 using OpenAI Whisper-1
    with word-level timestamps and the 3-word tiles logic.
    """
    script_path = os.getenv("CAPTION_PY_PATH", "/app/caption.py")
    cmd = ["python", script_path, video_local, "--output", output_local, "--model", "whisper-1"]
    if language:
        cmd += ["--language", language]
    if max_words and max_words > 0:
        cmd += ["--max-words", str(max_words)]
    print("[OPENAI-CAPTION] running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"caption.py failed: {e}")

def _maybe_captions_txt_to_srt(video_local: str) -> str | None:
    """
    caption.py emits transcripts/<stem>-captions.txt:
        <start> --> <end>
        TEXT

    Convert to SRT so we can upload alongside the MP4.
    """
    v = pathlib.Path(video_local)
    txt_path = v.parent / "transcripts" / f"{v.stem}-captions.txt"
    if not txt_path.exists():
        return None

    def fmt_hms(t):
        ms_total = int(round(float(t) * 1000))
        hh = ms_total // 3_600_000; ms_total %= 3_600_000
        mm = ms_total // 60_000;    ms_total %= 60_000
        ss = ms_total // 1000;      ms = ms_total % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    blocks = []
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    parts = [p for p in raw.split("\n\n") if p.strip()]
    for part in parts:
        lines = [ln.strip() for ln in part.splitlines() if ln.strip()]
        if len(lines) >= 2 and "-->" in lines[0]:
            tline = lines[0]
            a, b = [x.strip() for x in tline.split("-->")]
            try:
                start = float(a)
                end = float(b)
                text = " ".join(lines[1:])
                blocks.append((start, end, text))
            except Exception:
                continue

    if not blocks:
        return None

    srt_lines = []
    for i, (s, e, t) in enumerate(blocks, 1):
        srt_lines += [str(i), f"{fmt_hms(s)} --> {fmt_hms(e)}", t, ""]
    srt_text = "\n".join(srt_lines) + "\n"

    fd, srt_path = tempfile.mkstemp(prefix="captions_from_txt_", suffix=".srt")
    os.close(fd)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    return srt_path

# --------- FastWhisper (FORCE SRT; run + poll) ---------
def _fastwh_to_srt(video_url: str, *, return_data: bool = False) -> str | tuple[str, dict]:
    """
    Submit to Faster-Whisper with a hard schema:
      input.audio: string (URL)
      input.model: "large-v3"
      input.transcription: "srt"
    Poll until COMPLETED and return SRT (or convert VTT).
    """
    base = f"https://api.runpod.ai/v2/{FASTWH_ID}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

    def _req(method, path, **kw):
        kw.setdefault("timeout", (20, 600))  # connect, read
        r = requests.request(method, f"{base}{path}", headers=headers, **kw)
        r.raise_for_status()
        return r.json()

    payload = {
        "input": {
            "audio": video_url,
            "model": "large-v3",
            "transcription": "srt",
            "enable_vad": FASTWH_VAD,
            "word_timestamps": FASTWH_WORDTS,
        }
    }

    safe = json.loads(json.dumps(payload))
    if isinstance(safe["input"].get("audio"), str) and len(safe["input"]["audio"]) > 128:
        safe["input"]["audio"] = safe["input"]["audio"][:128] + "...(trunc)"
    print(f"[FASTWH] POST /run payload={json.dumps(safe, ensure_ascii=False)}")

    run = _req("POST", "/run", json=payload)
    run_id = run.get("id")
    print(f"[FASTWH] started id={run_id} status={run.get('status')} delay={run.get('delayTime')} worker={run.get('workerId')}")
    if not run_id:
        raise RuntimeError(f"/run did not return id. Body: {run}")

    import time
    last = None
    t0 = time.time()
    while True:
        st = _req("GET", f"/status/{run_id}")
        status = st.get("status")
        if status != last:
            print(f"[FASTWH] poll id={run_id} status={status} delay={st.get('delayTime')} exec={st.get('executionTime')} worker={st.get('workerId')}")
            last = status
        if status in ("COMPLETED", "COMPLETED_WITH_ERRORS"):
            data = st
            break
        if status in ("FAILED", "CANCELLED", "TIMED_OUT", "DEAD"):
            raise RuntimeError(f"Run ended {status}: {json.dumps(st)[:800]}")
        if time.time() - t0 > 3600:
            raise RuntimeError("Timeout after 3600s.")
        time.sleep(5)

    def maybe_srt(x):
        if isinstance(x, str):
            if x.lstrip().startswith("WEBVTT"):
                return _vtt_to_srt(x)
            if "-->" in x:
                return x
        return None

    out = data.get("output")

    if isinstance(out, dict):
        for k in ("srt", "text_srt", "vtt", "transcription", "text"):
            srt = maybe_srt(out.get(k))
            if srt:
                return (srt, data) if return_data else srt
        if isinstance(out.get("segments"), list):
            srt = _segments_to_srt(out["segments"])
            if srt.strip():
                print("[FASTWH] built SRT from output.segments")
                return (srt, data) if return_data else srt

    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict):
            for k in ("srt", "text_srt", "vtt", "transcription", "text"):
                srt = maybe_srt(first.get(k))
                if srt:
                    return (srt, data) if return_data else srt
            if isinstance(first.get("segments"), list):
                srt = _segments_to_srt(first["segments"])
                if srt.strip():
                    print("[FASTWH] built SRT from output[0].segments")
                    return (srt, data) if return_data else srt
        else:
            srt = maybe_srt(first)
            if srt:
                return (srt, data) if return_data else srt

    srt = maybe_srt(out)
    if srt:
        return (srt, data) if return_data else srt

    for k in ("srt", "text_srt", "vtt", "transcription", "text"):
        srt = maybe_srt(data.get(k))
        if srt:
            return (srt, data) if return_data else srt
    if isinstance(data.get("segments"), list):
        srt = _segments_to_srt(data["segments"])
        if srt.strip():
            print("[FASTWH] built SRT from top-level segments")
            return (srt, data) if return_data else srt

    nest = data.get("data")
    if isinstance(nest, dict):
        for k in ("srt", "text_srt", "vtt", "transcription", "text"):
            srt = maybe_srt(nest.get(k))
            if srt:
                return (srt, data) if return_data else srt
        if isinstance(nest.get("segments"), list):
            srt = _segments_to_srt(nest["segments"])
            if srt.strip():
                print("[FASTWH] built SRT from data.segments")
                return (srt, data) if return_data else srt

    raise RuntimeError(f"COMPLETED but no SRT/VTT/segments in response. Snip: {json.dumps(data)[:800]}")

# --------- main handler ---------
def handler(event):
    inp = (event or {}).get("input") or {}
    job_id     = inp.get("job_id")
    video_url  = inp.get("video_url")
    style      = inp.get("style")
    output_key = inp.get("output_key")   # optional: where to put the SRT
    burn       = bool(inp.get("burn", True))

    if not job_id:
        raise RuntimeError("job_id is required")
    if not video_url:
        raise RuntimeError("video_url is required")

    # Download once (for duration trim + burn-in or caption.py)
    vid_fd, vid_local = tempfile.mkstemp(prefix="video_", suffix=".mp4"); os.close(vid_fd)
    _download_url_to(vid_local, video_url)

    # --------- Branch: Use caption.py (OpenAI) pipeline ---------
    if CAPTION_BACKEND == "openai":
        print("[BACKEND] Using caption.py (OpenAI Whisper-1, 3-word tiles)")
        # Run caption.py to create captioned MP4
        out_fd, out_local = tempfile.mkstemp(prefix="captioned_", suffix=".mp4"); os.close(out_fd)
        max_words_for_tiles = MAX_WORDS_PER_CU if MAX_WORDS_PER_CU > 0 else 3
        _run_caption_py(
            video_local=vid_local,
            output_local=out_local,
            language=LANG_HINT,
            max_words=max_words_for_tiles
        )

        # Try to collect captions.txt -> SRT (optional)
        srt_local = _maybe_captions_txt_to_srt(vid_local)
        result = {}

        # Upload SRT if available
        if srt_local:
            base = pathlib.Path(output_key).stem if output_key else (pathlib.Path(urllib.parse.urlparse(video_url).path).stem or "caption")
            srt_key = output_key if output_key else _key(job_id, "captions", "transcripts", f"{base}.srt")
            up_srt = _upload_tmp_to_s3(srt_local, srt_key, content_type="application/x-subrip")
            result.update({"srt_key": up_srt["key"], "srt_url": up_srt["url"]})

        # Upload captioned MP4
        cap_key = inp.get("output_video_key") or _key(job_id, "captions", f"{_slugify(pathlib.Path(urllib.parse.urlparse(video_url).path).stem or 'captioned')}.mp4")
        up_cap = _upload_tmp_to_s3(out_local, cap_key, content_type="video/mp4")
        result.update({"captioned_key": up_cap["key"], "captioned_url": up_cap["url"]})
        return result

    # --------- Default: FastWhisper SRT + ffmpeg burn ---------
    print("[BACKEND] Using FastWhisper (RunPod) SRT + ffmpeg burn path")
    # Use provided SRT (if any) unless forced to re-transcribe; else transcribe via RunPod (force SRT)
    ass_local = None
    srt_text = None
    srt_from_words = False
    srt_url_in  = inp.get("srt_url")
    srt_text_in = inp.get("srt_text")
    srt_key_in  = inp.get("srt_key")
    force_fastwh = CAPTION_FORCE_FASTWH or WORD_HIGHLIGHT
    if force_fastwh and (srt_text_in or srt_url_in or srt_key_in):
        print("[CAPTION] Ignoring provided SRT due to force_fastwh/word_highlight.")
        srt_text_in = None
        srt_url_in = None
        srt_key_in = None
    if srt_text_in:
        srt_text = srt_text_in
    elif srt_url_in:
        with urllib.request.urlopen(srt_url_in) as r:
            _raw = r.read().decode("utf-8", "ignore")
            srt_text = _vtt_to_srt(_raw) if _raw.lstrip().startswith("WEBVTT") else _raw
    elif srt_key_in:
        obj = s3.get_object(Bucket=AWS_S3_BUCKET, Key=srt_key_in)
        _raw = obj["Body"].read().decode("utf-8", "ignore")
        srt_text = _vtt_to_srt(_raw) if _raw.lstrip().startswith("WEBVTT") else _raw
    else:
        srt_text, fw_data = _fastwh_to_srt(video_url, return_data=True)
        words = _fastwh_extract_words(fw_data)
        _apply_capitalization_to_words(words)
        if WORD_HIGHLIGHT and (not words) and WORD_HIGHLIGHT_APPROX and srt_text:
            words = _approx_words_from_srt_blocks(srt_text)
            _apply_capitalization_to_words(words)
            if words:
                print(f"[FASTWH] approximated word timings from SRT words={len(words)}")
        if words and (MAX_WORDS_PER_CU > 0 or MAX_CUE_DURATION > 0 or WORD_HIGHLIGHT):
            max_words = MAX_WORDS_PER_CU if MAX_WORDS_PER_CU > 0 else 3
            cues = _words_to_cues(words, max_words, MAX_CUE_DURATION, WORD_GAP_SPLIT_SEC, MIN_WORDS_PER_CUE)
            word_srt = _cues_to_srt(cues)
            if word_srt.strip():
                srt_text = word_srt
                srt_from_words = True
                print(f"[FASTWH] built word-based SRT words={len(words)} max_words={max_words} max_dur={MAX_CUE_DURATION}")
            if WORD_HIGHLIGHT and cues:
                try:
                    video_w, video_h = _probe_video_size(vid_local)
                    font_size = max(18, int(video_h * FONT_SIZE_PCT))
                    margin_lr = int(video_w * SAFEZONE_SIDE_PCT)
                    margin_v = int(video_h * SAFEZONE_BOTTOM_PCT) + int(video_h * SAFEZONE_PAD_PCT)
                    ass_text = _cues_to_ass_karaoke(
                        cues, video_w, video_h, font_size, margin_lr, margin_v,
                        WORD_HIGHLIGHT_COLOR, WORD_INACTIVE_COLOR
                    )
                    ass_fd, ass_local = tempfile.mkstemp(prefix="captions_", suffix=".ass"); os.close(ass_fd)
                    with open(ass_local, "w", encoding="utf-8") as f:
                        f.write(ass_text)
                    print("[FASTWH] built ASS karaoke captions")
                except Exception as e:
                    print("[FASTWH] failed to build ASS karaoke, falling back:", e)

    srt_fd, srt_local = tempfile.mkstemp(prefix="captions_", suffix=".srt"); os.close(srt_fd)
    with open(srt_local, "w", encoding="utf-8") as f:
        f.write(srt_text)

    # Trim SRT to video duration (best-effort)
    try:
        dur = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration",
             "-of","default=noprint_wrappers=1:nokey=1", vid_local],
            text=True).strip()
        DUR = float(dur)
        trimmed = []
        block=[]
        def tosec(hms):
            hh,mm,ssms = hms.split(":"); ss,ms=ssms.split(",")
            return int(hh)*3600 + int(mm)*60 + int(ss) + int(ms)/1000.0
        def fmt(t):
            if t<0: t=0.0
            h=int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int((t*1000)%1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        with open(srt_local,"r",encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    block.append(ln.rstrip("\n"))
                else:
                    if len(block)>=2 and "-->" in block[1]:
                        idx = str(len(trimmed)+1)
                        a,b = [x.strip() for x in block[1].split("-->")]
                        s = tosec(a); e = tosec(b)
                        if s < DUR:
                            if e > DUR: e = max(s, DUR-0.01)
                            blk = [idx, f"{fmt(s)} --> {fmt(e)}"] + block[2:]
                            trimmed.append("\n".join(blk))
                    block=[]
        if trimmed:
            with open(srt_local,"w",encoding="utf-8") as f:
                f.write("\n\n".join(trimmed) + "\n\n")
            srt_text = open(srt_local,"r",encoding="utf-8").read()
    except Exception:
        pass

    # Derive base name from provided output_key or video filename
    base = None
    if output_key:
        base = pathlib.Path(output_key).stem
    elif srt_key_in:
        base = pathlib.Path(srt_key_in).stem
    else:
        base = pathlib.Path(urllib.parse.urlparse(video_url).path).stem or "caption"

    # Upload SRT
    srt_key = output_key if output_key else _key(job_id, "captions", "transcripts", f"{base}.srt")
    up_srt = _upload_tmp_to_s3(srt_local, srt_key, content_type="application/x-subrip")
    result = {"srt_key": up_srt["key"], "srt_url": up_srt["url"]}

    # Optional: chunkify SRT (if AWK tool is present in the image)
    try:
        if not srt_from_words and (MAX_WORDS_PER_CU > 0 or MAX_CUE_DURATION > 0):
            tmp_fd, tuned_srt = tempfile.mkstemp(prefix="captions_tuned_", suffix=".srt"); os.close(tmp_fd)
            awk = "/app/tools/srt_chunkify.awk"
            cmd = ["awk", f"-vW={MAX_WORDS_PER_CU}", f"-vD={MAX_CUE_DURATION}", "-f", awk, srt_local]
            with open(tuned_srt, "w", encoding="utf-8") as outf:
                subprocess.run(cmd, check=True, stdout=outf)
            srt_local = tuned_srt
            srt_text = open(srt_local,"r",encoding="utf-8").read()
    except Exception as _e:
        print("[CHUNKIFY] Skipped (no awk or error):", _e)

    if not burn:
        return result

    # ------- SINGLE MP4 BURN (optional) -------
    if burn:
        if not _has_ffmpeg():
            raise RuntimeError("ffmpeg not present in image; cannot burn captions.")
        base = pathlib.Path(srt_key).stem if output_key else (pathlib.Path(urllib.parse.urlparse(video_url).path).stem or "captioned")
        out_fd, out_local = tempfile.mkstemp(prefix="captioned_", suffix=".mp4"); os.close(out_fd)
        if ass_local:
            if SAFEZONE_ENFORCE:
                video_w, video_h = _probe_video_size(vid_local)
                font_size = max(18, int(video_h * FONT_SIZE_PCT))
                if not srt_text:
                    with open(srt_local, "r", encoding="utf-8") as f:
                        srt_text = f.read()
                _check_srt_safezone(srt_text, video_w, video_h, font_size)
            _burn_captions_ffmpeg(vid_local, ass_local, out_local, style, sub_is_ass=True)
        else:
            _burn_captions_ffmpeg(vid_local, srt_local, out_local, style)
        cap_key = inp.get("output_video_key") or _key(job_id, "captions", f"{base}.mp4")
        up_cap = _upload_tmp_to_s3(out_local, cap_key, content_type="video/mp4")
        result.update({"captioned_key": up_cap["key"], "captioned_url": up_cap["url"]})
    return result

def _safe_handler(event):
    try:
        return handler(event)
    except Exception as e:
        print("[FATAL]", e)
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    print('[BOOT] starting runpod serverless...')
    runpod.serverless.start({"handler": _safe_handler})
