import os, re, json, tempfile, subprocess, urllib.request, boto3
from botocore.client import Config
import runpod

# --------- ENV ---------
AWS_REGION     = os.getenv("AWS_REGION", "us-east-1")
AWS_S3_BUCKET  = os.getenv("AWS_S3_BUCKET")
S3_PREFIX_BASE = os.getenv("S3_PREFIX_BASE", "jobs")

if not AWS_S3_BUCKET:
    raise RuntimeError("AWS_S3_BUCKET is required for S3-only caption worker")

# Plain AWS S3 client (NO RunPod endpoint)
s3 = boto3.client("s3", region_name=AWS_REGION, config=Config(s3={"addressing_style":"virtual"}))

def _key(job_id: str, *parts: str) -> str:
    safe = [p.strip("/").replace("\\","/") for p in parts if p]
    return "/".join([S3_PREFIX_BASE.strip("/"), job_id] + safe)

def _presign(key: str, expires=7*24*3600) -> str:
    return s3.generate_presigned_url("get_object", Params={"Bucket": AWS_S3_BUCKET, "Key": key}, ExpiresIn=expires)

def _download_s3_key_to_tmp(key: str) -> str:
    fd, path = tempfile.mkstemp(prefix="cap_", suffix=os.path.basename(key).replace("/", "_"))
    os.close(fd)
    s3.download_file(AWS_S3_BUCKET, key, path)
    return path

def _upload_tmp_to_s3(path: str, key: str, content_type: str | None = None) -> dict:
    extra = {"ContentType": content_type} if content_type else {}
    s3.upload_file(path, AWS_S3_BUCKET, key, ExtraArgs=extra)
    return {"key": key, "url": _presign(key)}

# ----- timestamped/captions.txt -> SRT (robust) -----
def _timestamped_txt_to_srt(txt_path: str, srt_path: str):
    """
    Accepts either:
      1) One-line entries with pipe:
         HH:MM:SS.mmm --> HH:MM:SS.mmm | text
      2) Two-line blocks with seconds:
         12.34 --> 15.67
         text
      3) Two-line blocks with HH:MM:SS.mmm:
         00:00:12.340 --> 00:00:15.670
         text
    """
    def _fmt_srt_time_from_secs(secs: float) -> str:
        ms_total = int(round(secs * 1000.0))
        hh = ms_total // 3_600_000; ms_total %= 3_600_000
        mm = ms_total // 60_000;    ms_total %= 60_000
        ss = ms_total // 1000
        ms = ms_total % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    def _parse_hms_to_secs(hms: str) -> float:
        hh, mm, rest = hms.split(":")
        ss, mmm = rest.split(".")
        return (int(hh)*3600) + (int(mm)*60) + int(ss) + (int(mmm)/1000.0)

    pat_line_one = re.compile(
        r"^\s*(?P<s_hms>\d\d:\d\d:\d\d\.\d{3})\s*-->\s*(?P<e_hms>\d\d:\d\d:\d\d\.\d{3})\s*\|\s*(?P<text>.+?)\s*$"
    )
    pat_secs = re.compile(r"^\s*(?P<s_sec>\d+(?:\.\d+)?)\s*-->\s*(?P<e_sec>\d+(?:\.\d+)?)\s*$")
    pat_hms_only = re.compile(r"^\s*(?P<s_hms>\d\d:\d\d:\d\d\.\d{3})\s*-->\s*(?P<e_hms>\d\d:\d\d:\d\d\.\d{3})\s*$")

    with open(txt_path, "r", encoding="utf-8") as fin:
        lines = [ln.rstrip("\n") for ln in fin]

    out = []
    i = 0
    idx = 1
    while i < len(lines):
        ln = lines[i].strip()

        m1 = pat_line_one.match(ln)
        if m1:
            s = _parse_hms_to_secs(m1.group("s_hms"))
            e = _parse_hms_to_secs(m1.group("e_hms"))
            text = m1.group("text").strip()
            out.append((idx, s, e, text)); idx += 1; i += 1
            continue

        m2 = pat_secs.match(ln)
        if m2 and i+1 < len(lines):
            text = lines[i+1].strip()
            if text:
                s = float(m2.group("s_sec")); e = float(m2.group("e_sec"))
                out.append((idx, s, e, text)); idx += 1; i += 2
                if i < len(lines) and not lines[i].strip(): i += 1
                continue

        m3 = pat_hms_only.match(ln)
        if m3 and i+1 < len(lines):
            text = lines[i+1].strip()
            if text:
                s = _parse_hms_to_secs(m3.group("s_hms"))
                e = _parse_hms_to_secs(m3.group("e_hms"))
                out.append((idx, s, e, text)); idx += 1; i += 2
                if i < len(lines) and not lines[i].strip(): i += 1
                continue

        i += 1

    if not out:
        raise RuntimeError("No valid caption lines found in transcript file; check its format.")

    with open(srt_path, "w", encoding="utf-8") as fout:
        for idx, s, e, text in out:
            fout.write(f"{idx}\n{_fmt_srt_time_from_secs(s)} --> {_fmt_srt_time_from_secs(e)}\n{text}\n\n")

def _has_ffmpeg() -> bool:
    from shutil import which
    return which("ffmpeg") is not None and which("ffprobe") is not None

def _download_url_to(path: str, url: str):
    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1<<20)
            if not chunk: break
            f.write(chunk)

def _escape_for_subtitles(path: str) -> str:
    # https://ffmpeg.org/ffmpeg-filters.html#subtitles-1
    return path.replace("\\", "\\\\").replace(":", "\\:")

def _burn_captions_ffmpeg(video_path: str, srt_path: str, out_path: str, style: str | None):
    fonts_dir = "/usr/local/share/fonts/custom"
    base_style = "FontName=MREARLN,Fontsize=36,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=3,Shadow=0,Alignment=2"
    eff_style = (style.strip() if style else base_style)
    srt_esc = _escape_for_subtitles(srt_path)
    flt = f"subtitles={srt_esc}:fontsdir={fonts_dir}:force_style={eff_style}"
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","verbose","-y",
        "-i", video_path,
        "-vf", flt,
        "-c:a","copy",
        out_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", "ignore") if e.stderr else str(e)
        raise RuntimeError(f"ffmpeg failed.\nFilter:\n{flt}\n\nStderr:\n{err}")

def handler(event):
    """
    Input:
      {
        "job_id": "20250920-xyz",
        "video_url": "https://...mp4",   # optional
        "style": "Fontsize=24,PrimaryColour=&H00FFFFFF"  # optional
      }
    Output: SRT upload (always) and optional MP4 with burned captions.
    """
    inp = event.get("input") or {}
    job_id = inp.get("job_id")
    if not job_id:
        raise RuntimeError("job_id is required")

    # 1) Find transcript on S3 (accept several filenames)
    ts_key = _key(job_id, "transcripts", "timestamped.txt")
    candidates = [
        ts_key,
        _key(job_id, "transcripts", "timestamps.txt"),
        _key(job_id, "transcripts", "captions.txt"),   # <--- NEW fallback
    ]
    found = None
    for k in candidates:
        try:
            s3.head_object(Bucket=AWS_S3_BUCKET, Key=k)
            found = k; break
        except Exception:
            continue
    if not found:
        raise RuntimeError(f"Transcript file not found; tried: {', '.join(candidates)}")

    ts_local = _download_s3_key_to_tmp(found)

    # 2) Make SRT
    srt_fd, srt_local = tempfile.mkstemp(prefix="captions_", suffix=".srt"); os.close(srt_fd)
    _timestamped_txt_to_srt(ts_local, srt_local)

    # 3) Upload SRT
    srt_key = _key(job_id, "captions", "captions.srt")
    up_srt = _upload_tmp_to_s3(srt_local, srt_key, content_type="application/x-subrip")
    result = {"srt_key": up_srt["key"], "srt_url": up_srt["url"]}

    # 4) Optional: burn-in video captions
    video_url = inp.get("video_url")
    style     = inp.get("style")
    if video_url:
        if not _has_ffmpeg():
            raise RuntimeError("ffmpeg not present in image; omit video_url to only build SRT.")
        vid_fd, vid_local = tempfile.mkstemp(prefix="video_", suffix=".mp4"); os.close(vid_fd)
        _download_url_to(vid_local, video_url)
        out_fd, out_local = tempfile.mkstemp(prefix="captioned_", suffix=".mp4"); os.close(out_fd)
        _burn_captions_ffmpeg(vid_local, srt_local, out_local, style)
        cap_key = _key(job_id, "captions", "captioned.mp4")
        up_cap = _upload_tmp_to_s3(out_local, cap_key, content_type="video/mp4")
        result.update({"captioned_key": up_cap["key"], "captioned_url": up_cap["url"]})

    return result

runpod.serverless.start({"handler": handler})
