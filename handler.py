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

# ----- timestamped.txt -> SRT -----
def _timestamped_txt_to_srt(txt_path: str, srt_path: str):
    # Lines like: 00:00:01.000 --> 00:00:03.500 | Some text
    pat = re.compile(r"^\s*(\d\d:\d\d:\d\d\.\d{3})\s*-->\s*(\d\d:\d\d:\d\d\.\d{3})\s*\|\s*(.+?)\s*$")
    i = 1
    wrote_any = False
    with open(txt_path, "r", encoding="utf-8") as fin, open(srt_path, "w", encoding="utf-8") as fout:
        for line in fin:
            m = pat.match(line)
            if not m:
                continue
            start, end, text = m.group(1), m.group(2), m.group(3)
            start = start.replace(".", ",")
            end   = end.replace(".", ",")
            fout.write(f"{i}\n{start} --> {end}\n{text}\n\n")
            i += 1
            wrote_any = True
    if not wrote_any:
        raise RuntimeError("No valid caption lines found in transcripts/timestamped.txt")

def _has_ffmpeg() -> bool:
    from shutil import which
    return which("ffmpeg") is not None and which("ffprobe") is not None

def _download_url_to(path: str, url: str):
    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1<<20)
            if not chunk:
                break
            f.write(chunk)

def _burn_captions_ffmpeg(video_path: str, srt_path: str, out_path: str, style: str | None):
    # Use ffmpeg subtitles filter; style is optional (libass). Keep simple for portability.
    flt = f"subtitles='{srt_path}'"
    if style:
        style = style.replace("'", "").replace('"', "")
        flt = f"subtitles='{srt_path}':force_style='{style}'"
    cmd = [
        "ffmpeg","-hide_banner","-y",
        "-i", video_path,
        "-vf", flt,
        "-c:a","copy",
        out_path
    ]
    subprocess.check_call(cmd)

def handler(event):
    """
    Input:
      {
        "job_id": "20250920-xyz",
        "video_url": "https://...mp4"   (optional: produce captioned.mp4),
        "style": "Fontsize=24,PrimaryColour=&H00FFFFFF" (optional)
      }
    Output:
      {
        "srt_key": "...",
        "srt_url": "...",
        "captioned_key": "...",      (if video_url provided)
        "captioned_url": "..."
      }
    """
    inp = event.get("input") or {}
    job_id = inp.get("job_id")
    if not job_id:
        raise RuntimeError("job_id is required")

    # 1) Find transcripts/timestamped.txt on AWS S3
    ts_key = _key(job_id, "transcripts", "timestamped.txt")
    # allow a fallback name if upstream differs
    candidates = [ts_key, _key(job_id, "transcripts", "timestamps.txt")]
    found = None
    for k in candidates:
        try:
            s3.head_object(Bucket=AWS_S3_BUCKET, Key=k)
            found = k; break
        except Exception:
            continue
    if not found:
        raise RuntimeError(f"timestamped.txt not found at s3://{AWS_S3_BUCKET}/{ts_key}")

    ts_local = _download_s3_key_to_tmp(found)

    # 2) Make SRT
    srt_fd, srt_local = tempfile.mkstemp(prefix="captions_", suffix=".srt"); os.close(srt_fd)
    _timestamped_txt_to_srt(ts_local, srt_local)

    # 3) Upload SRT
    srt_key = _key(job_id, "captions", "captions.srt")
    up_srt = _upload_tmp_to_s3(srt_local, srt_key, content_type="application/x-subrip")
    result = {"srt_key": up_srt["key"], "srt_url": up_srt["url"]}

    # 4) Optional: burn-in
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
