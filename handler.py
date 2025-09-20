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
    # Use libass subtitles with custom fonts dir and default MREARLN style
    fonts_dir = "/usr/local/share/fonts/custom"
    base_style = "FontName=MREARLN,Fontsize=36,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=3,Shadow=0,Alignment=2"
    eff_style = (style.strip() if style else base_style)
    flt = f"subtitles={srt_path}:fontsdir={fonts_dir}:force_style={eff_style}"
    cmd = [
        "ffmpeg","-hide_banner","-y",
        "-i", video_path,
        "-vf", flt,
        "-c:a","copy",
        out_path
    ]
    subprocess.check_call(cmd)




def handler(event):







runpod.serverless.start({"handler": handler})
