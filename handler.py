import os, re, json, tempfile, subprocess, urllib.request, boto3, requests, shlex, pathlib
from botocore.client import Config
import runpod

# --------- ENV ---------
AWS_REGION     = os.getenv("AWS_REGION", "us-east-1")
AWS_S3_BUCKET  = os.getenv("AWS_S3_BUCKET")
S3_PREFIX_BASE = os.getenv("S3_PREFIX_BASE", "jobs")

# FastWhisper endpoint (preferred)
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
FASTWH_ID      = os.getenv("RUNPOD_FASTWHISPER_ENDPOINT_ID")
FASTWH_TRANS   = "srt"  # forced to srt to guarantee timestamps
FASTWH_VAD     = os.getenv("FASTWH_ENABLE_VAD", "true").lower() in ("1","true","yes","on")
FASTWH_WORDTS  = os.getenv("FASTWH_WORD_TIMESTAMPS", "true").lower() in ("1","true","yes","on")
LANG_HINT      = os.getenv("TRANSCRIBE_LANG") or None               # e.g. "en"
print(f"[CFG] FASTWH id={FASTWH_ID} trans={FASTWH_TRANS} vad={FASTWH_VAD} word_ts={FASTWH_WORDTS} lang={LANG_HINT}")

# OpenAI fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FALLBACK_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")

if not AWS_S3_BUCKET:
    raise RuntimeError("AWS_S3_BUCKET is required")

# --------- S3 client ---------
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

# --------- utils ---------
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
    return path.replace("\\", "\\\\").replace(":", "\\:")

def _burn_captions_ffmpeg(video_path: str, srt_path: str, out_path: str, style: str | None):
    fonts_dir = "/usr/local/share/fonts/custom"
    base_style = "FontName=MREARLN,Fontsize=36,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=3,Shadow=0,Alignment=2"
    eff_style = (style.strip() if style else base_style)
    srt_esc = _escape_for_subtitles(srt_path)
    flt = f"subtitles={srt_esc}:fontsdir={fonts_dir}:force_style={eff_style}"
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error","-stats","-threads","1","-y",
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

# --------- transcription paths ---------
def _timestamped_txt_to_srt(txt_path: str, srt_path: str):
    # supports:
    #  (A) "12.34 --> 15.67\ntext"
    #  (B) "00:00:12.340 --> 00:00:15.670\ntext"
    pat_secs = re.compile(r"^\s*(?P<s>\d+(?:\.\d+)?)\s*-->\s*(?P<e>\d+(?:\.\d+)?)\s*$")
    pat_hms  = re.compile(r"^\s*(?P<s>\d\d:\d\d:\d\d\.\d{3})\s*-->\s*(?P<e>\d\d:\d\d:\d\d\.\d{3})\s*$")
    def _fmt_hms(t):
        ms_total = int(round(float(t)*1000)) if isinstance(t,(int,float)) or t.replace('.','',1).isdigit() else None
        if ms_total is None:
            hh,mm,rest=t.split(":"); ss,mmm=rest.split(".")
            ms_total=(int(hh)*3600+int(mm)*60+int(ss))*1000+int(mmm)
        hh=ms_total//3_600_000; ms_total%=3_600_000
        mm=ms_total//60_000; ms_total%=60_000
        ss=ms_total//1000; ms=ms_total%1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
    lines = [ln.rstrip("\n") for ln in open(txt_path, "r", encoding="utf-8")]
    out=[]; i=0; idx=1
    while i < len(lines):
        m = pat_secs.match(lines[i]) or pat_hms.match(lines[i])
        if m and i+1 < len(lines) and lines[i+1].strip():
            s=m.group("s"); e=m.group("e")
            text=lines[i+1].strip()
            out.append((idx, s, e, text)); idx+=1; i+=2
            if i < len(lines) and not lines[i].strip(): i+=1
        else:
            i+=1
    if not out: raise RuntimeError("No caption lines found to convert to SRT.")
    with open(srt_path,"w",encoding="utf-8") as f:
        for i,s,e,t in out:
            f.write(f"{i}\n{_fmt_hms(s)} --> {_fmt_hms(e)}\n{t}\n\n")

def _vtt_to_srt(vtt: str) -> str:
    lines = [ln.rstrip("\n") for ln in vtt.splitlines()]
    out = []
    idx = 1
    buf = []
    def flush_block():
        nonlocal idx
        if not buf: return
        # buf[0] should be the timeline line like 00:00:01.000 --> 00:00:02.500
        # Convert dots to commas on the timeline only.
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
            # start of a new cue
            flush_block()
            buf.append(s)
        elif s == "":
            flush_block()
        else:
            buf.append(ln)
    flush_block()
    return "\n".join(out) + "\n"
# --------- FastWhisper endpoint call ---------
def _fastwh_to_srt(video_url: str) -> str:
    """
    Calls RunPod FastWhisper endpoint and returns the SRT text.
    Requires: RUNPOD_API_KEY, RUNPOD_FASTWHISPER_ENDPOINT_ID
    """
    if not (RUNPOD_API_KEY and FASTWH_ID):
        raise RuntimeError("RUNPOD_API_KEY and RUNPOD_FASTWHISPER_ENDPOINT_ID are required to use FastWhisper endpoint.")

    url = f"https://api.runpod.ai/v2/{FASTWH_ID}/runsync"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type":"application/json"}
    payload = {
        "input": {
            "audio": video_url,          # the worker can fetch from S3
            "transcription": "srt",   # expect "srt"
            "enable_vad": FASTWH_VAD,
            "word_timestamps": FASTWH_WORDTS,
            **({"language": LANG_HINT} if LANG_HINT else {})
        }
    }
    print("[FASTWH] payload:", json.dumps(payload)[:500])

    r = requests.post(url, headers=headers, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    
    print(f"[FASTWH] status={r.status_code} url={url}")
    try:
        body_snip = (r.text or "")[:1000].replace("\n"," ")
        print("[FASTWH] body:", body_snip)
        print("[FASTWH] headers:", {k:v for k,v in list(r.headers.items())[:6]})
    except Exception as _e:
        print("[FASTWH] log-exc:", _e)
    print("[FASTWH] json keys:", list((data or {}).keys()))
    outv = (data or {}).get("output")
    print("[FASTWH] output type:", type(outv), "keys:", (list(outv.keys()) if isinstance(outv, dict) else None))

    out = data.get("output") or {}
    # Accept common shapes from Faster Whisper Hub
    if isinstance(out, dict):
        tx = out.get("transcription")
        if isinstance(tx, str):
            if "-->" in tx:           # SRT-looking
                return tx
            if tx.lstrip().startswith("WEBVTT"):
                return _vtt_to_srt(tx)
        # some templates expose explicit keys too
        if isinstance(out.get("srt"), str) and "-->" in out["srt"]:
            return out["srt"]
        if isinstance(out.get("vtt"), str) and out["vtt"].lstrip().startswith("WEBVTT"):
            return _vtt_to_srt(out["vtt"])
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        # typical keys: "srt", "vtt", "text", "segments" depending on template
        if "srt" in out and isinstance(out["srt"], str):
            return out["srt"]
        if "text_srt" in out and isinstance(out["text_srt"], str):
            return out["text_srt"]
        if "text" in out and isinstance(out["text"], str) and "-->" in out["text"]:
            return out["text"]  # already srt-like
    if isinstance(out, dict) and "vtt" in out and isinstance(out["vtt"], str):
        return _vtt_to_srt(out["vtt"])

    if isinstance(out, str) and out.lstrip().startswith("WEBVTT"):
        return _vtt_to_srt(out)

    if isinstance(data, dict):
        if isinstance(data.get("vtt"), str):
            return _vtt_to_srt(data["vtt"])
        if isinstance(data.get("srt"), str):
            return data["srt"]

    import textwrap
    try:
        body_snip = textwrap.shorten(r.text.replace("\n"," "), width=600, placeholder=" ... ")
        print("[FASTWH] raw body snippet:", body_snip)
    except Exception:
        pass

    if isinstance(out, dict) and "vtt" in out and isinstance(out["vtt"], str):
        return _vtt_to_srt(out["vtt"])

    if isinstance(out, str) and out.lstrip().startswith("WEBVTT"):
        return _vtt_to_srt(out)

    if isinstance(data, dict):
        if isinstance(data.get("vtt"), str):
            return _vtt_to_srt(data["vtt"])
        if isinstance(data.get("srt"), str):
            return data["srt"]

    import textwrap
    try:
        body_snip = textwrap.shorten(r.text.replace("\n"," "), width=600, placeholder=" ... ")
        print("[FASTWH] raw body snippet:", body_snip)
    except Exception:
        pass

    # Lenient fallbacks for odd templates

    if isinstance(data, dict):

        # direct top-level srt/vtt

        if isinstance(data.get("srt"), str) and "-->" in data["srt"]:

            return data["srt"]

        if isinstance(data.get("vtt"), str) and data["vtt"].lstrip().startswith("WEBVTT"):

            return _vtt_to_srt(data["vtt"])

        # common nesting: data.get("data", {}).get("srt")

        nest = data.get("data") or {}

        if isinstance(nest, dict):

            if isinstance(nest.get("srt"), str) and "-->" in nest["srt"]:

                return nest["srt"]

            if isinstance(nest.get("vtt"), str) and nest["vtt"].lstrip().startswith("WEBVTT"):

                return _vtt_to_srt(nest["vtt"])


    raise RuntimeError(f"FastWhisper response did not contain SRT text. Output keys: {list(out.keys()) if isinstance(out,dict) else type(out)}")

# --------- OpenAI fallback (plain whisper) ---------

# --------- main handler ---------
def _openai_to_txt(video_path: str) -> str:
    """
    Fallback: use caption.py (OpenAI Whisper) to produce timestamped TXT,
    then we convert to SRT. Logged so failures aren\x27t silent.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("No RUNPOD endpoint available and OPENAI_API_KEY missing for fallback.")
    here = pathlib.Path(__file__).parent.resolve()
    cap_py = here / "caption.py"
    out = subprocess.run([
        "python3", str(cap_py), str(video_path), "--model", FALLBACK_MODEL,
        "--language", (LANG_HINT or "")], capture_output=True, text=True)
    print("[FALLBACK] rc:", out.returncode)
    if out.stdout: print("[FALLBACK][stdout]", out.stdout[:800])
    if out.stderr: print("[FALLBACK][stderr]", out.stderr[:800])
    if out.returncode != 0:
        raise RuntimeError("caption.py failed")
    transcripts_dir = pathlib.Path(video_path).parent / "transcripts"
    guess = list(transcripts_dir.glob(f"{pathlib.Path(video_path).stem}-captions.txt"))
    if not guess:
        raise RuntimeError("Fallback transcriber ran but no transcripts/*-captions.txt found.")
    return open(guess[0], "r", encoding="utf-8").read()

def handler(event):
    """
    Input:
      {
        "job_id": "20250920-xyz",
        "video_url": "https://...mp4",      # REQUIRED now
        "style": "Fontsize=24,PrimaryColour=&H00FFFFFF"  # optional
      }
    Output: SRT upload and MP4 with burned captions.
    """
    inp = event.get("input") or {}
    job_id = inp.get("job_id")
    video_url = inp.get("video_url")
    style     = inp.get("style")
    if not job_id:
        raise RuntimeError("job_id is required")
    if not video_url:
        raise RuntimeError("video_url is required")

    # 1) download video
    vid_fd, vid_local = tempfile.mkstemp(prefix="video_", suffix=".mp4"); os.close(vid_fd)
    _download_url_to(vid_local, video_url)

    # 2) get SRT text (prefer FastWhisper endpoint)
    try:
        srt_text = _fastwh_to_srt(video_url)
    except Exception as e:
        # fallback to OpenAI Whisper
        srt_text = None
        try:
            txt_text = _openai_to_txt(vid_local)      # seconds-based caption blocks
            # convert that to SRT
            txt_fd, txt_local = tempfile.mkstemp(prefix="captions_", suffix=".txt"); os.close(txt_fd)
            with open(txt_local,"w",encoding="utf-8") as f: f.write(txt_text)
            srt_fd, srt_local = tempfile.mkstemp(prefix="captions_", suffix=".srt"); os.close(srt_fd)
            _timestamped_txt_to_srt(txt_local, srt_local)
            srt_text = open(srt_local,"r",encoding="utf-8").read()
        except Exception as e2:
            raise RuntimeError(f"Transcription failed. Endpoint error: {e}. Fallback error: {e2}")

    # 3) write SRT to tmp
    srt_fd, srt_local = tempfile.mkstemp(prefix="captions_", suffix=".srt"); os.close(srt_fd)
    with open(srt_local,"w",encoding="utf-8") as f: f.write(srt_text)

    # 3a) (optional) trim SRT to video duration to avoid tail overrun
    try:
        dur = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", vid_local],
            text=True
        ).strip()
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
            with open(srt_local,"w",encoding="utf-8") as f: f.write("\n\n".join(trimmed) + "\n\n")
    except Exception:
        pass

    # 4) upload SRT
    srt_key = _key(job_id, "captions", "captions.srt")
    up_srt = _upload_tmp_to_s3(srt_local, srt_key, content_type="application/x-subrip")

    result = {"srt_key": up_srt["key"], "srt_url": up_srt["url"]}

    # 5) burn-in captions
    if not _has_ffmpeg():
        raise RuntimeError("ffmpeg not present in image; cannot burn captions.")
    out_fd, out_local = tempfile.mkstemp(prefix="captioned_", suffix=".mp4"); os.close(out_fd)
    _burn_captions_ffmpeg(vid_local, srt_local, out_local, style)
    cap_key = _key(job_id, "captions", "captioned.mp4")
    up_cap = _upload_tmp_to_s3(out_local, cap_key, content_type="video/mp4")
    result.update({"captioned_key": up_cap["key"], "captioned_url": up_cap["url"]})

    return result

runpod.serverless.start({"handler": handler})
