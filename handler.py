import traceback
import os, re, json, tempfile, subprocess, urllib.request, boto3, requests, pathlib
from botocore.client import Config
import runpod

print("[BOOT] importing handler.py")

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
print(f"[CFG] S3_BUCKET={AWS_S3_BUCKET!r} region={AWS_REGION!r} OPENAI_KEY={'set' if OPENAI_API_KEY else 'missing'}")

# Caption styling & chunking controls
FONT_FAMILY      = os.getenv("FONT_FAMILY", "MisterEarl BT")
# IMPORTANT: these must be plain numbers in env (e.g., "5"), not "5 # comment"
MAX_WORDS_PER_CU = int(os.getenv("MAX_WORDS_PER_CUE", "0"))
MAX_CUE_DURATION = float(os.getenv("MAX_CUE_DURATION", "0"))

if not AWS_S3_BUCKET:
    raise RuntimeError("AWS_S3_BUCKET is required")

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

def _download_url_to(path: str, url: str):
    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1<<20)
            if not chunk:
                break
            f.write(chunk)

def _escape_for_subtitles(path: str) -> str:
    return path.replace("\\", "\\\\").replace(":", "\\:")

def _burn_captions_ffmpeg(video_path: str, srt_path: str, out_path: str, style: str | None):
    fonts_dir = "/usr/local/share/fonts/custom"
    base_style = f"FontName={FONT_FAMILY},Fontsize=36,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=3,Shadow=0,Alignment=2"
    eff_style = (style.strip() if style else base_style)
    fs_esc = eff_style.replace(",", "\\,").replace(";", "\\;")
    srt_esc = _escape_for_subtitles(srt_path)
    flt = f"subtitles={srt_esc}:fontsdir={fonts_dir}:force_style={fs_esc}"
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

# --------- transcription helpers ---------
def _timestamped_txt_to_srt(txt_path: str, srt_path: str):
    pat_secs = re.compile(r"^\s*(?P<s>\d+(?:\.\d+)?)\s*-->\s*(?P<e>\d+(?:\.\d+)?)\s*$")
    pat_hms  = re.compile(r"^\s*(?P<s>\d\d:\d\d:\d\d\.\d{3})\s*-->\s*(?P<e>\d\d:\d\d:\d\d\.\d{3})\s*$")
    def _fmt_hms(t):
        ms_total = int(round(float(t)*1000)) if isinstance(t,(int,float)) or (isinstance(t,str) and t.replace('.','',1).isdigit()) else None
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
            if i < len(lines) and not lines[i].strip():
                i+=1
        else:
            i+=1
    if not out:
        raise RuntimeError("No caption lines found to convert to SRT.")
    with open(srt_path,"w",encoding="utf-8") as f:
        for i,s,e,t in out:
            f.write(f"{i}\n{_fmt_hms(s)} --> {_fmt_hms(e)}\n{t}\n\n")

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

# --------- FastWhisper endpoint call ---------
def _fastwh_to_srt(video_url: str) -> str:
    if not (RUNPOD_API_KEY and FASTWH_ID):
        raise RuntimeError("RUNPOD_API_KEY and RUNPOD_FASTWHISPER_ENDPOINT_ID are required to use FastWhisper endpoint.")
    url = f"https://api.runpod.ai/v2/{FASTWH_ID}/runsync"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type":"application/json"}
    payload = {"input": {"audio": video_url, "transcription": "srt", "enable_vad": FASTWH_VAD, "word_timestamps": FASTWH_WORDTS}}
    if LANG_HINT:
        payload["input"]["language"] = LANG_HINT
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

    out = data.get("output")
    # accept various shapes
    if isinstance(out, dict):
        if isinstance(out.get("srt"), str) and "-->" in out["srt"]:
            return out["srt"]
        tx = out.get("transcription")
        if isinstance(tx, str):
            if "-->" in tx:
                return tx
            if tx.lstrip().startswith("WEBVTT"):
                return _vtt_to_srt(tx)
        if isinstance(out.get("vtt"), str) and out["vtt"].lstrip().startswith("WEBVTT"):
            return _vtt_to_srt(out["vtt"])
    if isinstance(out, str):
        if out.lstrip().startswith("WEBVTT"):
            return _vtt_to_srt(out)
        return out

    # top-level fallback keys
    if isinstance(data.get("srt"), str) and "-->" in data["srt"]:
        return data["srt"]
    if isinstance(data.get("vtt"), str) and data["vtt"].lstrip().startswith("WEBVTT"):
        return _vtt_to_srt(data["vtt"])
    # common nesting: {"data":{"srt": "..."}}
    nest = data.get("data")
    if isinstance(nest, dict):
        if isinstance(nest.get("srt"), str) and "-->" in nest["srt"]:
            return nest["srt"]
        if isinstance(nest.get("vtt"), str) and nest["vtt"].lstrip().startswith("WEBVTT"):
            return _vtt_to_srt(nest["vtt"])

    raise RuntimeError("FastWhisper response did not contain SRT text.")

# --------- OpenAI fallback (plain whisper) ---------
def _openai_to_txt(video_path: str) -> str:
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

# --------- main handler ---------
def handler(event):
    inp = (event or {}).get("input") or {}
    job_id = inp.get("job_id")
    video_url = inp.get("video_url")
    style     = inp.get("style")
    if not job_id:
        raise RuntimeError("job_id is required")
    if not video_url:
        raise RuntimeError("video_url is required")

    # 1) download video (only needed for fallback + trimming)
    vid_fd, vid_local = tempfile.mkstemp(prefix="video_", suffix=".mp4"); os.close(vid_fd)
    _download_url_to(vid_local, video_url)

    # 2) get SRT text
    try:
        srt_text = _fastwh_to_srt(video_url)
    except Exception as e:
        srt_text = None
        try:
            txt_text = _openai_to_txt(vid_local)
            txt_fd, txt_local = tempfile.mkstemp(prefix="captions_", suffix=".txt"); os.close(txt_fd)
            with open(txt_local,"w",encoding="utf-8") as f:
                f.write(txt_text)
            srt_fd, srt_local_tmp = tempfile.mkstemp(prefix="captions_", suffix=".srt"); os.close(srt_fd)
            _timestamped_txt_to_srt(txt_local, srt_local_tmp)
            srt_text = open(srt_local_tmp,"r",encoding="utf-8").read()
        except Exception as e2:
            raise RuntimeError(f"Transcription failed. Endpoint error: {e}. Fallback error: {e2}")

    # 3) write SRT to tmp
    srt_fd, srt_local = tempfile.mkstemp(prefix="captions_", suffix=".srt"); os.close(srt_fd)
    with open(srt_local,"w",encoding="utf-8") as f:
        f.write(srt_text)

    # 3a) trim SRT to video duration (best effort)
    try:
        dur = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", vid_local],
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
    except Exception:
        pass

    # 4) upload SRT
    srt_key = _key(job_id, "captions", "captions.srt")
    up_srt = _upload_tmp_to_s3(srt_local, srt_key, content_type="application/x-subrip")
    result = {"srt_key": up_srt["key"], "srt_url": up_srt["url"]}

    # 4a) optional: chunkify SRT
    try:
        if MAX_WORDS_PER_CU > 0 or MAX_CUE_DURATION > 0:
            tmp_fd, tuned_srt = tempfile.mkstemp(prefix="captions_tuned_", suffix=".srt"); os.close(tmp_fd)
            awk = "/app/tools/srt_chunkify.awk"
            cmd = ["awk", f"-vW={MAX_WORDS_PER_CU}", f"-vD={MAX_CUE_DURATION}", "-f", awk, srt_local]
            with open(tuned_srt, "w", encoding="utf-8") as outf:
                subprocess.run(cmd, check=True, stdout=outf)
            srt_local = tuned_srt
    except Exception as _e:
        print("[CHUNKIFY] Skipped (no awk or error):", _e)

    # 5) burn-in captions
    if not _has_ffmpeg():
        raise RuntimeError("ffmpeg not present in image; cannot burn captions.")
    out_fd, out_local = tempfile.mkstemp(prefix="captioned_", suffix=".mp4"); os.close(out_fd)
    _burn_captions_ffmpeg(vid_local, srt_local, out_local, style)
    cap_key = _key(job_id, "captions", "captioned.mp4")
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
