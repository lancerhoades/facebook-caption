import traceback
import os, re, json, tempfile, subprocess, urllib.request, boto3, requests, pathlib
from botocore.client import Config
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

FONT_FAMILY      = os.getenv("FONT_FAMILY", "MisterEarl BT")
MAX_WORDS_PER_CU = int(os.getenv("MAX_WORDS_PER_CUE", "0"))
MAX_CUE_DURATION = float(os.getenv("MAX_CUE_DURATION", "0"))

print(f"[CFG] S3_BUCKET={AWS_S3_BUCKET!r} region={AWS_REGION!r}")
print(f"[CFG] FASTWH id={FASTWH_ID} vad={FASTWH_VAD} word_ts={FASTWH_WORDTS} lang={LANG_HINT}")

if not AWS_S3_BUCKET:
    raise RuntimeError("AWS_S3_BUCKET is required")
if not (RUNPOD_API_KEY and FASTWH_ID):
    raise RuntimeError("RUNPOD_API_KEY and RUNPOD_FASTWHISPER_ENDPOINT_ID are required")

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
    base_style = f"FontName={FONT_FAMILY},Fontsize=30,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0,Alignment=2"
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

# --------- FastWhisper (FORCE SRT) ---------
# --------- FastWhisper (FORCE SRT; run + poll) ---------
def _fastwh_to_srt(video_url: str) -> str:
    """
    Submit to Faster-Whisper with a hard schema:
      input.audio: string (URL)
      input.model: "large-v3"
      input.transcription: "srt"
    Poll until COMPLETED and return SRT.
    """
    base = f"https://api.runpod.ai/v2/{FASTWH_ID}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

    def _req(method, path, **kw):
        # conservative timeouts so status polling never hangs this pod
        kw.setdefault("timeout", (20, 600))
        r = requests.request(method, f"{base}{path}", headers=headers, **kw)
        r.raise_for_status()
        return r.json()

    # ----- STRICT SCHEMA (hardcoded) -----
    payload = {
        "input": {
            "audio": video_url,
            "model": "large-v3",
            "transcription": "srt"
        }
    }

    # Log (truncate long URL)
    safe = json.loads(json.dumps(payload))  # deep copy
    if isinstance(safe["input"].get("audio"), str) and len(safe["input"]["audio"]) > 128:
        safe["input"]["audio"] = safe["input"]["audio"][:128] + "...(trunc)"
    print(f"[FASTWH] POST /run payload={json.dumps(safe, ensure_ascii=False)}")

    # Start run
    run = _req("POST", "/run", json=payload)
    run_id = run.get("id")
    print(f"[FASTWH] started id={run_id} status={run.get('status')} delay={run.get('delayTime')} worker={run.get('workerId')}")
    if not run_id:
        raise RuntimeError(f"/run did not return id. Body: {run}")

    # Poll
    import time
    t0 = time.time()
    last = None
    POLL_S = 5
    MAX_WAIT_S = 3600  # 1 hour cap for truly long files

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
        if time.time() - t0 > MAX_WAIT_S:
            raise RuntimeError(f"Timeout after {MAX_WAIT_S}s. Last: {json.dumps(st)[:800]}")
        time.sleep(POLL_S)

    # ----- Extract SRT from the final response -----
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
            if srt: return srt
        if isinstance(out.get("segments"), list):
            srt = _segments_to_srt(out["segments"])
            if srt.strip():
                print("[FASTWH] built SRT from output.segments")
                return srt

    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict):
            for k in ("srt", "text_srt", "vtt", "transcription", "text"):
                srt = maybe_srt(first.get(k))
                if srt: return srt
            if isinstance(first.get("segments"), list):
                srt = _segments_to_srt(first["segments"])
                if srt.strip():
                    print("[FASTWH] built SRT from output[0].segments")
                    return srt
        else:
            srt = maybe_srt(first)
            if srt: return srt

    srt = maybe_srt(out)
    if srt: return srt

    for k in ("srt", "text_srt", "vtt", "transcription", "text"):
        srt = maybe_srt(data.get(k))
        if srt: return srt
    if isinstance(data.get("segments"), list):
        srt = _segments_to_srt(data["segments"])
        if srt.strip():
            print("[FASTWH] built SRT from top-level segments")
            return srt

    nest = data.get("data")
    if isinstance(nest, dict):
        for k in ("srt", "text_srt", "vtt", "transcription", "text"):
            srt = maybe_srt(nest.get(k))
            if srt: return srt
        if isinstance(nest.get("segments"), list):
            srt = _segments_to_srt(nest["segments"])
            if srt.strip():
                print("[FASTWH] built SRT from data.segments")
                return srt

    raise RuntimeError(f"COMPLETED but no SRT/VTT/segments in response. Snip: {json.dumps(data)[:800]}")

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

    # Download once (for duration trim + burn-in)
    vid_fd, vid_local = tempfile.mkstemp(prefix="video_", suffix=".mp4"); os.close(vid_fd)
    _download_url_to(vid_local, video_url)

    # Transcribe via RunPod (force SRT)
    srt_text = _fastwh_to_srt(video_url)

    # Write SRT to tmp
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
    except Exception:
        pass

    # Upload SRT
    srt_key = _key(job_id, "captions", "captions.srt")
    up_srt = _upload_tmp_to_s3(srt_local, srt_key, content_type="application/x-subrip")
    result = {"srt_key": up_srt["key"], "srt_url": up_srt["url"]}

    # Optional: chunkify SRT (if AWK tool is present in the image)
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

    # Burn-in captions
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
