# handler.py
import os
import subprocess
import runpod

def handler(event):
    inp = event.get("input", {})
    video_url = inp.get("video_url")
    out_name = inp.get("output_name", "output-captioned.mp4")

    if not video_url:
        return {"error": "video_url is required"}

    in_mp4 = "/tmp/input.mp4"
    out_mp4 = f"/tmp/{out_name}"

    # download video
    subprocess.run(["wget", "-O", in_mp4, video_url], check=True)

    # run your existing caption script
    subprocess.run(["python", "/app/caption.py", in_mp4, "--output", out_mp4], check=True)

    # for now, just return metadata (later you can upload to S3/Drive and return a URL)
    size = os.path.getsize(out_mp4)
    return {"status": "ok", "output_path": out_mp4, "size_bytes": size}

# >>> THIS LINE STARTS THE WORKER <<<
runpod.serverless.start({"handler": handler})
