import os
import uuid
import tempfile
import subprocess
import requests
import runpod

# at top of handler.py
print("RunPod worker starting…", flush=True)


def handler(event):
    inp = event.get("input", {})
    video_url = inp.get("video_url")
    out_name = inp.get("output_name", f"output-{uuid.uuid4().hex}.mp4")

    if not video_url:
        return {"error": "video_url is required"}

    # temp workspace
    with tempfile.TemporaryDirectory() as td:
        in_mp4 = os.path.join(td, "input.mp4")
        out_mp4 = os.path.join(td, out_name)

        try:
            # download via Python (no wget dependency)
            with requests.get(video_url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(in_mp4, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)

            # run your existing script (reads OPENAI_API_KEY from env)
            proc = subprocess.run(
                ["python", "/app/caption.py", in_mp4, "--output", out_mp4],
                check=True,
                capture_output=True,
                text=True
            )

            if not os.path.exists(out_mp4):
                return {"error": "caption.py completed but no output file found."}

            size = os.path.getsize(out_mp4)
            return {
                "status": "ok",
                "output_path": out_mp4,  # NOTE: path is inside container; upload this if you need a URL
                "size_bytes": size,
                "stdout": proc.stdout[-1000:]  # tail for debugging
            }

        except subprocess.CalledProcessError as e:
            return {
                "error": "caption_script_failed",
                "return_code": e.returncode,
                "stderr": e.stderr[-2000:],
                "stdout": e.stdout[-1000:]
            }
        except Exception as e:
            return {"error": f"runtime_error: {e.__class__.__name__}: {e}"}

# start the Runpod worker
runpod.serverless.start({"handler": handler})
