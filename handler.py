import runpod, subprocess

def handler(event):
    inp = event.get("input", {})
    video_url = inp.get("video_url")
    if not video_url:
        return {"error": "video_url is required"}

    out_name = inp.get("output_name", "output-captioned.mp4")

    # Download input video
    subprocess.run(["wget", "-O", "input.mp4", video_url], check=True)

    # Run your existing caption script
    subprocess.run(["python", "caption.py", "input.mp4", "--output", out_name], check=True)

    # TODO: upload out_name to cloud storage and return a URL
    return {"status": "done", "output": out_name}

runpod.serverless.start({"handler": handler})
