import runpod
import subprocess

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

    # Upload result to transfer.sh for a download link
    url = subprocess.check_output(
        ["curl", "--silent", "--upload-file", out_name, f"https://transfer.sh/{out_name}"],
        text=True
    ).strip()

    return {"status": "done", "download_url": url, "output_file": out_name}

# Important: RunPod entrypoint
runpod.serverless.start({"handler": handler})
