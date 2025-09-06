import os
import uuid
import tempfile
import subprocess
import requests
import runpod
import boto3
from botocore.config import Config

print("RunPod worker starting…", flush=True)

# --- S3 config ---
S3_ENDPOINT = "https://s3api-us-ks-2.runpod.io"
S3_BUCKET = "cqk82s22rj"
S3_REGION = "us-ks-2"
URL_TTL_SECONDS = int(os.getenv("URL_TTL_SECONDS", "86400"))  # 24h default

# Expect your access keys in env vars (set them in RunPod dashboard)
S3_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET = os.getenv("S3_SECRET_KEY")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    region_name=S3_REGION,
    # SigV4 only; (no addressing override)
    config=Config(signature_version="s3v4"),
)

def handler(event):
    inp = event.get("input", {})
    video_url = inp.get("video_url")
    out_name = inp.get("output_name", f"output-{uuid.uuid4().hex}.mp4")

    if not video_url:
        return {"error": "video_url is required"}

    with tempfile.TemporaryDirectory() as td:
        in_mp4 = os.path.join(td, "input.mp4")
        # keep also writing to the mounted volume for persistence
        out_mp4 = os.path.join("/runpod-volume", out_name)

        try:
            # Download source
            with requests.get(video_url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(in_mp4, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)

            # Run caption job
            proc = subprocess.run(
                ["python", "/app/caption.py", in_mp4, "--output", out_mp4],
                check=True,
                capture_output=True,
                text=True,
            )

            if not os.path.exists(out_mp4):
                return {"error": "caption.py completed but no output file found."}

            # Upload to RunPod S3 volume (no ACLs; gateway doesn’t support x-amz-acl)
            s3.upload_file(
                out_mp4, S3_BUCKET, out_name,
                ExtraArgs={"ContentType": "video/mp4"}
            )

            # Presigned (was previously working from the worker’s POV)
            presigned_url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET, "Key": out_name},
                ExpiresIn=URL_TTL_SECONDS,
            )

            # Also return plain path-style for reference (may be 403 until bucket policy allows it)
            plain_url = f"{S3_ENDPOINT}/{S3_BUCKET}/{out_name}"

            return {
                "status": "ok",
                "file_url": presigned_url,
                "plain_url": plain_url,
                "s3_key": out_name,
                "size_bytes": os.path.getsize(out_mp4),
                "stdout": proc.stdout[-1000:],
            }

        except subprocess.CalledProcessError as e:
            return {
                "error": "caption_script_failed",
                "return_code": e.returncode,
                "stderr": e.stderr[-2000:],
                "stdout": e.stdout[-1000:],
            }
        except Exception as e:
            return {"error": f"runtime_error: {e.__class__.__name__}: {e}"}

runpod.serverless.start({"handler": handler})
