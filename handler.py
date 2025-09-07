import os
import uuid
import tempfile
import subprocess
import requests
import runpod
import boto3
from botocore.config import Config

print("RunPod worker starting…", flush=True)

# -----------------------------------------------------------------------------
# Config: choose backend via STORAGE_BACKEND = "aws" or "runpod"
# -----------------------------------------------------------------------------
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "aws").strip().lower()
URL_TTL_SECONDS = int(os.getenv("URL_TTL_SECONDS", "86400"))  # presigned URL TTL
PUBLIC_PREFIX   = os.getenv("PUBLIC_PREFIX", "facebook-caption").strip("/")

# ---- AWS S3 (your own AWS account) ----
AWS_BUCKET = os.getenv("AWS_S3_BUCKET")             # e.g. "toltranscriberfinal"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# Optional explicit endpoint (usually leave unset for AWS):
AWS_ENDPOINT = os.getenv("AWS_S3_ENDPOINT")         # e.g. "https://s3.us-east-1.amazonaws.com"

# ---- RunPod S3 (RunPod network volume via S3 API) ----
RP_ENDPOINT = os.getenv("RP_S3_ENDPOINT", "https://s3api-us-ks-2.runpod.io")
RP_BUCKET   = os.getenv("RP_S3_BUCKET",   "cqk82s22rj")
RP_REGION   = os.getenv("RP_S3_REGION",   "us-ks-2")

# ---- Credentials (both backends read the same env names) ----
ACCESS_KEY = (
    os.getenv("AWS_ACCESS_KEY_ID") or
    os.getenv("S3_ACCESS_KEY")     or
    os.getenv("RUN_S3_ACCESS_KEY")
)
SECRET_KEY = (
    os.getenv("AWS_SECRET_ACCESS_KEY") or
    os.getenv("S3_SECRET_KEY")         or
    os.getenv("RUN_S3_SECRET_KEY")
)

if not ACCESS_KEY or not SECRET_KEY:
    raise RuntimeError("Missing S3 credentials. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                       "or S3_ACCESS_KEY/S3_SECRET_KEY in your environment.")

def build_client_and_urls():
    """
    Returns: (s3_client, bucket, region, public_base_url)
      - public_base_url is the base to build a plain URL:
        * AWS:   https://<bucket>.s3.<region>.amazonaws.com
        * RunPod: https://s3api-<region>.runpod.io/<bucket>
    """
    if STORAGE_BACKEND == "aws":
        # Use AWS' own endpoint resolution unless AWS_ENDPOINT is explicitly set
        client = boto3.client(
            "s3",
            endpoint_url=(AWS_ENDPOINT or None),
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            region_name=AWS_REGION,
            config=Config(signature_version="s3v4"),
        )
        if AWS_ENDPOINT:
            public_base = f"{AWS_ENDPOINT.rstrip('/')}/{AWS_BUCKET}"
        else:
            public_base = f"https://{AWS_BUCKET}.s3.{AWS_REGION}.amazonaws.com"
        return client, AWS_BUCKET, AWS_REGION, public_base

    # Default to RunPod S3
    client = boto3.client(
        "s3",
        endpoint_url=RP_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=RP_REGION,
        config=Config(signature_version="s3v4"),  # path/virtual both OK; RunPod is picky about auth, not path
    )
    public_base = f"{RP_ENDPOINT.rstrip('/')}/{RP_BUCKET}"
    return client, RP_BUCKET, RP_REGION, public_base


def handler(event):
    inp = event.get("input", {})
    video_url = inp.get("video_url")
    out_name  = inp.get("output_name", f"output-{uuid.uuid4().hex}.mp4")

    if not video_url:
        return {"error": "video_url is required"}

    s3, bucket, region, public_base = build_client_and_urls()
    s3_key = f"{PUBLIC_PREFIX}/{out_name.lstrip('/')}"

    with tempfile.TemporaryDirectory() as td:
        in_mp4  = os.path.join(td, "input.mp4")
        # keep a persistent copy on the mounted volume
        out_mp4 = os.path.join("/runpod-volume", out_name)

        try:
            # Download source video
            with requests.get(video_url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(in_mp4, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)

            # Run captioner
            proc = subprocess.run(
                ["python", "/app/caption.py", in_mp4, "--output", out_mp4],
                check=True,
                capture_output=True,
                text=True,
            )

            if not os.path.exists(out_mp4):
                return {"error": "caption.py completed but no output file found."}

            # Upload (no ACLs; rely on presigned or bucket policy)
            s3.upload_file(out_mp4, bucket, s3_key, ExtraArgs={"ContentType": "video/mp4"})

            # Presigned URL (works everywhere, including browsers, for AWS proper)
            presigned_url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": s3_key},
                ExpiresIn=URL_TTL_SECONDS,
            )

            # Plain URL (will only be browser-accessible if the bucket/prefix is public)
            plain_url = f"{public_base}/{s3_key}"

            return {
                "status": "ok",
                "backend": STORAGE_BACKEND,
                "file_url": presigned_url,   # always safe to use
                "plain_url": plain_url,      # requires public policy if you want no-token direct access
                "bucket": bucket,
                "region": region,
                "s3_key": s3_key,
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
