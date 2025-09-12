import os, json, boto3
from botocore.config import Config

endpoint = "https://s3api-us-ks-2.runpod.io"
region   = "us-ks-2"
bucket   = "cqk82s22rj"

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
    region_name=region,
    config=Config(signature_version="s3v4", s3={"addressing_style":"path"}),
)

policy = {
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadGetObject",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": f"arn:aws:s3:::{bucket}/*"
  }]
}

s3.put_bucket_policy(Bucket=bucket, Policy=json.dumps(policy))
print("Bucket policy applied: public read for objects.")
