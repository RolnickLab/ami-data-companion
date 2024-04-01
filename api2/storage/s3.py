import os
import pathlib
import urllib.parse
from dataclasses import dataclass

import boto3
import boto3.resources.base
import boto3.session
import botocore
import botocore.config
import botocore.exceptions
import PIL
import PIL.Image
from mypy_boto3_s3.client import S3Client
from mypy_boto3_s3.service_resource import Bucket, S3ServiceResource

from .. import logger


@dataclass
class S3Config:
    endpoint_url: str | None
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    prefix: str
    public_base_url: str | None = None


def with_trailing_slash(s: str):
    return s if s.endswith("/") else f"{s}/"


def without_trailing_slash(s: str):
    return s.rstrip("/")


def split_uri(s3_uri: str):
    """
    Split S3 URI into bucket and prefix
    # s3://<BUCKET>/<PREFIX>/<SUBPREFIX>
    """

    # If filename in path, remove it
    if "." in s3_uri.split("/")[-1]:
        s3_uri = "/".join(s3_uri.split("/")[:-1])

    path = s3_uri.replace("s3://", "")
    bucket, *prefix = path.split("/")
    prefix = "/".join(prefix)
    return bucket, prefix


def get_session(config: S3Config) -> boto3.session.Session:
    session = boto3.Session(
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
    )
    return session


def get_client(config: S3Config) -> S3Client:
    session = get_session(config)
    if config.endpoint_url:
        client = session.client(
            service_name="s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            config=botocore.config.Config(signature_version="s3v4"),
        )
    else:
        client = session.client(
            service_name="s3",
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
        )

    return client  # type: ignore


def get_resource(config: S3Config) -> S3ServiceResource:
    session = get_session(config)
    s3 = session.resource(
        "s3",
        endpoint_url=config.endpoint_url,
        # api_version="s3v4",
    )
    return s3  # type: ignore


def get_bucket(config: S3Config) -> Bucket:
    s3 = get_resource(config)
    bucket = s3.Bucket(config.bucket_name)
    return bucket


def read_file(config: S3Config, key: str) -> bytes:
    bucket = get_bucket(config)
    if config.prefix:
        # Use path join to ensure there are no extra or missing slashes
        key = pathlib.Path(config.prefix, key).as_posix()
    obj = bucket.Object(key)
    return obj.get()["Body"].read()


def write_file(config: S3Config, key: str, fileobj):
    s3 = get_resource(config)
    if config.prefix:
        # Use path join to ensure there are no extra or missing slashes
        key = os.path.join(config.prefix, key.lstrip("/"))
    resp = s3.meta.client.upload_fileobj(
        Fileobj=fileobj, Bucket=config.bucket_name, Key=key
    )
    print(resp)
    return key


def file_exists(config: S3Config, key: str) -> bool:
    bucket = get_bucket(config)
    if config.prefix:
        # Use path join to ensure there are no extra or missing slashes
        key = pathlib.Path(config.prefix, key).as_posix()
    obj = bucket.Object(key)
    try:
        obj.load()
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise
    else:
        return True


def read_image(config: S3Config, key: str) -> PIL.Image.Image:
    """
    Download an image from S3 and return as a PIL Image.
    """
    bucket = get_bucket(config)
    obj = bucket.Object(key)
    logger.info(f"Fetching image {key} from S3")
    print(obj)
    try:
        img = PIL.Image.open(obj.get()["Body"])
    except PIL.UnidentifiedImageError:
        logger.error(f"Could not read image {key}")
        raise
    return img


def public_url(config: S3Config, key: str):
    """
    Return public URL for a given key.

    @TODO Handle non-public buckets with signed URLs
    """
    if config.public_base_url:
        url = urllib.parse.urljoin(config.public_base_url, key.lstrip("/"))
    else:
        client = get_client(config)
        url = client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": config.bucket_name, "Key": key},
        ).split("?")[0]
    return url
