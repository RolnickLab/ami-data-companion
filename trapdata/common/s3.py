import boto3
import botocore
from rich import print

from mypy_boto3_s3.client import S3Client

S3_ENDPOINT = "https://object-arbutus.cloud.computecanada.ca"
# https://object-arbutus.cloud.computecanada.ca/<CONTAINER>/<PREFIX>/<FILENAME>
PUBLIC_BASE_URL = S3_ENDPOINT
DEFAULT_BUCKET = "ami-trapdata"


def with_trailing_slash(s: str):
    return s if s.endswith("/") else f"{s}/"


def without_trailing_slash(s: str):
    return s.rstrip("/")


def get_session():
    session = boto3.Session(profile_name="ami")
    return session


def get_client():
    session = get_session()
    client: S3Client = session.client(
        service_name="s3",
        endpoint_url=S3_ENDPOINT,
        config=botocore.config.Config(signature_version="s3v4"),
    )

    return client


def get_resource():
    session = get_session()
    s3 = session.resource(
        "s3",
        endpoint_url=S3_ENDPOINT,
        # api_version="s3v4",
    )
    return s3


def list_buckets():
    s3 = get_client()
    return s3.list_buckets().get("Buckets", [])


def get_bucket(bucket_name: str = DEFAULT_BUCKET):
    s3 = get_resource()
    bucket = s3.Bucket(DEFAULT_BUCKET)
    return bucket


def list_projects():
    client = get_client()
    resp = client.list_objects_v2(Bucket=DEFAULT_BUCKET, Prefix="", Delimiter="/")
    prefixes = [without_trailing_slash(item["Prefix"]) for item in resp["CommonPrefixes"]]  # type: ignore
    return prefixes


def list_deployments(project: str):
    client = get_client()
    resp = client.list_objects_v2(
        Bucket=DEFAULT_BUCKET, Prefix=with_trailing_slash(project), Delimiter="/"
    )
    if len(resp) and "CommonPrefixes" in resp.keys():
        prefixes = [
            without_trailing_slash(item["Prefix"]) for item in resp["CommonPrefixes"]  # type: ignore
        ]
    else:
        prefixes = []
    return prefixes


def count_files(deployment: str):
    bucket = get_bucket()
    count = sum(1 for _ in bucket.objects.filter(Prefix=deployment).all())
    return count


def list_files(deployment: str, limit: int = 10000):
    bucket = get_bucket()
    # bucket.objects.filter(Prefix=prefix).all()
    objects = (
        bucket.objects.filter(Prefix=with_trailing_slash(deployment)).limit(limit).all()
    )
    return objects


def public_url(key: str):
    return f"{PUBLIC_BASE_URL}/{DEFAULT_BUCKET}/{key}"


def test():
    # boto3.set_stream_logger(name="botocore")

    projects = list_projects()
    print("Projects:", projects)
    for project in projects:
        deployments = list_deployments(project)
        print("\tDeployments:", deployments)

        for deployment in deployments:
            # print("\t\tFile Count:", count_files(deployment))

            for file in list_files(deployment, limit=1):
                print(file)
                print("\t\t\tSample:", public_url(file.key))
