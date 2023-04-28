import boto3
import botocore
import dotenv
from mypy_boto3_s3.client import S3Client
from rich import print

dotenv.load_dotenv()

S3_ENDPOINT = "https://object-arbutus.cloud.computecanada.ca"
# https://object-arbutus.cloud.computecanada.ca/<CONTAINER>/<PREFIX>/<FILENAME>
PUBLIC_BASE_URL = S3_ENDPOINT
DEFAULT_BUCKET = "ami-trapdata"


def parse_s3_url(url: str):
    """Parse an S3 URL into its components.

    Args:
        url (str): An S3 URL.

    Returns:
        dict: A dictionary with keys: bucket, key, and url.
    """
    if url.startswith("s3://"):
        url = url[5:]
    if url.startswith(S3_ENDPOINT):
        url = url[len(S3_ENDPOINT) :]
    if url.startswith("/"):
        url = url[1:]

    parts = url.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return {
        "bucket": bucket,
        "key": key,
        "url": f"{S3_ENDPOINT}/{bucket}/{key}",
    }


def with_trailing_slash(s: str):
    return s if s.endswith("/") else f"{s}/"


def without_trailing_slash(s: str):
    return s.rstrip("/")


def get_session():
    session = boto3.Session()
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
    for item in (
        bucket.objects.filter(Prefix=with_trailing_slash(deployment)).limit(limit).all()
    ):
        if item.size > 0:  # Ignore directories
            yield item


def public_url(key: str):
    return f"{PUBLIC_BASE_URL}/{DEFAULT_BUCKET}/{key}"


def test():
    # boto3.set_stream_logger(name="botocore")

    projects = list_projects()
    print("Projects:", projects)
    for project in projects:
        deployments = list_deployments(project)

        for deployment in deployments:
            # print("\t\tFile Count:", count_files(deployment))
            print("\tDeployment:", deployment)

            for file in list_files(deployment, limit=3):
                # print(file)
                print("\t\t\tSample:", public_url(file.key))


if __name__ == "__main__":
    test()
    import ipdb

    ipdb.set_trace()
