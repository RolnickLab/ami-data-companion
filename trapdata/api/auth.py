from functools import lru_cache

import requests

from . import settings

# @TODO use Pydantic settings or env vars


@lru_cache
def get_token():
    resp = requests.post(
        settings.api_base_url + "auth/token/login/",
        json={"email": settings.api_username, "password": settings.api_password},
    )
    print(resp.content)
    resp.raise_for_status()
    token = resp.json()["auth_token"]
    return token


def get_session():
    session = requests.Session()
    token = get_token()
    session.headers.update({"Authorization": f"Token {token}"})
    return session


def get_current_user():
    resp = get_session().get(settings.api_base_url + "users/me/")
    resp.raise_for_status()
    return resp.json()
