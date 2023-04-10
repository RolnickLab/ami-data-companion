from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/stats")


from pydantic import BaseModel


class Msg(BaseModel):
    msg: str


@router.get(
    "/",
    response_model=Msg,
    status_code=200,
    include_in_schema=False,
)
def test_hello_world() -> Any:
    return {"msg": "Hello world!"}
