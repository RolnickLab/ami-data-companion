# Can these be imported from the OpenAPI spec yaml?
import pydantic


class IncomingSourceImage(pydantic.BaseModel):
    id: int
    url: str
    width: int
    height: int
    timestamp: str

    class Config:
        # Allow extra fields to keep this schema simple
        extra = "ignore"
