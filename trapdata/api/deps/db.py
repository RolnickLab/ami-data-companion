from typing import Generator

from sqlalchemy import orm

from trapdata.cli import read_settings
from trapdata.db.base import get_session_class

settings = read_settings()


def get_session() -> Generator[orm.Session, None, None]:
    Session = get_session_class(db_path=settings.database_url)
    with Session() as session:
        yield session
        session.close()
