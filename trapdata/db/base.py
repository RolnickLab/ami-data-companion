import contextlib
import pathlib
from typing import Generator
from rich import print

import sqlalchemy as sa
import sqlalchemy.exc
from sqlalchemy import orm
from alembic.config import Config
from alembic import command as alembic

from trapdata import logger


def get_safe_db_path(db_path: str):
    # Return filepath or URL of database without credentials
    return sa.engine.url.make_url(db_path)


def get_alembic_config(db_path: str) -> Config:
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", "trapdata.db:migrations")
    alembic_cfg.set_main_option("sqlalchemy.url", str(db_path))
    return alembic_cfg


def get_db(db_path, create=False, update=False):
    """
    db_path supports any database URL format supported by sqlalchemy
    sqlite_filepath = "~/trapdata.db"
    db_path = f"sqlite+pysqlite:///{file_path}",
    db_path = ":memory:"
    db_path = "postgresql://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]"
    """
    if not db_path:
        Exception("No database URL specified")
    else:
        pass
        # logger.debug(f"Using DB from path: {db_path}")
        # logger.debug(f"Using DB from path: {get_safe_db_path()}")

    db_path = get_safe_db_path(db_path)

    connect_args = {
        "sqlite": {
            "timeout": 10,  # A longer timeout is necessary for SQLite and multiple PyTorch workers
            "check_same_thread": False,
        },
        "postgresql": {},
    }

    db = sa.create_engine(
        db_path,
        echo=False,
        future=True,
        connect_args=connect_args.get(db_path.drivername, {}),
    )

    alembic_cfg = get_alembic_config(db_path)

    # @TODO this is basically checking if the environment is the local app install
    # let's make a way to set & check the environment.
    if db.dialect.name != "sqlite" and (create or update):
        logger.warn(
            "Database is something other that sqlite, you must create & update it with the CLI tools."
        )
        return db

    if create:
        from . import Base

        logger.info("Creating database tables if necessary")
        db_filepath = pathlib.Path(db.url.database)
        if not db_filepath.exists():
            logger.info(f"Creating {db_filepath} and parent directories")
            db_filepath.parent.mkdir(parents=True, exist_ok=True)
            Base.metadata.create_all(db, checkfirst=True)
            alembic.stamp(alembic_cfg, "head")

    if update:
        # @TODO See this post for a more complete implementation
        # https://pawamoy.github.io/posts/testing-fastapi-ormar-alembic-apps/
        logger.debug("Running any database migrations if necessary")
        alembic.upgrade(alembic_cfg, "head")

    return db


def get_session_class(db_path, **kwargs) -> orm.sessionmaker[orm.Session]:
    """
    Use this to create a pre-configured Session class.
    Attach it to the running app.
    Then we don't have to pass around the db_path
    """
    Session = orm.sessionmaker(
        bind=get_db(db_path, create=False, update=False),
        expire_on_commit=False,  # Currently only need this for `pull_n_from_queue`
        autoflush=False,
        autocommit=False,
        **kwargs,
    )
    return Session


@contextlib.contextmanager
def get_session(db_path: str, **kwargs) -> Generator[orm.Session, None, None]:
    """
    Convenience method to start and close a pre-configured database session.

    >>> db_path = ":memory:"
    >>> with get_session(db_path) as sesh:
    >>>     num_images = sesh.query(Image).filter_by().count()
    >>> num_images
    0
    """

    DatabaseSession = get_session_class(db_path, **kwargs)
    session = DatabaseSession()
    try:
        yield session
    except Exception as e:
        logger.error(e)
        session.rollback()
        raise
    finally:
        session.close()


def check_db(db_path, create=True, update=True, quiet=False):
    """
    Try opening a database session.
    """
    from trapdata.db.models import __models__

    db_dsn = get_safe_db_path(db_path)
    try:
        logger.info(f"Checking DB {db_dsn}")
        get_db(db_path, create=create, update=update)
        with get_session(db_path) as sesh:
            # May have to check each model to detect schema changes
            # @TODO probably a better way to do this!
            for ModelClass in __models__:
                logger.debug(f"Testing model {ModelClass}")
                count = sesh.query(ModelClass).count()
                logger.debug(
                    f"Found {count} records in table '{ModelClass.__tablename__}'"
                )
    except (sqlalchemy.exc.OperationalError, alembic.util.exc.CommandError) as e:
        msg = f"Error opening database session: {e}"
        logger.error(msg)
        if db_dsn.get_dialect().name == "sqlite":
            # @TODO standardize the way we check for a local environment and sqlite
            print(
                f'[b][yellow]Quick fix:[/yellow][/b] rename or delete the local database file: "{str(db_dsn.database)}"'
            )
        if quiet:
            return False
        else:
            raise
    else:
        return True


def query(db_path, q, **kwargs):
    with get_session(db_path) as sesh:
        return list(sesh.query(q, **kwargs))


def get_or_create(session, model, defaults=None, **kwargs):
    # https://stackoverflow.com/a/2587041/966058
    instance = session.query(model).filter_by(**kwargs).one_or_none()
    if instance:
        return instance, False
    else:
        kwargs |= defaults or {}
        instance = model(**kwargs)
        try:
            session.add(instance)
            session.commit()
        except Exception:
            # The actual exception depends on the specific database so we catch all exceptions.
            # This is similar to the official documentation: https://docs.sqlalchemy.org/en/latest/orm/session_transaction.html
            session.rollback()
            instance = session.query(model).filter_by(**kwargs).one()
            return instance, False
        else:
            return instance, True
