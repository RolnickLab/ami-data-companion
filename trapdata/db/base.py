import contextlib
import pathlib
import time
from typing import Generator

import sqlalchemy as sa
import sqlalchemy.exc
from alembic import command as alembic
from alembic.config import Config
from rich import print
from sqlalchemy import orm

from trapdata import logger
from trapdata.common.types import DatabaseURL

DATABASE_SCHEMA_NAMESPACE = "trapdata"

DIALECT_CONNECTION_ARGS = {
    "sqlite": {
        "timeout": 10,  # A longer timeout is necessary for SQLite and multiple PyTorch workers
        "check_same_thread": False,
    },
    "postgresql": {"options": f"-csearch_path={DATABASE_SCHEMA_NAMESPACE}"},
}

SUPPORTED_DIALECTS = list(DIALECT_CONNECTION_ARGS.keys())


def get_safe_db_path(db_path: DatabaseURL) -> sa.engine.url.URL:
    """
    Return filepath or URL of database without credentials

    `db_path` supports any database connection string format supported by SQLAlchemy.

    sqlite_filepath = "~/trapdata.db"
    db_path = f"sqlite+pysqlite:///{file_path}",
    db_path = ":memory:"
    db_path = "postgresql://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]"
    """

    return sa.engine.url.make_url(db_path)


def get_alembic_config(db_path: DatabaseURL) -> Config:
    connection_string = get_safe_db_path(db_path).render_as_string(hide_password=False)
    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", "trapdata.db:migrations")
    alembic_cfg.set_main_option("sqlalchemy.url", connection_string)
    return alembic_cfg


def get_dialect(db_path: DatabaseURL) -> str:
    """
    Return the SQL dialect of the database (sqlite, postgresql, etc.)
    """
    return get_safe_db_path(db_path).get_dialect().name


def create_db(db_path: DatabaseURL) -> None:
    """
    Create database tables and sqlite file if necessary.
    """
    db_path = get_safe_db_path(db_path)

    logger.debug(f"Creating database tables for {db_path} if necessary")

    if get_dialect(db_path) == "sqlite":
        # Create parent directory if it doesn't exist
        assert db_path.database, "No filepath specified for sqlite database."
        logger.debug("Creating parent directories for database file if necessary")
        pathlib.Path(db_path.database).parent.mkdir(parents=True, exist_ok=True)

    db = get_db(db_path)

    from . import Base

    with db.connect() as con:
        if not db.dialect.has_schema(con, DATABASE_SCHEMA_NAMESPACE):
            print("CREATING SCHEMS")
            con.execute(sqlalchemy.schema.CreateSchema(DATABASE_SCHEMA_NAMESPACE))
    Base.metadata.schema = DATABASE_SCHEMA_NAMESPACE
    Base.metadata.create_all(db, checkfirst=True)
    alembic_cfg = get_alembic_config(db_path)
    alembic.stamp(alembic_cfg, "head")


def migrate(db_path: DatabaseURL) -> None:
    """
    Run database migrations.

    # @TODO See this post for a more complete implementation
    # https://pawamoy.github.io/posts/testing-fastapi-ormar-alembic-apps/
    """
    logger.debug("Running any database migrations if necessary")
    alembic_cfg = get_alembic_config(db_path)
    alembic.upgrade(alembic_cfg, "head")


def get_db(db_path, create=False, update=False):
    """ """
    db_path = get_safe_db_path(db_path)

    dialect = get_dialect(db_path)

    db = sa.create_engine(
        db_path,
        echo=False,
        future=True,
        connect_args=DIALECT_CONNECTION_ARGS.get(dialect, {}),
    )
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
    Convenience method to check if a database is accessible and create it if it doesn't exist,

    Allows the interface calling this method to handle any errors gracefully.

    @TODO rethink this and which interfaces are using it.
    """
    from trapdata.db.models import __models__

    db_dsn = get_safe_db_path(db_path)
    try:
        logger.info(f"Checking DB {db_dsn}")

        if create:
            create_db(db_path)

        if update:
            if get_dialect(db_path) == "sqlite":
                migrate(db_path)
            else:
                logger.warning(
                    "Skipping database migrations for non-sqlite database. Run them manually."
                )

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
        logger.warning(msg)
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


def reset_db(db_path: DatabaseURL) -> None:
    db_path = get_safe_db_path(db_path)
    if get_dialect(db_path) == "sqlite" and db_path.database:
        path = pathlib.Path(db_path.database)
        timestamp = int(time.time())
        backup_path = path.with_stem(f"{path.stem}-{timestamp}")
        path.rename(backup_path)
        logger.info(f"Backup of {path.name} saved to {backup_path}")
    else:
        raise NotImplementedError("Only implemented for sqlite databases")
    logger.info("Recreating database and tables")
    create_db(db_path)


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


from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def get_async_session_class(db_path: str) -> async_sessionmaker[AsyncSession]:
    async_engine = create_async_engine(db_path, pool_pre_ping=True)

    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    return async_session_maker
