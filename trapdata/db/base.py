import contextlib
import functools
import os
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
from trapdata.common.schemas import DatabaseURL

DIALECT_CONNECTION_ARGS = {
    "sqlite": {
        "timeout": 30,  # Increased timeout for bulk operationds
        "check_same_thread": False,
        "isolation_level": None,  # Autocommit mode for better performance
    },
    "postgresql": {
        # PostgreSQL-specific connection arguments (if any needed)
    },
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


@functools.lru_cache(maxsize=None)
def _get_engine(db_path_str: str, dialect: str, pid: int) -> sa.engine.Engine:
    """
    Build (and cache) a single Engine per (db_path, dialect, pid).

    `get_db` calls this function to get a cached Engine for the current process. 
    The `pid` argument is used to key the cache so that different processes 
    (including parent/child processes after forking) will compute a
    different cache key and each build their own Engine. This avoids reusing and 
    potentially corrupting connections that would otherwise be shared across forks.

    Arguments:
        db_path_str: database connection string (filepath or URL)
        dialect: database dialect (sqlite, postgresql, etc.)
        pid: process ID (used to key the cache for fork safety)
            Even though this is not used in the function itself,
            lru_cache will key the cache on it, so that a forked process will
            get a new engine instead of reusing the parent's engine.
    """
    engine_kwargs = {
        "echo": False,
        "future": True,
        "connect_args": DIALECT_CONNECTION_ARGS.get(dialect, {}),
    }

    if dialect == "postgresql":
        engine_kwargs.update(
            {
                "pool_size": 5,  # Reused across all calls now, so this can be modest
                "max_overflow": 10,
                "pool_pre_ping": True,
                "pool_recycle": 3600,
            }
        )

    return sa.create_engine(db_path_str, **engine_kwargs)


def get_db(db_path):
    """ """
    db_path = get_safe_db_path(db_path)
    dialect = get_dialect(db_path)

    # Reuse a single cached engine (and its connection pool) per
    # (db_path, pid), rather than creating a new engine/pool on every
    # call -- see _get_engine's docstring for why pid is included.
    db = _get_engine(db_path.render_as_string(hide_password=False), dialect, os.getpid())
    return db


def get_session_class(db_path, **kwargs) -> orm.sessionmaker[orm.Session]:
    """
    Use this to create a pre-configured Session class.
    Attach it to the running app.
    Then we don't have to pass around the db_path
    """
    Session = orm.sessionmaker(
        bind=get_db(db_path),
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
    dialect = get_dialect(db_path)

    if dialect == "sqlite" and db_path.database:
        path = pathlib.Path(db_path.database)
        timestamp = int(time.time())
        backup_path = path.with_stem(f"{path.stem}-{timestamp}")
        path.rename(backup_path)
        logger.info(f"Backup of {path.name} saved to {backup_path}")
        logger.info("Recreating database and tables")
        create_db(db_path)
    else:
        # Truncate every app table in place: clear all rows and reset 
        # auto-increment sequences, but leave the schema and Alembic 
        # migration state untouched, so migrations don't need to be re-run
        from . import Base

        engine = get_db(db_path)
        table_names = ", ".join(f'"{t.name}"' for t in Base.metadata.sorted_tables)
        logger.warning(f"Truncating all data in tables: {table_names}")
        with engine.begin() as conn:
            conn.execute(
                sa.text(f"TRUNCATE TABLE {table_names} RESTART IDENTITY CASCADE")
            )
        logger.info("All tables truncated. Schema and migrations are unchanged.")


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