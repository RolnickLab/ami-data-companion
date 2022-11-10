import contextlib
import pathlib

import sqlalchemy as sa
from sqlalchemy import orm

from trapdata import logger


def get_safe_db_path(db):
    # Return filepath or URL of database without credentials
    return db.path


def get_db(db_path, create=False):
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

    db = sa.create_engine(
        str(db_path),
        echo=False,
        future=True,
        connect_args={
            "timeout": 10,  # A longer timeout is necessary for SQLite and multiple PyTorch workers
        },
    )

    if create:
        from . import Base

        logger.info("Creating database tables if necessary")
        if db.dialect.name == "sqlite":
            db_filepath = pathlib.Path(db.url.database)
            if not db_filepath.exists():
                logger.info(f"Creating {db_filepath} and parent directories")
                db_filepath.parent.mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(db, checkfirst=True)

    return db


def get_session_class(db_path):
    """
    Use this to create a pre-configured Session class.
    Attach it to the running app.
    Then we don't have to pass around the db_path
    """
    Session = orm.sessionmaker(
        bind=get_db(db_path),
        expire_on_commit=False,  # Only need this for select methods (`pull_n_from_queue`)
    )
    return Session


@contextlib.contextmanager
def get_session(db_path):
    """
    Convenience method to start and close a database session.

    The database is a file-based sqlite database, so we store
    in the base directory of the trap images.
    All image paths in the database will be relative to the location
    of this base directory.


    SQL Alchemy also has a sessionmaker utility that could be used.
    # return orm.sessionmaker(db).begin()

    Usage:

    >>> directory = "/tmp/images"
    >>> with get_session(db_path) as sesh:
    >>>     num_images = sesh.query(Image).filter_by().count()
    >>> num_images
    0
    """
    db = get_db(db_path)

    session = orm.Session(
        db,
        # @TODO this only needs to happen
        # in the `pull_n_from_queue` method
        # but need to use sessionmaker first.
        expire_on_commit=False
        # autoflush=False,
        # autocommit=False,
    )

    try:
        yield session
    except Exception as e:
        logger.error(e)
        session.rollback()
        raise
    finally:
        session.close()


def check_db(db_path, create=True, quiet=False):
    """
    Try opening a database session.
    """
    from trapdata.db.models import __models__

    logger.debug(f"Checking DB {db_path}")

    try:
        get_db(db_path, create=True)
        with get_session(db_path) as sesh:
            # May have to check each model to detect schema changes
            # @TODO probably a better way to do this!
            for ModelClass in __models__:
                logger.debug(f"Testing model {ModelClass}")
                count = sesh.query(ModelClass).count()
                logger.debug(
                    f"Found {count} records in table '{ModelClass.__tablename__}'"
                )
    except sa.exc.OperationalError as e:
        logger.error(f"Error opening database session: {e}")
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
