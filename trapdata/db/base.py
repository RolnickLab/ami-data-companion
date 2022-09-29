import contextlib
import pathlib

import sqlalchemy as sa
from sqlalchemy import orm

from trapdata import logger
from trapdata.common.files import archive_file


def db_path(directory):
    db_name = "trapdata.db"
    filepath = pathlib.Path(directory) / db_name
    return filepath


def get_db(directory=None, create=False):

    if directory:
        filepath = db_path(directory)
        if filepath.exists():
            if create:
                archive_file(filepath)
        else:
            create = True
        location = filepath

    else:
        # Only works in a scoped session. Used for tests.
        location = ":memory:"

    db = sa.create_engine(
        f"sqlite+pysqlite:///{location}",
        echo=False,
        future=True,
    )

    if create:
        from . import Base

        logger.info("Creating database tables")
        Base.metadata.create_all(db)

    return db


@contextlib.contextmanager
def get_session(directory):
    """
    Convience method to start and close a database session.

    The database is a file-based sqlite database, so we store
    in the base directory of the trap images.
    All image paths in the database will be relative to the location
    of this base directory.


    SQL Alchemy also has a sessionmaker utility that could be used.
    # return orm.sessionmaker(db).begin()

    Usage:

    >>> directory = "/tmp/images"
    >>> with get_session(directory) as sess:
    >>>     num_images = sess.query(Image).filter_by().count()
    >>> num_images
    0
    """
    db = get_db(directory)
    session = orm.Session(db)

    yield session

    session.close()


def check_db(directory):
    """
    Try opening a database session.
    """
    from trapdata.models import __models__

    try:
        with get_session(directory) as sess:
            # May have to check each model to detect schema changes
            # @TODO probably a better way to do this!
            for ModelClass in __models__:
                logger.debug(f"Testing model {ModelClass}")
                count = sess.query(ModelClass).count()
                logger.debug(
                    f"Found {count} records in table '{ModelClass.__tablename__}'"
                )
    except sa.exc.OperationalError as e:
        logger.error(f"Error opening database session: {e}")
        return False
    else:
        return True


def query(directory, q, **kwargs):
    with get_session(directory) as sess:
        return list(sess.query(q, **kwargs))


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
