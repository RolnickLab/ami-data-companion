import sqlalchemy as sa

from .base import get_session
from .models import *


def count_species(monitoring_session):
    """
    Count number of species detected in a monitoring session.

    # @TODO try getting the largest example of each detected species
    # SQLAlchemy Query to GROUP BY and fetch MAX date
    query = db.select([
        USERS.c.email,
        USERS.c.first_name,
        USERS.c.last_name,
        db.func.max(USERS.c.created_on)
    ]).group_by(USERS.c.first_name, USERS.c.last_name)
    """

    with get_session(monitoring_session.base_directory) as sess:
        return (
            sess.query(
                DetectedObject.specific_label,
                sa.func.count().label("count"),
            )
            .group_by(DetectedObject.specific_label)
            .order_by(sa.desc("count"))
            .filter_by(monitoring_session=monitoring_session)
            .all()
        )


# These are aparently needed because none of the fancy stuff seems to work
def monitoring_session_images(ms):
    with get_session(ms.base_directory) as sess:
        return list(sess.query(Image).filter_by(monitoring_session_id=ms.id).all())


def monitoring_session_images_count(ms):
    with get_session(ms.base_directory) as sess:
        return int(sess.query(Image).filter_by(monitoring_session_id=ms.id).count())


def update_all_aggregates(directory):
    with get_session(directory) as sess:
        for ms in sess.query(MonitoringSession).all():
            ms.update_aggregates()
        sess.commit()
