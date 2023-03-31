import random
import statistics

import sqlalchemy as sa

from trapdata import constants
from trapdata.db import models

from .base import get_session


def count_species(db_path, monitoring_session=None):
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

    with get_session(db_path) as sesh:
        query = (
            sa.select(
                sa.func.coalesce(
                    models.DetectedObject.specific_label,
                    models.DetectedObject.binary_label,
                ).label("label"),
                sa.func.mean(models.DetectedObject.specific_label_score).label(
                    "avg_score"
                ),
                sa.func.count().label("count"),
                models.DetectedObject.path.label("image_path"),
                # sa.func.max(models.DetectedObject.area_pixels),
            )
            .group_by("label")
            .order_by(sa.desc("count"))
        )
        if monitoring_session:
            query = query.filter_by(monitoring_session=monitoring_session)
        return sesh.execute(query).all()


def classification_results(
    db_path, monitoring_session=None, classification_threshold: float = -1
):
    """
    Get all species in a monitoring session.

    Fallback to moth/non-moth binary label if confidence score is too low.
    """

    query = sa.select(
        models.DetectedObject.id,
        models.DetectedObject.specific_label,
        models.DetectedObject.specific_label_score,
        models.DetectedObject.binary_label,
        models.DetectedObject.binary_label_score,
        models.DetectedObject.monitoring_session_id,
        models.DetectedObject.path,
    ).where(models.DetectedObject.binary_label == constants.POSITIVE_BINARY_LABEL)
    if monitoring_session:
        query = query.filter_by(monitoring_session=monitoring_session)

    results = []
    with get_session(db_path) as sesh:
        for record in sesh.execute(query).unique().all():
            if (
                record.specific_label_score
                and record.specific_label_score >= classification_threshold
            ):
                label = record.specific_label
            else:
                label = record.binary_label

            # Seeing the average score for the specific label is more helpful
            # than seeing the average binary score
            score = record.specific_label_score or 0

            results.append(
                {
                    "id": record.id,
                    "label": label,
                    "score": score,
                    "image_path": record.path,
                    "monitoring_session": record.monitoring_session_id,
                }
            )

    return results


def summarize_results(
    db_path,
    monitoring_session=None,
    classification_threshold: float = -1,
    num_examples=3,
):
    results = classification_results(
        db_path=db_path,
        monitoring_session=monitoring_session,
        classification_threshold=classification_threshold,
    )

    index = {result["label"]: [] for result in results}
    summary = []
    for result in results:
        index[result["label"]].append(result)

    for label, items in index.items():
        count = len(items)
        mean_score = statistics.mean([item["score"] for item in items])
        # items.sort(key=lambda item: item["score"], reverse=True)
        # best_example = items[0]
        random.shuffle(items)
        examples = [item for item in items[:num_examples]]
        summary.append(
            {
                "label": label,
                "count": count,
                "mean_score": mean_score,
                "examples": examples,
            }
        )
    summary.sort(key=lambda item: item["count"], reverse=True)
    return summary

    # Pandas implementation:
    # df = pd.DataFrame(results)
    # stats = df.groupby("label").agg({"score": ["max", "mean"], "label": ["count"]})

    # best_examples = (
    #     df.sort_values("score", ascending=False)
    #     .drop_duplicates("label", keep="first")
    #     .set_index("label")
    # )[["image_path"]]
    # summary = best_examples.join(stats)
    # summary.columns = ["example_image_path", "max_score", "mean_score", "count"]
    # return summary.to_dict(orient="records")


def count_species_with_images(db_session=None, monitoring_session=None):
    """
    Count number of species detected in a monitoring session.

    # @TODO try getting the largest example of each detected species
    # SQLAlchemy Query to GROUP BY and fetch MAX date
    """

    query = (
        sa.select(
            sa.func.coalesce(
                models.DetectedObject.specific_label,
                models.DetectedObject.binary_label,
                models.DetectedObject.path,
            ).label("label"),
            sa.func.count().label("count"),
            sa.func.max(models.DetectedObject.area_pixels),
        )
        .group_by("label")
        .order_by(sa.desc("count"))
    )
    if db_session:
        return db_session.execute(query).all()
    else:
        return query


def update_all_aggregates(directory):
    # Call the update_aggregates method of every model
    raise NotImplementedError

    with get_session(directory) as sesh:
        for ms in sesh.query(None).all():
            ms.update_aggregates(sesh)
        sesh.commit()
