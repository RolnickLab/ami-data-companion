import datetime
import subprocess
import sys

import typer
from rich import print
from sqlalchemy import select

from trapdata.cli import settings
from trapdata.db.base import check_db, get_session_class
from trapdata.db.models import MonitoringSession
from trapdata.db.models.occurrences import get_unique_species_by_track
from trapdata.tests.test_pipeline import process_deployments

cli = typer.Typer(no_args_is_help=True)


@cli.command()
def all():
    """
    Run pytest tests. Would like this to be the default command.
    """
    # return_code = pytest.main(["--doctest-modules", "-v", "."])
    # return_code = pytest.main(["-v", "."])

    return_code = subprocess.call(["pytest", "-v", "."])
    sys.exit(return_code)


@cli.command()
def nothing():
    print("It works!")


@cli.command()
def database():
    return check_db(db_path=settings.database_url, create=True, quiet=False)


@cli.command()
def pipeline():
    process_deployments()


@cli.command()
def species_by_track(event_day: datetime.datetime):
    """"""
    Session = get_session_class(settings.database_url)
    session = Session()
    event = session.execute(
        select(MonitoringSession).where(
            # MonitoringSession.base_directory="",  @TODO retrieve from settings?
            MonitoringSession.day
            == event_day.date(),
        )
    ).scalar_one()
    print(f"Matched of event: {event}")
    get_unique_species_by_track(
        settings.database_url,
        image_base_path=event.base_directory,
        monitoring_session=event,
        classification_threshold=0.1,
    )


if __name__ == "__main__":
    cli()
