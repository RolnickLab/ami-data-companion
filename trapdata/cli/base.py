import pathlib
from typing import Optional

import typer

from trapdata.cli import export, queue, settings, shell, show, test
from trapdata.db.base import get_session_class
from trapdata.db.models.events import get_or_create_monitoring_sessions
from trapdata.db.models.queue import add_monitoring_session_to_queue
from trapdata.ml.pipeline import start_pipeline

cli = typer.Typer(no_args_is_help=True)
cli.add_typer(export.cli, name="export", help="Export data in various formats")
cli.add_typer(shell.cli, name="shell", help="Open an interactive shell")
cli.add_typer(test.cli, name="test", help="Run tests")
cli.add_typer(show.cli, name="show", help="Show data for use in other commands")
cli.add_typer(
    queue.cli, name="queue", help="Add and manage images in the processing queue"
)


@cli.command("import")
def import_data(image_base_path: Optional[pathlib.Path] = None, queue: bool = True):
    """
    Import images from a deployment into the database.

    Defaults to the `image_base_path` configured in .env or trapdata.ini
    The image_base_path is a proxy for a unique trap deployment.
    """
    image_base_path = image_base_path or settings.image_base_path
    events = get_or_create_monitoring_sessions(settings.database_url, image_base_path)
    if queue:
        for event in events:
            add_monitoring_session_to_queue(
                db_path=settings.database_url,
                monitoring_session=event,
            )
    print(events)


@cli.command("run")
def run_pipeline():
    """
    Process all images currently in the queue.
    """
    Session = get_session_class(settings.database_url)
    session = Session()
    start_pipeline(
        session=session,
        image_base_path=settings.image_base_path,
        settings=settings,
    )


if __name__ == "__main__":
    cli()
