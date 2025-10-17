import pathlib
from typing import Optional

import typer

from trapdata.cli import db, export, queue, settings, shell, show, test
from trapdata.db.base import get_session_class
from trapdata.db.models.events import get_or_create_monitoring_sessions
from trapdata.db.models.queue import add_monitoring_session_to_queue
from trapdata.ml.pipeline import start_pipeline

cli = typer.Typer(no_args_is_help=True)
cli.add_typer(export.cli, name="export", help="Export data in various formats")
cli.add_typer(shell.cli, name="shell", help="Open an interactive shell")
cli.add_typer(test.cli, name="test", help="Run tests")
cli.add_typer(show.cli, name="show", help="Show data for use in other commands")
cli.add_typer(db.cli, name="db", help="Create, update and manage the database")
cli.add_typer(
    queue.cli, name="queue", help="Add and manage images in the processing queue"
)


@cli.command()
def gui():
    """
    Launch graphic interface
    """
    from trapdata.ui.main import run

    run()


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


@cli.command("api-pipeline")
def run_api_pipeline(source_image_ids: list[int]):
    """
    Process all images via the AMI platform API.
    """
    from trapdata.api.pipeline import start_pipeline as start_api_pipeline

    start_api_pipeline(
        source_image_ids=source_image_ids,
        settings=settings,
    )


@cli.command("gradio")
def run_gradio():
    """
    Run the gradio interface.
    """
    from trapdata.api.demo import app

    app.queue().launch(show_api=False, server_name="0.0.0.0", server_port=7861)


@cli.command("api")
def run_api(port: int = 2000):
    """
    Run the API.
    """
    import uvicorn

    uvicorn.run("trapdata.api.api:app", host="0.0.0.0", port=port, reload=True)


@cli.command("worker")
def worker():
    """
    Run the worker to process images from the REST API queue.
    """
    from trapdata.cli.worker import run_worker

    run_worker()


if __name__ == "__main__":
    cli()
