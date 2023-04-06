import pathlib

import typer
from rich.console import Console

from trapdata.cli import settings
from trapdata.db.base import get_session_class
from trapdata.db.models.events import get_or_create_monitoring_sessions
from trapdata.db.models.queue import (
    add_monitoring_session_to_queue,
    add_sample_to_queue,
)
from trapdata.ml.pipeline import start_pipeline
from trapdata.settings import PipelineSettings

cli = typer.Typer(no_args_is_help=True)

console = Console()


@cli.command()
def add_sample_to_queue_sample(sample_size: int = 4):
    """
    Placeholder method to scan image_base_path for monitoring sessions
    and add a sample of images to the pipeline processing queue.
    """
    get_or_create_monitoring_sessions(settings.database_url, settings.image_base_path)
    add_sample_to_queue(settings.database_url, sample_size=sample_size)


@cli.command()
def pipeline(import_data: bool = False):
    """
    Run all models on images currently in the queue.
    """
    base_settings = dict(settings)
    del base_settings["image_base_path"]
    pipeline_settings = PipelineSettings(**base_settings)

    if import_data:
        events = get_or_create_monitoring_sessions(
            pipeline_settings.database_url, pipeline_settings.image_base_path
        )
        for event in events:
            add_monitoring_session_to_queue(
                db_path=settings.database_url,
                monitoring_session=event,
            )

    Session = get_session_class(settings.database_url)
    session = Session()
    start_pipeline(
        session=session,
        image_base_path=pipeline_settings.image_base_path,
        settings=settings,
    )


@cli.command()
def single(deployment_data: pathlib.Path):
    """ """
    Session = get_session_class(settings.database_url)
    session = Session()
    start_pipeline(
        session=session,
        image_base_path=deployment_data,
        settings=settings,
        single=True,
    )


if __name__ == "__main__":
    cli()
