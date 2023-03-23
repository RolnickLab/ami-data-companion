import typer
from rich import print
from rich.console import Console
from rich.table import Table
from sqlalchemy import select, func

from trapdata.db.base import get_session_class
from trapdata.settings import Settings as BaseSettings
from trapdata.cli import settings
from trapdata.db import models
from trapdata.db.models.events import get_or_create_monitoring_sessions
from trapdata.db.models.queue import add_sample_to_queue
from trapdata import logger
from trapdata.ml.pipeline import start_pipeline

cli = typer.Typer(no_args_is_help=True)

console = Console()


class PipelineSettings(BaseSettings):
    image_base_path: str  # Override default settings to enforce image_base_path


base_settings = dict(settings)
del base_settings["image_base_path"]
settings = PipelineSettings(**base_settings)


@cli.command()
def queue_sample(sample_size: int = 4):
    """
    Placeholder method to scan image_base_path for monitoring sessions
    and add a sample of images to the pipeline processing queue.
    """
    get_or_create_monitoring_sessions(settings.database_url, settings.image_base_path)
    add_sample_to_queue(settings.database_url, sample_size=sample_size)


@cli.command()
def pipeline():
    """
    Run all models on images currently in the queue.
    """
    Session = get_session_class(settings.database_url)
    session = Session()
    start_pipeline(
        session=session,
        image_base_path=settings.image_base_path,
        settings=settings,
    )


@cli.command()
def single():
    """ """
    Session = get_session_class(settings.database_url)
    session = Session()
    start_pipeline(
        session=session,
        image_base_path=settings.image_base_path,
        settings=settings,
        single=True,
    )


if __name__ == "__main__":
    cli()
