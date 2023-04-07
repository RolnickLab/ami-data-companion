import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from trapdata.cli import settings
from trapdata.db.models.events import get_or_create_monitoring_sessions
from trapdata.db.models.queue import (
    ImageQueue,
    add_monitoring_session_to_queue,
    add_sample_to_queue,
    all_queues,
    clear_all_queues,
)

cli = typer.Typer(no_args_is_help=True)

console = Console()


@cli.command()
def sample(sample_size: int = 4):
    """
    Add a sample of images to the pipeline processing queue.

    Run this command after importing data to test the pipeline.
    """
    add_sample_to_queue(settings.database_url, sample_size=sample_size)


@cli.command()
def all():
    """
    Add all images to the processing queue.
    """
    events = get_or_create_monitoring_sessions(
        settings.database_url, settings.image_base_path
    )
    for event in events:
        add_monitoring_session_to_queue(
            db_path=settings.database_url,
            monitoring_session=event,
        )


@cli.command()
def unprocessed_detections():
    """
    Add all unprocessed detections to the processing queue.
    """
    for queue in all_queues(
        db_path=settings.database_url, base_directory=settings.image_base_path
    ).values():
        if not isinstance(queue, ImageQueue):
            queue.add_unprocessed()


@cli.command()
def clear():
    """
    Clear images from the first stage of the processing queue.
    """
    queue = ImageQueue(settings.database_url, base_directory=settings.image_base_path)
    queue.clear_queue()


@cli.command()
def clear_everything():
    """
    Clear all images and detections from all processing queues.
    """
    clear_all_queues(settings.database_url, base_directory=settings.image_base_path)


def get_queue_table():
    table = Table("Queue", "Unprocessed", "Queued", "Done")
    for name, queue in all_queues(
        db_path=settings.database_url, base_directory=settings.image_base_path
    ).items():
        row_values = [
            queue.name,
            queue.unprocessed_count(),
            queue.queue_count(),
            queue.done_count(),
        ]
        table.add_row(*[str(val) for val in row_values])
    return table


@cli.command()
def status(watch: bool = False):
    """
    Show counts waiting in each queue.
    """

    if watch:
        with Live(get_queue_table(), refresh_per_second=1) as live:
            while True:
                live.update(get_queue_table())
    else:
        console.print(get_queue_table())


if __name__ == "__main__":
    cli()
