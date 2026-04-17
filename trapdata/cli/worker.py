"""CLI commands for Antenna worker."""

from typing import Annotated

import typer

from trapdata.api.api import PIPELINE_CHOICES

cli = typer.Typer(help="Antenna worker commands for remote processing")

_PIPELINE_HELP = (
    "Pipeline to use for processing (e.g., moth_binary, panama_moths_2024). "
    "Can be specified multiple times. Defaults to all pipelines if not specified."
)
_PROJECT_HELP = (
    "Limit to jobs for specific Antenna project IDs. "
    "Can be specified multiple times. Defaults to all projects the auth token has access to."
)


def _validate_pipelines(pipelines: list[str] | None) -> list[str]:
    """Resolve and validate the --pipeline option."""
    if not pipelines:
        return list(PIPELINE_CHOICES.keys())

    invalid = [p for p in pipelines if p not in PIPELINE_CHOICES]
    if invalid:
        raise typer.BadParameter(
            f"Invalid pipeline(s): {', '.join(invalid)}. "
            f"Must be one of: {', '.join(PIPELINE_CHOICES.keys())}"
        )
    return pipelines


def _start_worker(pipelines: list[str] | None, project: list[int] | None) -> None:
    """Shared implementation for ``ami worker`` and ``ami worker run``."""
    validated = _validate_pipelines(pipelines)
    project_ids = project or []

    from trapdata.antenna.worker import run_worker

    run_worker(pipelines=validated, project_ids=project_ids)


@cli.callback(invoke_without_command=True)
def worker_callback(
    ctx: typer.Context,
    pipelines: Annotated[
        list[str] | None,
        typer.Option("--pipeline", help=_PIPELINE_HELP),
    ] = None,
    project: Annotated[
        list[int] | None,
        typer.Option("--project", help=_PROJECT_HELP),
    ] = None,
):
    """
    Run the worker to process images from the Antenna API queue.

    Can be invoked as 'ami worker' or 'ami worker run'.
    """
    if ctx.invoked_subcommand is not None:
        return
    _start_worker(pipelines, project)


@cli.command("run")
def run_cmd(
    pipelines: Annotated[
        list[str] | None,
        typer.Option("--pipeline", help=_PIPELINE_HELP),
    ] = None,
    project: Annotated[
        list[int] | None,
        typer.Option("--project", help=_PROJECT_HELP),
    ] = None,
):
    """
    Run the worker to process images from the Antenna API queue.

    Alias for 'ami worker' — both forms are identical.
    """
    _start_worker(pipelines, project)


@cli.command("register")
def register(
    project: Annotated[
        list[int] | None,
        typer.Option(
            "--project",
            help="Specific project IDs to register pipelines for. "
            "If not specified, registers for all accessible projects.",
        ),
    ] = None,
    pipelines: Annotated[
        list[str] | None,
        typer.Option("--pipeline", help=_PIPELINE_HELP),
    ] = None,
):
    """
    Register available pipelines with the Antenna platform for specified projects.

    This command registers the processing service and its pipeline configurations
    with the Antenna platform for the specified projects (or all accessible projects
    if none specified).

    When --pipeline is specified, only those pipelines are advertised (instead of all).

    The service name is read from the AMI_ANTENNA_SERVICE_NAME configuration setting.
    Hostname will be added automatically to the service name.

    Examples:
        ami worker register --project 1 --project 2
        ami worker register --pipeline mothbot_insect_orders_2025
        ami worker register  # registers all pipelines for all accessible projects
    """
    from trapdata.antenna.registration import register_pipelines
    from trapdata.settings import read_settings

    settings = read_settings()
    project_ids = project if project else []
    validated_pipelines = _validate_pipelines(pipelines)
    register_pipelines(
        project_ids=project_ids,
        service_name=settings.antenna_service_name,
        pipeline_slugs=validated_pipelines,
    )
