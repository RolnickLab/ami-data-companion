"""CLI commands for Antenna worker."""

from typing import Annotated

import typer

from trapdata.api.api import select_pipelines

cli = typer.Typer(help="Antenna worker commands for remote processing")

_PIPELINE_HELP = (
    "Pipeline to use for processing (e.g., moth_binary, panama_moths_2024). "
    "Can be specified multiple times. Defaults to the pipelines in the "
    "AMI_PIPELINES setting, or to all pipelines if that is not set."
)
_PROJECT_HELP = (
    "Limit to jobs for specific Antenna project IDs. "
    "Can be specified multiple times. Defaults to all projects the auth token has access to."
)


def _resolve_pipelines(pipelines: list[str] | None) -> list[str]:
    """Resolve the --pipeline option; it takes precedence over AMI_PIPELINES."""
    try:
        return list(select_pipelines(pipelines or None))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


def _start_worker(pipelines: list[str] | None, project: list[int] | None) -> None:
    """Shared implementation for ``ami worker`` and ``ami worker run``."""
    resolved = _resolve_pipelines(pipelines)
    project_ids = project or []

    from trapdata.antenna.worker import run_worker

    run_worker(pipelines=resolved, project_ids=project_ids)


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
        # Options given before a subcommand name do not reach the subcommand, so
        # it would run with the default selection.
        if pipelines or project:
            raise typer.BadParameter(
                "Put --pipeline and --project after the subcommand, for example "
                f"'ami worker {ctx.invoked_subcommand} --project 7'."
            )
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
    Register this service's pipelines with the Antenna platform for specified projects.

    This command registers the pipelines given with --pipeline, or else those in the
    AMI_PIPELINES setting, or every pipeline if neither is set, for the specified
    projects (or all accessible projects if none specified). Registration only adds
    pipelines: one left out stays registered in Antenna until it is removed there.

    The service name is read from the AMI_ANTENNA_SERVICE_NAME configuration setting.
    Hostname will be added automatically to the service name.

    Examples:
        ami worker register --project 1 --project 2
        ami worker register --pipeline mothbot_insect_orders_2025
        ami worker register  # registers for all accessible projects
    """
    # Resolve here, so an invalid value makes the command fail rather than log an
    # error and exit successfully.
    resolved = _resolve_pipelines(pipelines)

    from trapdata.antenna.registration import register_pipelines
    from trapdata.settings import read_settings

    settings = read_settings()
    project_ids = project if project else []
    register_pipelines(
        project_ids=project_ids,
        service_name=settings.antenna_service_name,
        pipeline_slugs=resolved,
    )
