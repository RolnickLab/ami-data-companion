"""CLI commands for Antenna worker."""

from typing import Annotated

import typer

from trapdata.api.api import CLASSIFIER_CHOICES

cli = typer.Typer(help="Antenna worker commands for remote processing")


@cli.callback(invoke_without_command=True)
def run(
    ctx: typer.Context,
    pipelines: Annotated[
        list[str] | None,
        typer.Option(
            "--pipeline",
            help="Pipeline to use for processing (e.g., moth_binary, panama_moths_2024). Can be specified multiple times. Defaults to all pipelines if not specified."
        ),
    ] = None,
):
    """
    Run the worker to process images from the Antenna API queue.

    Can be invoked as 'ami worker' or 'ami worker run'.
    """
    # Only run the worker if no subcommand was invoked
    if ctx.invoked_subcommand is not None:
        return

    if not pipelines:
        pipelines = list(CLASSIFIER_CHOICES.keys())

    # Validate that each pipeline is in CLASSIFIER_CHOICES
    invalid_pipelines = [
        pipeline for pipeline in pipelines if pipeline not in CLASSIFIER_CHOICES.keys()
    ]

    if invalid_pipelines:
        raise typer.BadParameter(
            f"Invalid pipeline(s): {', '.join(invalid_pipelines)}. Must be one of: {', '.join(CLASSIFIER_CHOICES.keys())}"
        )

    from trapdata.antenna.worker import run_worker

    run_worker(pipelines=pipelines)


@cli.command("register")
def register(
    name: Annotated[
        str,
        typer.Argument(
            help="Name for the processing service registration (e.g., 'AMI Data Companion on DRAC gpu-03'). "
            "Hostname will be added automatically.",
        ),
    ],
    project: Annotated[
        list[int] | None,
        typer.Option(
            help="Specific project IDs to register pipelines for. "
            "If not specified, registers for all accessible projects.",
        ),
    ] = None,
):
    """
    Register available pipelines with the Antenna platform for specified projects.

    This command registers all available pipeline configurations with the Antenna platform
    for the specified projects (or all accessible projects if none specified).

    Examples:
        ami worker register "AMI Data Companion on DRAC gpu-03" --project 1 --project 2
        ami worker register "My Processing Service"  # registers for all accessible projects
    """
    from trapdata.antenna.registration import register_pipelines

    project_ids = project if project else []
    register_pipelines(project_ids=project_ids, service_name=name)
