import typer

from trapdata.cli import export, tracking


cli = typer.Typer()
cli.add_typer(export.cli, name="export")
cli.add_typer(tracking.cli, name="tracking")


if __name__ == "__main__":
    cli()
