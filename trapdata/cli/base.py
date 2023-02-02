import typer

from trapdata.cli import export, tracking, shell


cli = typer.Typer()
cli.add_typer(export.cli, name="export")
cli.add_typer(tracking.cli, name="tracking")
cli.add_typer(shell.cli, name="shell")


if __name__ == "__main__":
    cli()
