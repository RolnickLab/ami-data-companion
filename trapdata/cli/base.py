import typer

from trapdata.cli import export


cli = typer.Typer()
cli.add_typer(export.cli, name="export")


if __name__ == "__main__":
    cli()
