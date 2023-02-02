import typer

from trapdata.cli import export, shell


cli = typer.Typer()
cli.add_typer(export.cli, name="export")
cli.add_typer(shell.cli, name="shell")


if __name__ == "__main__":
    cli()
