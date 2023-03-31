import typer

from trapdata.cli import export, run, shell, show, test

cli = typer.Typer(no_args_is_help=True)
cli.add_typer(export.cli, name="export", help="Export data in various formats")
cli.add_typer(shell.cli, name="shell", help="Open an interactive shell")
cli.add_typer(test.cli, name="test", help="Run tests")
cli.add_typer(show.cli, name="show", help="Show data for use in other commands")
cli.add_typer(run.cli, name="run", help="Commands for processing data")


@cli.command()
def gui():
    """
    Launch graphic interface
    """
    from trapdata.ui.main import run

    run()


if __name__ == "__main__":
    cli()
