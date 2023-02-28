import typer
from rich import print

cli = typer.Typer()


@cli.command()
def nothing():
    print("It works!")


if __name__ == "__main__":
    cli()
