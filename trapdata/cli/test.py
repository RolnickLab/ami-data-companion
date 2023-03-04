import typer
from rich import print


from trapdata.tests import test_pipeline

cli = typer.Typer()


@cli.command()
def nothing():
    print("It works!")


@cli.command()
def pipeline():
    test_pipeline.run()


if __name__ == "__main__":
    cli()
