from base import cli

if __name__ == "__main__":
    cli()

# import typer
# from pathlib import Path
#
#
# APP_NAME = "ami-manager"
#
#
# def main():
#     app_dir = typer.get_app_dir(APP_NAME)
#     config_path: Path = Path(app_dir) / "config.json"
#     if not config_path.is_file():
#         print("Config file doesn't exist yet")
#
#
# if __name__ == "__main__":
#     typer.run(main)
#
