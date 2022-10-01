import multiprocessing
from .ui.main import run


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required if using a frozen exe on Windows, which we will be
    run()
