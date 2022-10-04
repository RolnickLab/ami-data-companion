import os
import sys
import time
import multiprocessing

from trapdata.ml.models.localization import MothObjectDetector_FasterRCNN
from trapdata.db.models.queue import ImageQueue


def watch_queue(db_path, interval=1):
    # query queue(s)
    # the queue should contain the model name and parameters to be used
    # the model name is from the model registry, which is where the weights are specified as well
    # the queue entries can have their status updated (waiting, preprocessing, classifying, done)
    # the server queries the queue, groups by model, starts predict dataloader with that batch
    # if we want to do one model at a time, not sure how to order them

    model = MothObjectDetector_FasterRCNN(db_path)
    queue = ImageQueue(db_path, model=model)

    while True:
        print(f"Images in queue: {queue.queue_count()}")
        if queue.queue_count() > 0:
            queue.process_queue()
        time.sleep(interval)


def watch_queue_in_background(db_path):
    # https://docs.python.org/3/library/multiprocessing.html
    multiprocessing.set_start_method(
        "spawn"
    )  # Required for PyTorch, default on Windows
    p = multiprocessing.Process(
        target=watch_queue,
        args=(db_path,),
        daemon=True,
    )
    p.start()

    while True:
        print(p)
        if not p.is_alive():
            print("Subprocess exited")
            sys.exit(1)
        time.sleep(1)


if __name__ == "__main__":
    # db_path = "sqlite+pysqlite:///home/michael/.config/trapdata/trapdata.db"
    # db_path = sys.argv[1]
    db_path = os.environ["DATABASE_URL"]

    watch_queue(db_path)
    # watch_queue_in_background(db_path)
