import sys
import time
import multiprocessing
from functools import partial

from trapdata import db, ml
from trapdata.models.queue import ImageQueue
from trapdata.models.detections import save_detected_objects


def process_queue(queue):
    localization_results_callback = partial(save_detected_objects, queue.db_path)
    queue.process_queue(
        model_name="Custom FasterRCNN",
        models_dir="/home/michael/.config/trapdataanalyzer/models",
        base_directory=queue.db_path,  # base path for relative images
        results_callback=localization_results_callback,
        batch_size=2,
        num_workers=2,
    )
    print("Localization set complete")


def watch_queue(db_path, interval=1):
    # query queue(s)
    # the queue should contain the model name and parameters to be used
    # the model name is from the model registry, which is where the weights are specified as well
    # the queue entries can have their status updated (waiting, preprocessing, classifying, done)
    # the server queries the queue, groups by model, starts predict dataloader with that batch
    # if we want to do one model at a time, not sure how to order them

    queue = ImageQueue(db_path)

    while True:
        print(f"Images in queue: {queue.queue_count()}")
        if queue.queue_count() > 0:
            process_queue(queue)
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
    db_path = sys.argv[1]

    watch_queue(db_path)
    # watch_queue_in_background(db_path)
