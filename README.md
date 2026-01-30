# AMI Data Companion

Desktop app for analyzing images from autonomous insect monitoring stations using deep learning models

<table>
<tr>
<td>
<img width="200px" alt="Monitoring station deployment in field" src="https://user-images.githubusercontent.com/158175/212795444-3f638f4b-78f9-4f94-adf0-f2269427b441.png">
</td>
<td>
<img width="200px" alt="Screenshot of desktop application" src="https://user-images.githubusercontent.com/158175/212795253-6545c014-f82a-42c9-bd3a-919e471626cf.png">
</td>
<td>
<img width="200px" alt="Emerald moths detected in processed images" src="https://user-images.githubusercontent.com/158175/212794681-45a51172-1431-4475-87a8-9468032d6f7d.png">
</td>
</tr>
</table>

## Dependencies

- Requires Python 3.10. Use [Anaconda](https://www.anaconda.com/) (or [miniconda](https://docs.conda.io/en/latest/miniconda.html)) if you need to maintain multiple versions of Python or are unfamiliar with using Python and scientific packages, it is especially helpful on Windows. [PyEnv](https://github.com/pyenv/pyenv) is also a popular tool for managing multiple versions of python if you are familiar with the command line.

## Installation (for non-developers)

Install (or upgrade) the package with the following command

```sh
pip install https://github.com/RolnickLab/ami-data-companion/archive/main.zip
```

Optionally test the installation with the following command

```sh
ami test pipeline
```

## Installation (for developers)

Create an environment just for AMI and the data companion using conda (or virtualenv)

```sh
conda create -n ami python=3.10 anaconda
```

Clone the repository using the command line or the GitHub desktop app.

```sh
git clone git@github.com:RolnickLab/ami-data-companion.git
```

Install as an editable package. This will install the dependencies and install the `trapdata` console command

```sh
cd ami-data-companion
pip install -e .
```

Test the whole backend pipeline without the GUI using this command

```sh
python trapdata/tests/test_pipeline.py
# or
ami test pipeline
```

Run all other tests with:

```sh
ami test all
```

## GUI Usage

- Make a directory of sample images to test & learn the whole workflow more quickly.

- Launch the app by opening a terminal and then typing the command `ami gui`. You may need to activate your Python 3.10 environment first (`conda activate ami`).

- When the app GUI window opens, it will prompt you to select the root directory with your trapdata. Choose the directory with your sample images.

- The first time you process an image the app will download all of the ML models needed, which can take some time. _The status is only visible in the console!_

- **Important:** Look at the text in the console/terminal/shell to see the status of the application. The GUI may appear to hang or be stuck when scanning or processing a larger number of images, but it is not. For the time being, most feedback will only appear in the terminal.

- All progress and intermediate results are saved to a local database, so if you close the program or it crashes, the status will not be lost and you can pick up where it left off.

- The cropped images, reports, cached models & local database are stored in the "user data" directory which can be changed in the Settings panel. By default, the user data directory is in one of the locations below, You

  macOS:
  `/Library/Application Support/trapdata/`

  Linux:
  `~/.config/trapdata`

  Windows:
  `%AppData%/trapdata`

A short video of the application in use can be seen here: https://www.youtube.com/watch?v=DCPkxM_PvdQ

## CLI Usage

Configure models and the image_base_path for the deployment images you want to process, then see the example workflow below. Help can be viewed for any of the subcommands with `ami export --help`.

### Settings

There are two ways to configure settings

1. Using the graphic interface:
   - Run `ami gui` and click Settings. This will write settings to the file `trapdata.ini`
2. Using environment variables
   - Copy `.env.example` to `.env` and edit the values, or
   - Export the env variables to your shell environment

The CLI will read settings from either source, but will prioritize environment variables. The GUI only reads from `trapdata.ini`.

### Example workflow

The AMI Data Companion operates using a pipeline for data processing. By default it lists input images, finds detections in them, identifies which images need to be classified to moth species using a moth/nonmoth threshold, computes features for each object, and performs a photo-to-photo tracking procedure.

```sh
ami --help
ami test pipeline
ami show settings
ami import --no-queue
ami show sessions
ami queue sample --sample-size 10
ami queue status
ami run
ami show occurrences
ami queue all
ami run
ami queue status --watch  # Run in a 2nd shell or on another server connected to the same DB
ami show occurrences
ami export occurrences --format json --outfile denmark_sample.json --collect-images
```

The models, batch sizes, thresholds, etc. used for each step of the pipeline can be changed by modifying `.env`.

#### Example outputs

```sh
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ Queue                       ┃ Unprocessed ┃ Queued ┃ Done  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ Source images               │ 3500        │ 300    │ 200   │
│ Detected objects            │ 1200        │ 1100   │ 100   │
│ Unclassified objects        │ 0           │ 0      │ 50    │
│ Detections without features │ 0           │ 0      │ 50    │
│ Untracked detections        │ 0           │ 0      │ 50    │
└─────────────────────────────┴─────────────┴────────┴───────┘
```

The "queue" column indicates the type of objects in the queue, e.g., images, detections, etc. The "unprocessed" column indicates how many of those objects have not gone through the next step in the pipeline. The "queue" column indicates how many of those objects will be put through the pipeline once the `ami run` command is run.

For example, the outputs above indicate:

- **Source images**: out of a total of 4000 original source images, 500 were queued. 200 of the images in the queue have been processed into detections.
- **Detected objects**: In the 200 images, 1200 objects have been detected. All of were queued for the next steps of the pipeline. 100 have already moved into the next step
- **Unclassified objects**: 50 of the detections met the detection threshold. The lack of objects in the last 3 rows of the "unprocessed" and "queued" table means that all of these have had features generated for them, classifications have been saved from them, and tracking has been completed for them.

#### Troubleshooting queueing

```sh
ami queue --help
ami queue clear # Clear images that haven't been processed into detections
ami queue clear-everything # Clear queue for all of pipeline
ami queue reprocess-detections # Add all detections to the queue for reprocessing (e.g. use a different classifier)
ami queue unprocessed-images # Queue all images that have not been processed into detections
ami queue unprocessed-detections # Queue all detections that have not been processed in later steps of the pipeline
```

## Database

By default both the GUI and CLI will automatically create a local sqlite database by default. It is recommended to use a PostgreSQL database to increase performance for large datasets and
to process data from multiple server nodes.

You can test using PostgreSQL using Docker:

```sh
docker run -d -i --name ami-db -p 5432:5432 -e POSTGRES_HOST_AUTH_METHOD=trust -e POSTGRES_DB=ami postgres:14
docker logs ami-db --tail 100
```

Change the database connection string in the GUI Settings to `postgresql://postgres@localhost:5432/ami`
(or set it in the environment settings if only using the CLI)

Stop and remove the database container:

```sh
docker stop ami-db && docker remove ami-db
```

A script is available in the repo source to run the commands above.
`./scrips/start_db_container.sh`

## Adding new models

1. Create a new inference class in `trapdata/ml/models/classification.py` or `trapdata/ml/models/localization.py`. All models inherit from `InferenceBaseClass`, but there are more specific classes for classification and localization and different architectures. Choose the appropriate class to inherit from. It's best to copy an existing inference class that is similar to the new model you are adding.

2. Upload your model weights and category map to a cloud storage service and make sure the file is publicly accessible via a URL. The weights will be downloaded the first time the model is run. Alternatively, you can manually add the model weights to the configured `USER_DATA_PATH` directory under the subdir `USER_DATA_PATH/models/` (on macOS this is `~/Library/Application Support/trapdata/models`). However the model will not be available to other users unless they also manually add the model weights. The category map json file is simply a dict of species names and their indexes in your model's last layer. See the existing category maps for examples.

3. Select your model in the GUI settings or set the `SPECIES_CLASSIFICATION_MODEL` setting. If the model inherits from `SpeciesClassifier` class, it will automatically become one of the valid choices.

## Clearing the cache & starting fresh

Remove the index of images, all detections and classifications by removing the database file. This will not remove the images themselves, only the metadata about them. The database is located in the user data directory.

On macOS:

```
rm ~/Library/Application\ Support/trapdata/trapdata.db
```

On Linux:

```
rm ~/.config/trapdata/trapdata.db
```

On Windows:

```
del %AppData%\trapdata\trapdata.db
```

## Running the web API

The model inference pipeline can be run as a web API using FastAPI. This is what the Antenna platform uses to process images.

To run the API, use the following command:

```sh
ami api
```

View the interactive API docs at http://localhost:2000/

## Running the Antenna Worker

The worker polls the Antenna platform API for queued image processing jobs, downloads images, runs detection and classification, and posts results back to Antenna.

**Setup:**

1. Get your Antenna auth token from your Antenna project settings
2. Configure the worker in `.env`:

```sh
AMI_ANTENNA_API_BASE_URL=https://antenna.insectai.org/api/v2  # Or your Antenna instance
AMI_ANTENNA_API_AUTH_TOKEN=your_token_here
AMI_ANTENNA_API_BATCH_SIZE=4
AMI_NUM_WORKERS=2  # Safe for REST API (atomic task dequeue)
```

**Register pipelines (optional):**

Register available ML pipelines with your Antenna projects:

```sh
ami worker register "My Worker Name" --project 1 --project 2
# Or register for all accessible projects:
ami worker register "My Worker Name"
```

**Run the worker:**

```sh
# Process all pipelines:
ami worker

# Or specify specific pipeline(s):
ami worker --pipeline moth_binary
ami worker --pipeline moth_binary --pipeline panama_moths_2024
```

The worker will:

1. Poll Antenna for jobs matching the specified pipeline(s)
2. Download images from the job queue
3. Run detection and classification
4. Post results back to Antenna
5. Repeat until queue is empty, then sleep and poll again

**Notes:**

- Multiple workers can run in parallel (they won't duplicate work)
- Auth token ties results to your Antenna project
- Worker continues running until interrupted (Ctrl+C)
- Safe to run multiple workers on different machines

For more information, see the [Antenna platform documentation](https://github.com/RolnickLab/antenna).

## Web UI demo (Gradio)

A simple web UI is also available to test the inference pipeline. This is a quick way to test models on a remote server via a web browser.

```sh
ami gradio
```

Open http://localhost:7861/

Use ngrok to temporarily expose localhost to the internet:

```sh
ngrok http 7861
```
