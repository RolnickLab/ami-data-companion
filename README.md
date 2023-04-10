# AMI Data Manager

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
pip install https://github.com/RolnickLab/ami-data-manager/archive/main.zip
```

Optionally test the installation with the following command

```sh
trapdata-test
```

## Installation (for developers)

Create an environment just for AMI and the trapdata manager using conda (or virtualenv)

```sh
conda create -n ami python=3.10 anaconda
```

Clone the repository using the command line or the GitHub desktop app.

```sh
git clone git@github.com:RolnickLab/ami-data-manager.git
```

Install as an editable package. This will install the dependencies and install the `trapdata` console command

```sh
cd ami-data-manager
pip install -e .
```

Test the whole backend pipeline without the GUI using this command

```sh
python trapdata/tests/test_pipeline.py
```

## GUI Usage

- Make a directory of sample images to test & learn the whole workflow more quickly.

- Launch the app by opening a terminal and then typing the command ```trapdata```. You may need to activate your Python 3.10 environment first (`conda activate ami`).

- When the app GUI window opens, it will prompt you to select the root directory with your trapdata. Choose the directory with your sample images.

- The first time you process an image the app will download all of the ML models needed, which can take some time. _The status is only visible in the console!_

- **Important:** Look at the text in the console/terminal/shell to see the status of the application. The GUI may appear to hang or be stuck when scanning or processing a larger number of images, but it is not. For the time being, most feedback will only appear in the terminal.

- All progress and intermediate results are saved to a local database, so if you close the program or it crashes, the status will not be lost and you can pick up where it left off.

- The cropped images, reports, cached models & local database are stored in the "user data" directory which can be changed in the Settings panel. By default, the user data directory is in one of the locations below, You

    macOS:
    ```/Library/Application Support/trapdata/```

    Linux:
    ```~/.config/trapdata```

    Windows:
    ```%AppData%/trapdata```

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
```sh
ami --help
ami test pipeline
ami show settings
ami import --no-queue
ami show sessions
ami queue sample --sample-size 10
ami show queue
ami run
ami show occurrences
ami queue all
ami run
ami show queue --watch  # Run in a 2nd shell or on another server connected to the same DB
ami show occurrences
ami export occurrences --format json --outfile denmark_sample.json --collect-images
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
