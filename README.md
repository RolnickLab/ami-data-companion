# AMI Data Companion

Desktop app for analyzing images from autonomous insect monitoring stations


## Development

### Dependencies

_Requires Python 3.7 or above! Use conda if you need to maintain muliple versions of Python. Supports any version higher (3.9, 3.10 are even better)_

https://www.anaconda.com/

Create an environment just for AMI and the trapdata manager, or use the default (base) environment created by Anaconda if you are not working with other Python software.

`conda create -n ami python=3.10 anaconda`

#### Installation

Clone repository using the command line or the GitHub deskop app. (Optionally create a virtualenv to install in).
```
git clone git@github.com:mihow/trapdata.git
```

Install as an editable package if you want to launch the `trapdata` command from any directory. 
```
pip install -e .
```

Test the whole backend pipeline without the GUI using this command
```
python trapdata/tests/test_pipeline.py
```

### Usage

Make a directory of sample images to test & learn the whole workflow more quickly.

Launch the app by opening a terminal, activating your python enviornment and then typing

```trapdata```

When the app GUI window opens, it will prompt you to select the root directory with your trapdata. Choose the directory with your sample images.

The first time you process an image the app will download all of the ML models needed, which can take some time.

**Important** Look at the output in the terminal to see the status of the application. The GUI may appear to hang or be stuck when scanning or processing a larger number of images, but it is not. For the time being, most feedback will onlu appear in the terminal.

All progress and intermediate results are saved to a local database, so if you close the program or it crashes, the status will not be lost and you can pick up where it left off.

The cropped images, reports, cached models & local database are stored in the "user data" directory which can be changed in the Settings panel. By default, the user data directory is in one of the locations below, You 

macOS: 
```/Library/Application Support/trapdata/```

Linux:
```~/.config/trapdata```

Windows:
```%AppData%/trapdata```




