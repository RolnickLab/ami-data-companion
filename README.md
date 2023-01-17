# AMI Data Companion

Desktop app for analyzing images from autonomous insect monitoring stations

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


### Dependencies


- Requires Python 3.7 *or above*. Use Anaconda (or just "conda") if you need to maintain multiple versions of Python or are unfamiliar with using Python and scientific packages, it is especially helpful on Windows. https://www.anaconda.com/
- Requires Git to clone the source code and stay up-to-date with the latest changes. Anaconda comes with Git, but the GitHub Desktop application works well if you are less familiar with git. https://desktop.github.com/

### Installation (for non-developers)

Install (or upgrade) the package with the following command

```
pip install --editable git+https://github.com/mihow/trapdata.git#egg=trapdata
```

### Installation (for developers)

Create an environment just for AMI and the trapdata manager using conda (or virtualenv) 

```
conda create -n ami python=3.10 anaconda
```

Clone the repository using the command line or the GitHub desktop app.

```
git clone git@github.com:mihow/trapdata.git
```

Install as an editable package. This will install the dependencies and install the `trapdata` console command

```
cd trapdata
pip install -e .
```

Test the whole backend pipeline without the GUI using this command
```
python trapdata/tests/test_pipeline.py
```

### Usage

- Make a directory of sample images to test & learn the whole workflow more quickly.

- Launch the app by opening a terminal, activating your python environment and then typing the command: 
```trapdata```

- When the app GUI window opens, it will prompt you to select the root directory with your trapdata. Choose the directory with your sample images.

- The first time you process an image the app will download all of the ML models needed, which can take some time. The progress is only visible in the console!

- **Important** Look at the output in the console/terminal to see the status of the application. The GUI may appear to hang or be stuck when scanning or processing a larger number of images, but it is not. For the time being, most feedback will only appear in the terminal.

- All progress and intermediate results are saved to a local database, so if you close the program or it crashes, the status will not be lost and you can pick up where it left off.

- The cropped images, reports, cached models & local database are stored in the "user data" directory which can be changed in the Settings panel. By default, the user data directory is in one of the locations below, You 

    macOS: 
    ```/Library/Application Support/trapdata/```

    Linux:
    ```~/.config/trapdata```

    Windows:
    ```%AppData%/trapdata```




