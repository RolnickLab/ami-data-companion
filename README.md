# AMI Trap Data Processor

Desktop app for analyzing images from autonomous insect monitoring stations


## Development

### Installation

_Requires Python 3.7 or above! Use conda if you need to maintain muliple versions of Python._

#### Linux

```
python -m venv .venv
pip install -e .

trapdata
```

_or_

```
python -m venv .venv
pip install -r requirements.txt

# Launch graphical interface
python -m trapdata

# Test the whole backend pipeline
python trapdata/tests/test_pipeline.py
```

#### MacOSX

Download Kivy.app from https://github.com/kivy/kivy/releases/download/2.1.0/Kivy.dmg


Use the virtualenv included in Kivy.app


```
pushd /Applications/Kivy.app/Contents/Resources/venv/bin
source activate
source kivy_activate
popd
python -m pip install -r requirements
python -m trapdata
```
