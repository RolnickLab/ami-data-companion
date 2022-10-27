# AMI Data Companion

Desktop app for analyzing images from autonomous insect monitoring stations


## Development

### Installation

_Requires Python 3.7 or above! Use conda if you need to maintain muliple versions of Python._

_`conda create -n python39 python=3.9 anaconda`_

#### Linux

Clone repo & create virtualenv
```
git clone git@github.com:mihow/trapdata.git
python -m venv .venv
source .venv/bin/activate
```

Install as an editable package
```
pip install -e .
trapdata
```

_Or_ install and run as source

```
pip install -r requirements.txt
python -m trapdata
```

Test the whole backend pipeline without the GUI
```
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
python -m pip install -r requirements.txt
python -m trapdata
```
