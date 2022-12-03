from flaskwebgui import FlaskUI
from base import app

FlaskUI(app=app, width=600, height=600, server="fastapi").run()
