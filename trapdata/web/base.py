import os
import pathlib

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from trapdata.db import get_session_class
from trapdata.db.models.detections import get_objects_for_species, get_unique_species

db_path = os.getenv("DATABASE_URL")
DatabaseSession = get_session_class(db_path)
DB = DatabaseSession()


app = FastAPI()

app_root = pathlib.Path(__file__).parent
crops_root = "/home/michael/Projects/AMI/USERDATA/crops/"


app.mount("/static", StaticFiles(directory=app_root / "static"), name="static")
app.mount("/crops", StaticFiles(directory=crops_root), name="crops")

templates = Jinja2Templates(directory=app_root / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/detections", response_class=HTMLResponse)
async def detections(request: Request, limit: int = 2, offset: int = 0):
    species = get_unique_species(db_path)[offset : offset + limit]

    def objects_for_species(species):
        return get_objects_for_species(db_path, species)

    def relative_path(path):
        return str(pathlib.Path(path).relative_to(crops_root))

    return templates.TemplateResponse(
        "detections.html",
        {
            "request": request,
            "species_list": species,
            "get_objects": objects_for_species,
            "relative_path": relative_path,
        },
    )


@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse("item.html", {"request": request, "id": id})
