import os
import pathlib
import urllib.parse

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlalchemy import select

from trapdata.db import get_session_class
from trapdata.db.models.detections import get_objects_for_species, get_unique_species
from trapdata.db.models.events import MonitoringSession

# @TODO use pydantic settings module
db_path = os.getenv("DATABASE_URL")
assert db_path
user_data_path = os.getenv("USER_DATA_DIR")
assert user_data_path
DatabaseSession = get_session_class(db_path)


app = FastAPI()

app_root = pathlib.Path(__file__).parent
crops_root = pathlib.Path(user_data_path) / "crops"
if not crops_root.exists():
    crops_root.mkdir()


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

    next_url = request.url.include_query_params(limit=limit, offset=offset + limit)

    return templates.TemplateResponse(
        "detections.html",
        {
            "request": request,
            "species_list": species,
            "get_objects": objects_for_species,
            "relative_path": relative_path,
            "next_url": next_url,
        },
    )


@app.get("/surveys", response_class=HTMLResponse)
async def surveys(request: Request):
    with DatabaseSession() as session:
        surveys = session.execute(select(MonitoringSession)).scalars()

        # The location image dir is different for each deployment
        # Need a solution for this. Some might be on S3
        base_dir = session.execute(select(MonitoringSession.base_directory)).scalar()
        app.mount(
            "/trapdata/",
            StaticFiles(directory=base_dir),
            name="trapdata",
        )

        return templates.TemplateResponse(
            "surveys.html",
            {
                "request": request,
                "surveys": surveys,
                "session": session,
            },
        )


@app.get("/items/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse("item.html", {"request": request, "id": id})
