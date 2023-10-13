from trapdata import logger
from trapdata.api.factory import create_app

app = create_app()


def run():
    import uvicorn

    logger.info("Starting uvicorn in reload mode")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        reload=True,
        port=int("8000"),
    )


if __name__ == "__main__":
    run()
