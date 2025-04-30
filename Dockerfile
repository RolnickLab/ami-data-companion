FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==2.1

# Copy only the pyproject.toml and poetry.lock files first
COPY pyproject.toml poetry.lock ./

# Configure Poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction

# Copy the rest of the project files
COPY . /app/

# Create models directory
RUN mkdir -p /app/models

# Expose the FastAPI port
EXPOSE 2000

# Set environment variable for auto-reload (default: false)
ENV UVICORN_RELOAD=false

# Command to run the FastAPI application
CMD uvicorn trapdata.api.api:app --host 0.0.0.0 --port 2000 $([ "$UVICORN_RELOAD" = "true" ] && echo "--reload")
