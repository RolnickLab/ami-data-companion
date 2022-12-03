#! /bin/bash

set -a
source .env # this only needs to export variables for a single script
set +a
uvicorn trapdata.web.base:app --reload
