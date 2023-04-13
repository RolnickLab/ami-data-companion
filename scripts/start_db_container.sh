#! /bin/sh

set -o errexit
set -o nounset

CONTAINER_NAME=ami-db
HOST_PORT=5432
POSTGRES_VERSION=14
POSTGRES_DB=ami

docker run -d -i --name $CONTAINER_NAME -v "$(pwd)/db_data":/var/lib/postgresql/data --restart always -p $HOST_PORT:5432 -e POSTGRES_HOST_AUTH_METHOD=trust -e POSTGRES_DB=$POSTGRES_DB postgres:$POSTGRES_VERSION

docker logs ami-db --tail 100

echo 'Database started, Connection string: "postgresql://postgres@localhost:5432/ami"'
echo "Stop (and destroy) database with 'docker stop $CONTAINER_NAME' && docker remove $CONTAINER_NAME"
