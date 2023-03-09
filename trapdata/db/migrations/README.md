# Generic single-database configuration using Alembic

For a helpful guide see https://pawamoy.github.io/posts/testing-fastapi-ormar-alembic-apps/

Migrations are automatically run when the app or CLI startup as part of `trapdata.db.base.get_db()`

## Usage

Run the following commands from the package root (where `alembic.ini` lives)

Check if the database needs to be migrated according to any changes in the models
`alembic check`

Generate a new migration
`alembic revision --autogenerate -m "New results column for Events"`

Run the migration
`alembic upgrade head` # head will migrate up to the latest revision 


