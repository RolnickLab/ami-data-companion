import enum
import pathlib
import time

import sqlalchemy as sa
import sqlalchemy.orm as orm
import torch

dbpath = pathlib.Path("queue_test.db")
engine = sa.create_engine(f"sqlite:///{dbpath}")
Base = orm.declarative_base()

Session = orm.sessionmaker(engine)


class Status(enum.Enum):
    waiting = 1
    processing = 2
    done = 3


class Sample(Base):
    __tablename__ = "samples"
    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String)
    status = sa.Column(sa.Enum(Status))
    worker = sa.Column(sa.String)

    def __repr__(self):
        return f"<Sample {self.id} {self.name} is {self.status}>"


def create_db():
    if dbpath.exists():
        dbpath.unlink()

    Base.metadata.create_all(engine)


def populate_queue():
    with Session() as sesh:
        samples = [Sample(name=n, status=Status.waiting) for n in "abcdefg"]
        sesh.add_all(samples)
        sesh.commit()


def show_all():
    session = Session()
    stmt = sa.select(Sample)
    result = session.scalars(stmt).all()
    print(result)
    return result


def queue_count():
    session = Session()
    count = session.execute(
        sa.select(sa.func.count(Sample.id)).where(Sample.status == Status.waiting)
    ).scalar_one()
    print("Samples in queue:", count)
    return count


def get_one_from_queue():
    with Session() as sesh:
        sample = sesh.scalars(
            sa.select(Sample).where(Sample.status == Status.waiting)
        ).first()
        sample.status = Status.processing
        print("Pulled sample:", sample)
        sesh.commit()
        return sample


def pull_n_from_queue(n):
    # Pull from queue and immediately update the status
    # so no other worker will pull the same record.
    # requires used of the RETURNING sql method that
    # is only supported in sqlite >= 3.35 and SqlAlchemy >= 2.0
    select_stmt = (
        sa.select(Sample.id)
        .where((Sample.status == Status.waiting))
        .limit(n)
        .with_for_update()
    )
    with Session() as sesh:
        # sample_ids = sesh.execute(select_stmt).scalars().all()
        update_stmt = (
            sa.update(Sample)
            .where(Sample.id.in_(select_stmt.scalar_subquery()))
            # .where(Sample.status == Status.waiting)
            .values({"status": Status.processing})
            .returning(Sample.id)
        )
        # print(update_stmt)
        sample_ids = sesh.execute(update_stmt).scalars().all()
        sesh.commit()
    return sample_ids


# with engine.connect() as con:
#     names = [{"name": n, "queued": True} for n in "abcdefghijklmnop"]
#     results = con.execute(samples.insert(), names)
#
#     results = con.execute(samples.select().where(samples.c.queued == True))
#     print(results.fetchone())
#
#     results = con.execute(samples.select().where(samples.c.queued == True))
#     print(len(results))
#
#     results = con.execute(
#         samples.update().where(samples.c.id == 1).values(queued=False)
#     )
#     results = con.execute(samples.select())
#     print(results.fetchall())
#


class DatabaseDataset(torch.utils.data.IterableDataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.batch_size = 2

    def __len__(self):
        return queue_count()

    def __iter__(self):
        while len(self):
            worker_info = torch.utils.data.get_worker_info()
            # print("Worker info:", worker_info.id if worker_info else None)
            if worker_info:
                worker = worker_info.id
            else:
                worker = 1

            print("Using worker:", worker)
            # Add some jiggle to the workers to help prevent fetching the same records.
            # Ideally we could use UPDATE & OUTPUT as described here: https://towardsdatascience.com/sql-update-select-in-one-query-b067a7e60136
            # But it SQLite doesn't support it.
            # Probably better to have a local queue using queue.Queue module for the desktop app that uses multiproccessing
            # And then a server-side queue that uses PostgreSQL and the Update & Output or Redis.
            # *update*!
            # RETURNING seems to be supported! but needs sqlalchemy 2.0 beta
            # See https://github.com/litements/litequeue/blob/main/litequeue.py for details
            # time.sleep(worker * 1)
            time.sleep(1)  # simulate some work
            yield pull_n_from_queue(self.batch_size), worker


def collate(batch):
    print


def test_queries():
    show_all()
    queue_count()
    get_one_from_queue()
    queue_count()
    pull_n_from_queue(2, worker_id=1)
    queue_count()


def run():
    dataset = DatabaseDataset()
    print("Dataset length", len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=3,
        batch_size=None,
        batch_sampler=None,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )

    for i, (batch_data, worker) in enumerate(dataloader):
        print("batch num:", i, "worker:", worker, "data:", batch_data)


'''
# https://discuss.pytorch.org/t/dataloader-and-postgres-or-other-sql-an-option/25927/7

def __getitem__(self, idx: int) -> int:
    return idx

def collate(self, results: list) -> tuple:
    query = """SELECT label, feature FROM dataset WHERE index IN %s"""
    result = None
    conn = self.conn_pool.getconn()

    try:
        conn.set_session(readonly=True)
        cursor = conn.cursor()
        cursor.execute(query, tuple(results,))
        result = cursor.fetchall()

    except Error as conn_pool_error:
        print(conn_pool_error)

    finally:
        self.conn_pool.putconn(conn)
   #format batches here
   #format batches here

    if result is not None:
        return result
    else:
        throw Exception('problem in obtaining batches')

'''

if __name__ == "__main__":
    create_db()
    populate_queue()
    # test_queries()
    run()
