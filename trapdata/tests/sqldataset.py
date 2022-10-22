import enum
import time

import sqlalchemy as sa
import sqlalchemy.orm as orm
import torch

engine = sa.create_engine("sqlite:///saveme.db")
meta = sa.MetaData()
Base = orm.declarative_base()


class Status(enum.Enum):
    waiting = 1
    processing = 2
    done = 3


class Sample(Base):
    __tablename__ = "samples"
    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String)
    status = sa.Column(sa.Enum(Status))

    def __repr__(self):
        return f"<Sample {self.id} {self.name} is {self.status}>"


Base.metadata.create_all(engine)


def populate_queue():
    with orm.Session(engine) as sesh:
        samples = [Sample(name=n, status=Status.waiting) for n in "abcdefg"]
        sesh.add_all(samples)
        sesh.commit()


def show_all():
    session = orm.Session(engine)
    stmt = sa.select(Sample)
    result = session.scalars(stmt).all()
    print(result)
    return result


def queue_count():
    session = orm.Session(engine)
    count = session.execute(
        sa.select(sa.func.count(Sample.id)).where(Sample.status == Status.waiting)
    ).scalar_one()
    print("Samples in queue:", count)
    return count


def get_one_from_queue():
    with orm.Session(engine) as sesh:
        sample = sesh.scalars(
            sa.select(Sample).where(Sample.status == Status.waiting)
        ).first()
        sample.status = Status.processing
        print("Pulled sample:", sample)
        sesh.commit()
        return sample


def get_n_from_queue(n, offset):
    with orm.Session(engine, expire_on_commit=True) as sesh:
        samples = sesh.scalars(
            sa.select(Sample)
            .where(Sample.status == Status.waiting)
            .limit(n)
            .offset(offset)
        ).all()
        print(f"Pulled {len(samples)} samples")
        for sample in samples:
            sample.status = Status.processing
        sample_ids = [sample.id for sample in samples]
        sesh.commit()

    session = orm.Session(engine)
    stmt = sa.select(Sample).where(Sample.id.in_(sample_ids))
    result = session.scalars(stmt).all()
    return result


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
            time.sleep(1)
            yield get_n_from_queue(self.batch_size, offset=None), worker


def collate(batch):
    print


def test_queries():
    show_all()
    queue_count()
    get_one_from_queue()
    queue_count()
    get_n_from_queue(2)
    queue_count()


populate_queue()

dataset = DatabaseDataset()
print("Dataset length", len(dataset))

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=3,
    batch_size=None,
    batch_sampler=None,
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
