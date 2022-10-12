import copy

import torch


fname = "dataset.txt"

with open(fname, "w") as f:
    for i in range(0, 100):
        f.write(i)


class DatabaseDataset(torch.utils.data.IterableDataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.batch_size = 10

    def __len__(self):
        return len(open(fname).readlines())

    def __iter__(self):

        while len(self):
            worker_info = torch.utils.data.get_worker_info()
            # print("Worker info:", worker_info.id if worker_info else None)
            if worker_info:
                worker = worker_info.id
            else:
                worker = 1
            item = copy.copy(DB[0 : self.batch_size])
            # print("got item", item)
            del DB[0 : self.batch_size]
            # print("remaining:", len(DB))
            yield item, worker


def collate(batch):
    print


dataset = DatabaseDataset()
print("db length", len(dataset))

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=2,
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
