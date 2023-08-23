from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from utils import *

## 创建数据库连接
print('连接', '-' * 66)
connections.connect(
    "default",
    host="127.0.0.1",
    port='19530',
)

n_data_insert = 999
collection = Collection(name=collection_name)
data = [
    [10086 + i for i in range(n_data_insert)],
    [gen_embedding() for i in range(n_data_insert)]
]
mr = collection.insert(data)
print(mr)

## 创建相似性索引
index_param = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index(field_name=field_embedding, index_params=index_param)
print(collection.index().params)
