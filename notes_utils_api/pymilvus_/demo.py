"""
pymilvus 快速使用示例
示例
"""
from pymilvus import connections
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import pymilvus
import numpy as np


def gen_embedding(dim=2, lens=768) -> list:
    arr = np.random.randn(dim, lens)
    embed_list = arr.tolist()
    return embed_list


## 创建数据库连接
print('连接', '-' * 66)
connections.connect(
    "default",
    host="127.0.0.1",
    port='19530',
)

## 查 -----------------------------------------------#
print('查寻', '-' * 66)
# 查库名
list_collections = pymilvus.utility.list_collections()
print('已有库名: {}'.format(list_collections))

## 增 -----------------------------------------------#
# 增加一个库
print('新增', '-' * 66)
# 新库名
new_collection_name = "care_test"
# 新向量表名
vector_field_name = "embedding"
# 字段-主键
field_id = FieldSchema(name="id", dtype=DataType.INT64, description="embedding id", is_primary=True)
# 字段-向量
field_embedding_dim1 = FieldSchema(name=vector_field_name, dtype=DataType.FLOAT_VECTOR, dim=768)
field_embedding_dim2 = FieldSchema(name=vector_field_name, dtype=DataType.FLOAT_VECTOR, dim=768)
schema = CollectionSchema(fields=[field_id, field_embedding_dim1, field_embedding_dim2], auto_id=False,
                          description="care test using for dev")
# 执行创建
# collection = Collection(name=new_collection_name, schema=schema)
# 读取已经创建的
collection = Collection(name=new_collection_name)

# 插入一条数据
ids = [10086, 10087, 10089]
embed_1, embed_2, embed_3 = gen_embedding(), gen_embedding(), gen_embedding()
embedding_dim1 = [embed_1[0], embed_2[0], embed_3[0]]
embedding_dim2 = [embed_1[1], embed_2[1], embed_3[1]]
entities = [ids, embedding_dim1]
data = [
    {"id": 1, vector_field_name: [1.0, 2.0]},
    {"id": 2, vector_field_name: [5.0, 6.0]},
    {"id": 3, vector_field_name: [9.0, 10.0]}
]
mr = collection.insert(data)

## 删 -----------------------------------------------#


## 改 -----------------------------------------------#
