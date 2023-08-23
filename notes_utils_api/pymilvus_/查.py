from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import pymilvus

from utils import *

connections.connect(
    "default",
    host="127.0.0.1",
    port='19530',
)

print('-' * 88)
# 查库名
list_collections = pymilvus.utility.list_collections()
print('已有库名: {}'.format(list_collections))

print('-' * 88)
# 查字段名
collection = Collection(name=collection_name)
schema = collection.schema
print(f"{collection_name}: ", schema)
for field in schema.fields:
    print(f"Field name: {field.name}")
    print(f"Field type: {field.dtype}")
    print(f"Field description: {field.description}")

print('规则查询', '-' * 88)
# 加载到内存，很关键！！！！！！！！！！！！！！
collection.load()
print(collection.num_entities)
# 指定规则查询
result_expr = collection.query(
    expr='id > 10099 && id < 10102',
    output_fields=[field_id, field_embedding],
    consistency_level='Strong'
)
print(result_expr)

# 相似性查询向量
print('相似性查询', '-' * 88)
query_embedding = gen_embedding()
search_params = {"metric_type": "L2", "params": {"nprobe": 50}}
result = collection.search([query_embedding, ], field_embedding, search_params, limit=3)
print(result)
