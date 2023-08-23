from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from utils import *

connections.connect(
    "default",
    host="127.0.0.1",
    port='19530',
)

# 增加一个库
print('新增', '-' * 66)
dim = 2
lens = 768
# 字段-主键
field_id = FieldSchema(name=field_id, dtype=DataType.INT64, description="embedding id", is_primary=True)
# 字段-向量
field_embedding = FieldSchema(name=field_embedding, dtype=DataType.FLOAT_VECTOR, dim=dim * lens)
schema = CollectionSchema(fields=[field_id, field_embedding], auto_id=False,
                          description="care test using for dev")

# 执行创建
collection = Collection(name=collection_name, schema=schema)
# 读取已经创建的
# collection = Collection(name=new_collection_name)
