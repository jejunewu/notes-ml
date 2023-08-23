from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import pymilvus
from utils import *

connections.connect(
    "default",
    host="127.0.0.1",
    port='19530',
)

# 删除 collection
collection = Collection(name=collection_name)
collection.drop()
