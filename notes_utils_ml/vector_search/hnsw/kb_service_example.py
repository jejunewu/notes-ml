import json
import time

from notes_utils_ml.text_embedding.bge_hf import BgeEmbedding
from notes_utils_ml.vector_search.hnsw.kb_service import KnowledgeBaseService

start_time = time.time()
# 拉数据
with open('data_qa.jsonl', 'r') as f:
    text = [json.loads(line) for line in f]
queries = [item['query'] for item in text]
answers = [item['answer'] for item in text]

# ebd
bge_ebd = BgeEmbedding('large', device='cpu')
# queries_ebds = bge_ebd.embedding(queries)
ebd_time = time.time()
print('finish ebd', ebd_time - start_time)

# insert kb
kbs = KnowledgeBaseService(dim=1024)
# kbs.drop_kb()
# kbs.insert_kb(queries_ebds, text)
insert_time = time.time()
# print('finish insert', insert_time - ebd_time)

# search kb
query_test = queries[697]
query_test = "飞利浦好不好？"
query_test_ebd = bge_ebd.embedding([query_test])
res = kbs.query_kb(query_test_ebd)
search_time = time.time()
print('finish search', search_time - insert_time)
print(query_test)
print(res['text'])

info = kbs.get_kb_details()
print(info)
