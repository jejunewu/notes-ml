# encoding=utf-8

"""
基于 lad

"""
# from langchain.d
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Milvus

## 载入embedding-hf模型
# 输出长度是 1024
model_large_dir = r'/home/junjie/data/model/huggingface/bge-large-zh'
# 输出长度是 768
model_base_dir = r'/home/junjie/data/model/huggingface/bge-base-zh'
# model_name = "BAAI/bge-small-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_base_dir,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

texts = [
    "余弦相似性通过测量两个向量的夹角的余弦值来度量它们之间的相似性。0度角的余弦值是1，而其他任何角度的余弦值都不大于1；并且其最小值是-1。",
    "从而两个向量之间的角度的余弦值确定两个向量是否大致指向相同的方向。两个向量有相同的指向时，余弦相似度的值为1；两个向量夹角为90°时，余弦相似度的值为0；",
    "两个向量指向完全相反的方向时，余弦相似度的值为-1。这结果是与向量的长度无关的，仅仅与向量的指向方向相关。余弦相似度通常用于正空间，因此给出的值为-1到1之间。",
]
## txts -> docs
from langchain.schema.document import Document
docs = [Document(page_content=txt, metadata={"source": "local"}) for txt in texts]

## make embed
# vectors = embedding.embed_documents(docs)
# print(vectors)
# print(len(vectors[0]))
vector_stores = Milvus.from_documents(docs, embedding)
print(vector_stores)
