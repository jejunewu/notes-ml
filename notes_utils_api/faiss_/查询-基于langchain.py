from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

model_dir = "/workspace/models/bge-base-zh"
faiss_file_dir = r'/aidata/junjie/repo/github/Langchain-Chatchat/knowledge_base/samples/vector_store'


embeddings = HuggingFaceEmbeddings(model_name=model_dir, model_kwargs={'device': 'cuda'})
search_index = FAISS.load_local(faiss_file_dir, embeddings, normalize_L2=True)

query = "什么是科技创新？"
top_k = 5
docs = search_index.similarity_search_with_score(query, k=top_k, score_threshold=1)
print(docs)
