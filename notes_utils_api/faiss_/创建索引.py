import faiss
import numpy as np

# 创建数据
batch = 64
dim = 768  # dimensionality of the vectors
# lens = 2048  # size of the dataset
np.random.seed(0)
embedding = np.random.random((batch, dim)).astype('float32')
print(embedding.shape)

# 创建索引
index = faiss.IndexFlatL2(dim)  # build a flat index
index.add(embedding)  # add vectors to the index

# 保存索引到本地
faiss.write_index(index, "index.faiss")

# 读取索引
index2 = faiss.read_index("index.faiss")

# 查询
top_k = 10  # number of nearest neighbors to return
query = np.random.random((1, dim)).astype('float32')
distances, indices = index2.search(query, top_k)  # search the index

print(indices)  # indices of the k nearest neighbors
print(distances)  # distances to the k nearest neighbors
