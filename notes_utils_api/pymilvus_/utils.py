import numpy as np


def gen_embedding(dim=2, lens=768) -> list:
    arr = np.random.randn(dim, lens).flatten()
    embed_list = arr.tolist()
    return embed_list


# embeding params
embeding_len = 2 * 768

# milvus params
collection_name = "care_test"
field_id = "id"
field_embedding = "embedding"
