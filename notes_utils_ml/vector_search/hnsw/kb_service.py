from typing import Optional, Sequence
from hnswlib import Index
import hnswlib
import numpy as np
import time
import pickle
import os


class KnowledgeBaseService:
    def __init__(
            self,
            kb_index_file: Optional[str] = 'kb_index.bin',
            kb_text_file: Optional[str] = 'kb_text.bin',
            **kwargs
    ) -> None:
        """
        Args:
            kb_index_file: 保存的索引文件
            kb_text_file: 保存文档文件
            kwargs:
        """
        self.kb_index_file = kb_index_file
        self.kb_text_file = kb_text_file

        # init kb_index
        self.kb_index = self._init_kb_index(**kwargs)
        if os.path.exists(self.kb_index_file):
            self.kb_index.load_index(self.kb_index_file)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(self.kb_index_file)), exist_ok=True)

        # init kb_text
        self.kb_text = dict()
        if os.path.exists(self.kb_text_file):
            with open(self.kb_text_file, "rb") as file:
                self.kb_text = pickle.load(file)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(self.kb_text_file)), exist_ok=True)

    def _init_kb_index(
            self,
            space: str = 'l2',
            dim: int = 768,
            max_elements: int = 10000,
            ef_construction: int = 200,
            M: int = 16,
            ef: int = 100,
            num_threads: int = 4,
            **kwargs
    ) -> Index:
        """

        初始化 kb_index

        Args:
            space: 向量空间，可选`l2`, `cosine` or `ip`
            dim: 向量维度
            max_elements: 最大数据量
            ef_construction:  controls index search speed/build speed tradeoff
            M: M is tightly connected with internal dimensionality of the data.
                Strongly affects memory consumption (~M)
                Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
            ef: 用于控制召回精度，higher ef leads to better accuracy, but slower search
            num_threads: 创建/召回线程数

        Returns:
            初始化的 hnswlib-index
        """
        self.max_elements = max_elements
        self.dim = dim
        kb_index = hnswlib.Index(space=space, dim=dim, **kwargs)
        kb_index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        kb_index.set_ef(ef=ef)
        kb_index.set_num_threads(num_threads=num_threads)
        return kb_index

    def insert_kb(
            self,
            vector: Sequence,
            text: Sequence,
            ids: Optional[Sequence] = None,
            num_threads: int = -1,
            replace_deleted: bool = False
    ) -> None:
        """

        插入数据

        Args:
            vector: 向量和文本, shape -> [batch, dim]
            text: vector向量对应的文本, shape -> [batch, ]
            ids: 索引id，不写默认按时间方式索引
            num_threads:
            replace_deleted:

        """
        assert len(vector) == len(text), \
            "The first parameter `vector` and the second parameter `text` should have the same length. " \
            "And the order corresponds to each other."
        if ids is None:
            ids = np.arange(len(vector)) + int(time.time() * (10 ** int(len(str(self.max_elements)))))
        self.kb_index.add_items(vector, ids=ids, num_threads=num_threads, replace_deleted=replace_deleted)
        self.kb_index.save_index(self.kb_index_file)
        self.kb_text.update({_id: _item for _id, _item in zip(ids, text)})
        with open(self.kb_text_file, 'wb') as file:
            pickle.dump(self.kb_text, file)

    def query_kb(
            self,
            vector: Sequence,
            k=1,
            num_threads=-1
    ) -> dict:
        """

        向量知识库查询

        Args:
            vector: 查询向量，shape -> (batch, dim)
            k:
            num_threads:
        Return:
            result: {
                        "id": id 向量索引,
                        "vector": id对应的向量
                        "distance": 距离arr
                        "text": 形状[[result_text batch_0], [result_text batch_1], ...], 其中[result_text batch_0]长度为k
                    }

        """
        result_ids, result_distances = self.kb_index.knn_query(vector, k=k, num_threads=num_threads)
        result_vectors = self.kb_index.get_items(result_ids)
        result_text = [[self.kb_text[_id] for _id in batch] for batch in result_ids]
        return {
            "id": result_ids,
            "vector": result_vectors,
            "distance": result_distances,
            "text": result_text
        }

    def drop_kb(
            self,
            kb_index_file: Optional[str] = None,
            kb_text_file: Optional[str] = None
    ) -> None:
        """删除知识库"""
        kb_index_file = kb_index_file or self.kb_index_file
        kb_text_file = kb_text_file or self.kb_text_file
        os.remove(kb_index_file) if os.path.exists(kb_index_file) else None
        os.remove(kb_text_file) if os.path.exists(kb_text_file) else None

    def get_kb_details(self) -> dict:
        return {
            "kb_index": os.path.abspath(self.kb_index_file),
            "kb_index_last_modified": time.strftime("%Y-%m-%d %H:%M:%S",
                                                    time.localtime(os.path.getmtime(self.kb_index_file))),
            "kb_text": os.path.abspath(self.kb_text_file),
            "kb_text_last_modified": time.strftime("%Y-%m-%d %H:%M:%S",
                                                   time.localtime(os.path.getmtime(self.kb_text_file))),
            "kb_dim": self.dim,
            "kb_now_elements": len(self.kb_text),
            "kb_max_elements": self.max_elements,
        }


if __name__ == '__main__':
    # np.random.seed(666)
    dim = 768
    num_elements = 10000


    def gen_sample_data(num=5000, dim=768):
        data = np.float32(np.random.random((num, dim)))
        return data


    vector_insert = gen_sample_data()
    text_insert = [f"你好啊！id={i}" for i in range(vector_insert.shape[0])]
    print("data insert:", vector_insert.shape)
    print(len(text_insert))

    kbs = KnowledgeBaseService(max_elements=999999, M=16, ef_construction=100, dim=768)
    kbs.drop_kb()
    kbs.insert_kb(vector_insert, text_insert)

    data_query = np.random.random((1, 786))
    print("data_query:", data_query.shape)
    info = kbs.query_kb(data_query, k=1)
    print(info['text'])
    details = kbs.get_kb_details()
    print(details)
    # kbs.drop_kb()
    # print(labels)
    # print(distances)
