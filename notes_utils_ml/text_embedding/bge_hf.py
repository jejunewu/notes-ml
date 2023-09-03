from transformers import AutoTokenizer, AutoModel
import torch
import sys

BGE_MODEL_PATH = {
    "win32_large": r"J:\repo\huggingface\bge-large-zh",
    "linux_base": r"/home/junjie/data/model/huggingface/bge-base-zh",
}


class BgeEmbedding:
    def __init__(self, model: str = 'large', device='cpu'):
        model_dir = BGE_MODEL_PATH[f"{sys.platform}_{model}"]
        # load model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device)

    def embedding(self, queries: list, instruction: str = None):
        # 对于短查询到长文档的检索任务, 为查询加上指令
        if instruction:
            encoded_input = self.tokenizer([instruction + q for q in queries], padding=True, truncation=True,
                                           return_tensors='pt')
        else:
            encoded_input = self.tokenizer(queries, padding=True, truncation=True, return_tensors='pt')
        # 计算 embeddings
        with torch.no_grad():
            encoded_input.to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        # normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


if __name__ == '__main__':
    bge_ebd = BgeEmbedding(model='large')
    embeddings = bge_ebd.embedding(["你好吗", "我很好"])
    print(embeddings.shape)
