from transformers import AutoTokenizer, AutoModel
import torch

# Load model from HuggingFace Hub
# 输出长度是 1024
model_large_dir = r'/home/junjie/data/model/huggingface/bge-large-zh'
# 输出长度是 768
model_base_dir = r'/home/junjie/data/model/huggingface/bge-base-zh'
tokenizer = AutoTokenizer.from_pretrained(model_base_dir)
model = AutoModel.from_pretrained(model_base_dir)

# Tokenize sentences
# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# 对于短查询到长文档的检索任务, 为查询加上指令
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]

# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", sentence_embeddings)
print("Shape:", sentence_embeddings.shape)
