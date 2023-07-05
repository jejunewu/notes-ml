import os
from ml_utils.torchwu import try_gpu

############# 超参
# 句子最大长度
MAX_LENGTH = 20
# batch
BATCH_SIZE = 50
# LR
LR = 1e-2
# embedding_size
EMBED_SIZE = 32
NUM_HIDDENS = 32
NUM_LAYERS = 2
DROPOUT = 0.1

# gpu
DEVICE = try_gpu()

############## 路径
PROJECT_ROOT_PATH = os.path.abspath('.') + os.path.sep + os.path.join('..', '..')
CMN_ENG_PATH = os.path.join(PROJECT_ROOT_PATH, 'Example', 'CMN_ENG')
vocab_path = os.path.join(CMN_ENG_PATH, 'vocab')
corpus_path = os.path.join(CMN_ENG_PATH, 'corpus')
model_path = os.path.join(CMN_ENG_PATH, 'model')