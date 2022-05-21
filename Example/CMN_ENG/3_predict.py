import torch

from NeuralNetwork.Seq2Seq import Seq2SeqEncoder, Seq2SeqDecoder, EncoderDecoder
from config import *

"""
读取vocab
"""
en_vocab = torch.load(os.path.join(vocab_path, "en_vocab"))
zh_vocab = torch.load(os.path.join(vocab_path, "zh_vocab"))

encoder = Seq2SeqEncoder(
    vocab_size=len(en_vocab),
    embed_size=EMBED_SIZE,
    num_hiddens=NUM_HIDDENS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)
decoder = Seq2SeqDecoder(
    vocab_size=len(zh_vocab),
    embed_size=EMBED_SIZE,
    num_hiddens=NUM_HIDDENS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

net = EncoderDecoder(encoder=encoder, decoder=decoder).to(DEVICE)
net_params = torch.load(os.path.join(model_path, 'seq2seq_params.mdl'))
net.load_state_dict(net_params)

en_setence = 'Hello !'
en_setence_idx = torch.tensor([[en_vocab.stoi[token] for token in en_setence if token in en_vocab.stoi]], device=DEVICE)

zh_idx, _ = net(en_setence_idx, en_setence_idx)
print(zh_idx)
zh_idx = torch.argmax(zh_idx, dim=2).squeeze()

print(zh_idx)
zh = [zh_vocab.itos[idx] for idx in zh_idx]
print(zh)
