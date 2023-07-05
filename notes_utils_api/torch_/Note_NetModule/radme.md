### 时序卷积网络（Temporal Convolutional Network，TCN）

```:python
TemporalConvNet(num_inputs, num_channels)
num_inputs:输入特征长度，int
num_channels:输入通道，list
```

输入数据格式：tensor->(seq_size, feature_size, batch_size, )
X: (1, time_step, -1)
Y: (1, 1, -1)

## Transformer

```python
# 输入特征长度
enc_seq_len = time_step
dec_seq_len = 2
# 输出特征长度（单点为1）
output_sequence_length = 1

dim_val = 10
dim_attn = 5

n_heads = 3
n_decoder_layers = 3
n_encoder_layers = 3
batch_size = 10

model = Transformer(dim_val, dim_attn, 1, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers,
                    n_heads)
```

输入数据格式：tensor->(batch_size, feature_size, seq_size)
X: (-1, time_step, 1)
Y: (-1, 1)

## wutorchkeras

torchtrainer

更新时间：21.12.27
