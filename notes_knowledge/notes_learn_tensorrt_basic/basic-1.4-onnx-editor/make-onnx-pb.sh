#!/bin/bash

PROTOC_PATH=/aidata/junjie/soft/envs/python39/lib/python3.9/site-packages/torch/bin/protoc
# 请修改protoc为你要使用的版本protoc
#export LD_LIBRARY_PATH=${@NVLIB64}
protoc=${PROTOC_PATH}

rm -rf pbout
mkdir -p pbout

$protoc onnx-ml.proto --cpp_out=pbout --python_out=pbout