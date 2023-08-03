# LearnTensorRT-Basic

https://zifuture.com/group/LearnTensorRT-Basic


## TensorRT库
- 位置：/aidata/junjie/soft/TensorRT-8.6.1.6

## 下载文件
```shell
folder=basic-1.7-hello-plugin
sub_folder=src
file=main.cpp
mkdir -p ${folder}/${sub_folder}
wget https://zifuture.com/api/public/repo/detail/trt/${folder}/${sub_folder}/${file}?download=true -O ${folder}/${sub_folder}/${file}

```


## 编译
```shell
cd build
rm -r *
cmake .. && make
./*1.7*/bin/main
```