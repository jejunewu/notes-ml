## 使用c++编写扩py展



## 编译静态库

```shell
# 编译一个静态库
g++ -c mathfunlib.cpp -I.
# 创建一个新的静态库并向其中添加文件
ar -crv libmathfunlib.a mathfunlib.o
# -L是库搜索路径，-l后跟要链接的库名
g++ testLib.cpp -L. -lmathfunlib -I.
```


## 构建py动态库
```shell
python setup.py build_ext --inplace
```

## ref
- https://zhuanlan.zhihu.com/p/348417688