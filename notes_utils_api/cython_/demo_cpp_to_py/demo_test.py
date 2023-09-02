import ctypes
dll = ctypes.cdll.LoadLibrary
lib = dll('./cpp_add.so')  #刚刚生成的库文件的路径
res = lib.add(1, 3)
print(res)