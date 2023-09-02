import ctypes
dll = ctypes.cdll.LoadLibrary
lib = dll('./main.so')  #刚刚生成的库文件的路径
res = lib.func(1, 3)
print(res)