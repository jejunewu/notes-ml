from distutils.core import setup, Extension


def main():
    setup(name="fastaddition",
          version="2.5",
          description="The func of kun",
          author="iKun",
          author_email="ikun@kun.com",
          ext_modules=[
              Extension(
                  "fastaddition",  # 注意这个地方要和模块初始化的函数名对应
                  sources=["pyInterface.cpp"],
                  # 头文件目录，这里填当前目录即可
                  include_dirs=['.', ],
                  libraries=['mathfunlib'],
                  # '库文件目录，这里填当前目录即可'
                  library_dirs=['.', ],
                  language='c++',
                  extra_compile_args=['-std=c++11']
              )
          ]
          )


if __name__ == "__main__":
    main()
