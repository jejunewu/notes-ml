cmake_minimum_required(VERSION 3.23)
project(notes_learn_tensorrt_basic)

set(CMAKE_CXX_STANDARD 14)

file(MAKE_DIRECTORY bin)

#set (CMAKE_RUNTIME_OUTPUT_DIRECTORY bin/)

# if compiler is Clang, link libc++, otherwise, link default libstdc++
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -fsanitize=signed-integer-overflow")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
endif ()


## CUDA
set(CUDA_DIR /usr/local/cuda/)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIB64})


## TensorRT
set(TensorRT_DIR /aidata/junjie/soft/TensorRT-8.6.1.6)
include_directories(${TensorRT_DIR}/include)
link_directories(${TensorRT_DIR}/lib)


file(GLOB files src/*.cpp)
foreach (file ${files})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" exe ${file})
    add_executable(${exe} ${file})
    message(\ \ \ \ --\ src/${exe}.cpp\ will\ be\ compiled\ to\ bin/${exe})
endforeach ()

file(GLOB files concurrency/*.cpp)
foreach (file ${files})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" exe ${file})
    add_executable(${exe} ${file})
    target_link_libraries(${exe} pthread)
    message(\ \ \ \ --\ src/${exe}.cpp\ will\ be\ compiled\ to\ bin/${exe})
endforeach ()

#add_subdirectory(basic-1.1-hello)
#add_subdirectory(basic-1.2-hello-inference)
#add_subdirectory(basic-1.3-cnn-and-dynamic-shape)
#add_subdirectory(basic-1.5-onnx-parser)
add_subdirectory(basic-1.6-onnxparser-source)

target_link_libraries(main ${CUDA_LIBRARIES} nvinfer nvonnxparser)