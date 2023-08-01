//
// Created by junjie on 8/1/23.
//

// tensorRT include
#include "NvInfer.h"


// onnx解析器头文件
#include "NvOnnxParser.h"

//推理运行时的头文件
#include "NvInferRuntime.h"

// cuda include
#include "cuda_runtime.h"

//sys include
#include "stdio.h"
#include "math.h"
#include "iostream"
#include "fstream"
#include "vector"

using namespace std;

// ------------------------------ 1. Logger -------------------------------------------------//
inline const char *severity_string(nvinfer1::ILogger::Severity t) {
    switch (t) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:
            return "error";
        case nvinfer1::ILogger::Severity::kWARNING:
            return "warning";
        case nvinfer1::ILogger::Severity::kINFO:
            return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE:
            return "verbose";
        default:
            return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger {
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override {
        if (severity <= Severity::kINFO) {
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if (severity == Severity::kWARNING) {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            } else if (severity <= Severity::kERROR) {
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            } else {
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;

//------------------------------------------------------------------------------------------------//

//------------------------------------------- 构建模型函数 onnx读取 --> trt模型 --------------------------------------------//
bool build_model() {
    TRTLogger logger;

    //这是基本需要的组件
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    //onnxparse 解析器
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile("demo.onnx", 1)) {
        printf("Failed to parse demo.onnx\n");
        return false;
    }

    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB \n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    // 构建模型的输入，porfile提供多个输入
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    int input_channel = input_tensor->getDimensions().d[1];

    //配置输入的最小、最优、最大的维度
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(maxBatchSize, input_channel, 5, 5));
    //将profile添加到config中
    config->addOptimizationProfile(profile);

    //加载到TRT模型
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr) {
        printf("Builder engine failed.\n");
        return false;
    }

    // 将模型序列化，存为文件
    nvinfer1::IHostMemory *model_data = engine->serialize();
    FILE *f = fopen("engine.trtmodel", "wb");
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    parser->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return true;
}
//------------------------------------------------------------------------------------------------//

//------------------------------------------- 构建TRT模型推理函数 --------------------------------------------//
vector<unsigned char> load_trt_model(const string &file) {
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open()) {
        return {};
    }
    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, ios::beg);
        data.resize(length);
        in.read((char *) &data[0], length);
    }
    in.close();
    return data;
}

//void inference() {
//    TRTLogger logger;
//    auto engine_data = load
//}

//------------------------------------------------------------------------------------------------//




int main() {
    if (!build_model()) {
        return -1;
    }
}
